#!/usr/bin/env python3
"""
Pass A2: cache normalised raw SigLIP pooler outputs for the ".com"-RETAINED strings.

Needed for Table 10 (the suffix ablation), which cannot be dropped because it is the
whole of the R1-3 suffix argument. An honest Table 10 must FIT on a .com-retained train
split as well as evaluate on a .com-retained test split, so this covers train, validate
and test in the retained convention.

Head-independent, exactly like embed_raw.py, so it can run while the head sweep finishes.
"""
import os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tqdm.auto import tqdm

DATA = "/Users/sashajh2/itau-group2/data/processed"
CACHE = "/Users/sashajh2/itau-group2/colab/vate_cache"
BACKBONE = "google/siglip-base-patch16-224"
BATCH, SAVE_EVERY = 512, 40
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(CACHE, exist_ok=True)

# .com RETAINED on all three splits. train/validate _ref already carry it on disk;
# test_pairs_ref.parquet is the retained twin of test_pairs_all.parquet (same 256,886 pairs).
SPLITS = [("train", "train_pairs_ref.parquet"),
          ("validate", "validate_pairs_ref.parquet"),
          ("test", "test_pairs_ref.parquet")]


def load(fn):
    df = pd.read_parquet(os.path.join(DATA, fn))
    return df["fraudulent_name"].astype(str).tolist(), df["real_name"].astype(str).tolist()


print(f"device: {DEV}", flush=True)
allstr = set()
for k, fn in SPLITS:
    a, b = load(fn)
    allstr |= set(a) | set(b)
    frac = np.mean([s.endswith(".com") for s in a[:5000]])
    print(f"  {k:9s} {fn:28s} .com={frac:.2f}  union {len(allstr):,}", flush=True)

uniques = sorted(allstr)
N = len(uniques)
print(f"\n  TOTAL .com-retained unique strings: {N:,}", flush=True)
np.save(os.path.join(CACHE, "com_uniques.npy"), np.array(uniques, dtype=object), allow_pickle=True)

from transformers import AutoTokenizer, SiglipTextModel
backbone = SiglipTextModel.from_pretrained(BACKBONE, torch_dtype=torch.float32).eval().to(DEV)
tok = AutoTokenizer.from_pretrained(BACKBONE)

emb_path = os.path.join(CACHE, "com_emb768.npy")
idx_path = os.path.join(CACHE, "com_done.txt")
if os.path.exists(emb_path):
    mat = np.load(emb_path, mmap_mode="r+")
    done = int(open(idx_path).read().strip()) if os.path.exists(idx_path) else 0
    if mat.shape != (N, 768):
        mat = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, 768)); done = 0
    else:
        print(f"  resuming at {done:,}/{N:,}", flush=True)
else:
    mat = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, 768)); done = 0


@torch.no_grad()
def enc(texts):
    inp = tok(texts, return_tensors="pt", padding="max_length", truncation=True).to(DEV)
    return F.normalize(backbone(**inp).pooler_output, dim=1)


t0 = time.time()
for n, i in enumerate(tqdm(range(done, N, BATCH), desc="com-embed")):
    ch = uniques[i:i+BATCH]
    mat[i:i+len(ch)] = enc(ch).cpu().numpy()
    if (n + 1) % SAVE_EVERY == 0:
        mat.flush(); open(idx_path, "w").write(str(i + len(ch)))
mat.flush(); open(idx_path, "w").write(str(N))
el = time.time() - t0
print(f"\n  DONE: {N:,} strings in {el/60:.1f} min ({N/max(el,1):.0f}/s)", flush=True)
print(f"  -> {emb_path}", flush=True)
