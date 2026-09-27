#!/usr/bin/env python3
"""
Pass A: cache NORMALISED RAW SigLIP pooler outputs (768-d) for every string needed by
both the head retrain and the evaluation.

This is head-independent. It is the ~60-70 min cost in the pipeline, and once it exists
the 768->768->768 projection head can be retrained and evaluated in minutes on CPU,
across as many seeds as we like.

Matches run_local.py::encode_batch exactly up to (but NOT including) the projector:
    tok(padding="max_length", truncation=True) -> backbone -> pooler_output -> F.normalize
"""
import os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from tqdm.auto import tqdm

DATA_DIR  = "/Users/sashajh2/itau-group2/data/processed"
CACHE_DIR = "/Users/sashajh2/itau-group2/colab/vate_cache"
BACKBONE  = "google/siglip-base-patch16-224"
BATCH     = 512
SAVE_EVERY= 40
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(CACHE_DIR, exist_ok=True)

# eval splits (strip flag matches run_local.py SPLITS)
EVAL = {"train":("train_pairs_ref.parquet",True),
        "validate":("validate_pairs_ref.parquet",True),
        "test":("test_pairs_all.parquet",False)}
# curriculum files the projection head was trained on (already .com-stripped)
CURRIC = ["train_pairs_easy_100k.parquet",
          "train_pairs_medium_100k.parquet",
          "train_pairs_hard_100k.parquet"]

def load(fn, strip):
    df = pd.read_parquet(os.path.join(DATA_DIR, fn))
    a = df["fraudulent_name"].astype(str); b = df["real_name"].astype(str)
    if strip:
        a = a.str.removesuffix(".com"); b = b.str.removesuffix(".com")
    return a.tolist(), b.tolist()

print(f"device: {DEVICE}", flush=True)
allstr = set()
for k,(fn,st) in EVAL.items():
    a,b = load(fn,st); allstr |= set(a)|set(b)
    print(f"  {k:9s} {fn:32s} -> running union {len(allstr):,}", flush=True)
for fn in CURRIC:
    a,b = load(fn,False); allstr |= set(a)|set(b)
    print(f"  curric    {fn:32s} -> running union {len(allstr):,}", flush=True)

uniques = sorted(allstr)
N = len(uniques)
print(f"\n  TOTAL unique strings to embed: {N:,}", flush=True)
np.save(os.path.join(CACHE_DIR,"raw_uniques.npy"), np.array(uniques, dtype=object),
        allow_pickle=True)

from transformers import AutoTokenizer, SiglipTextModel
backbone = SiglipTextModel.from_pretrained(BACKBONE, torch_dtype=torch.float32).eval().to(DEVICE)
tok = AutoTokenizer.from_pretrained(BACKBONE)

emb_path = os.path.join(CACHE_DIR,"raw_emb768.npy")
idx_path = os.path.join(CACHE_DIR,"raw_done.txt")
if os.path.exists(emb_path):
    mat = np.load(emb_path, mmap_mode="r+")
    done = int(open(idx_path).read().strip()) if os.path.exists(idx_path) else 0
    if mat.shape != (N,768):
        mat = np.lib.format.open_memmap(emb_path,mode="w+",dtype=np.float32,shape=(N,768)); done=0
    else:
        print(f"  resuming at {done:,}/{N:,}", flush=True)
else:
    mat = np.lib.format.open_memmap(emb_path,mode="w+",dtype=np.float32,shape=(N,768)); done=0

@torch.no_grad()
def enc(texts):
    inp = tok(texts, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)
    return F.normalize(backbone(**inp).pooler_output, dim=1)

t0=time.time()
for n,i in enumerate(tqdm(range(done,N,BATCH), desc="raw-embed")):
    ch = uniques[i:i+BATCH]
    mat[i:i+len(ch)] = enc(ch).cpu().numpy()
    if (n+1)%SAVE_EVERY==0:
        mat.flush(); open(idx_path,"w").write(str(i+len(ch)))
mat.flush(); open(idx_path,"w").write(str(N))
el=time.time()-t0
print(f"\n  DONE: {N:,} strings in {el/60:.1f} min ({N/max(el,1):.0f}/s)", flush=True)
print(f"  -> {emb_path}", flush=True)
