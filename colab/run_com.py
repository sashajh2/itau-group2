#!/usr/bin/env python3
"""
The .com-RETAINED pass: Table 11 (suffix ablation) and Table 6's VA-TE row.

Table 11 is the whole of the R1-3 suffix argument and cannot be dropped. An honest
version must FIT on a .com-retained training split as well as evaluate on a
.com-retained test split, which is why this needs its own embedding cache
(embed_raw_com.py) rather than reusing the stripped one.

Same protocol as run_final.py, applied within the retained convention:
    fusion fitted on the retained train split
    thresholds selected by Youden's J on the retained validation split
    retained test split read once

Also emits the VA-TE row of Table 6, so that table and Table 11 agree on the same
quantity instead of differing by which head produced them.
"""
import argparse, json, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm.auto import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--head", required=True)
args = ap.parse_args()

DATA = "/Users/sashajh2/itau-group2/data/processed"
CACHE = "/Users/sashajh2/itau-group2/colab/vate_cache"
OUT = "/Users/sashajh2/itau-group2/colab/vate_out"
os.makedirs(OUT, exist_ok=True)
SEED = 42
GB_KW = dict(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# .com RETAINED on every split
SPLITS = {"train": "train_pairs_ref.parquet",
          "validate": "validate_pairs_ref.parquet",
          "test": "test_pairs_ref.parquet"}

print(f"device: {DEVICE}\n  head: {os.path.basename(args.head)}", flush=True)
data = {}
for k, fn in SPLITS.items():
    df = pd.read_parquet(os.path.join(DATA, fn))
    a = df["fraudulent_name"].astype(str).tolist()
    b = df["real_name"].astype(str).tolist()
    y = df["label"].values.astype(int)
    data[k] = dict(a=a, b=b, y=y)
    print(f"  {k:9s} n={len(y):>7,}  spoof={y.mean():.4f}  "
          f".com={np.mean([s.endswith('.com') for s in a[:5000]]):.2f}", flush=True)

uniques = list(np.load(os.path.join(CACHE, "com_uniques.npy"), allow_pickle=True))
pos = {s: i for i, s in enumerate(uniques)}
_expect = sorted({s for k in SPLITS for s in data[k]["a"] + data[k]["b"]})
if _expect != uniques:
    raise RuntimeError(f"com cache mismatch: cache {len(uniques):,}, splits {len(_expect):,}")
del _expect
RAW = np.load(os.path.join(CACHE, "com_emb768.npy"), mmap_mode="r")
if RAW.shape != (len(uniques), 768):
    raise RuntimeError(f"com cache shape {RAW.shape} != {(len(uniques), 768)}")
print(f"  {len(uniques):,} retained strings, cache verified", flush=True)

state = torch.load(args.head, map_location="cpu")
proj = {k.split("projector.")[-1]: v for k, v in state.items()}
head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
head.load_state_dict(proj); head.eval().to(DEVICE)
print(f"  head parameters: {sum(v.numel() for v in proj.values()):,}", flush=True)

E = np.zeros((len(uniques), 768), dtype=np.float32)
with torch.no_grad():
    for i in tqdm(range(0, len(uniques), 16384), desc="projecting"):
        blk = torch.from_numpy(np.ascontiguousarray(RAW[i:i + 16384])).to(DEVICE)
        E[i:i + 16384] = F.normalize(head(blk), dim=1).cpu().numpy()

for k in SPLITS:
    d = data[k]
    ia = np.fromiter((pos[s] for s in d["a"]), dtype=np.int64, count=len(d["a"]))
    ib = np.fromiter((pos[s] for s in d["b"]), dtype=np.int64, count=len(d["b"]))
    d["cos"] = np.einsum("ij,ij->i", E[ia], E[ib]).astype(np.float64)
    d["lev"] = np.array([Levenshtein.distance(x, y) for x, y in zip(d["a"], d["b"])], dtype=np.float64)
    d["tsr"] = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(d["a"], d["b"])], dtype=np.float64) / 100.0
    print(f"  {k:9s} features done", flush=True)


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[int(np.argmax(tpr - fpr))]


def row(name, y_val, s_val, y_te, s_te):
    t = youden(y_val, s_val)
    p = (s_te >= t).astype(int)
    tp = int(((p == 1) & (y_te == 1)).sum()); fp = int(((p == 1) & (y_te == 0)).sum())
    fn = int(((p == 0) & (y_te == 1)).sum()); tn = int(((p == 0) & (y_te == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(model=name, roc_auc=float(roc_auc_score(y_te, s_te)),
                pr_auc=float(average_precision_score(y_te, s_te)),
                accuracy=float((tp + tn) / len(y_te)), precision=float(prec), recall=float(rec),
                f1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0)


def X(k, feats):
    return np.column_stack([data[k][f] for f in feats])


def fuse(feats):
    m = GradientBoostingClassifier(**GB_KW).fit(X("train", feats), data["train"]["y"])
    return (m.predict_proba(X("validate", feats))[:, 1],
            m.predict_proba(X("test", feats))[:, 1])


yv, yt = data["validate"]["y"], data["test"]["y"]
print("\nhonest protocol, .com retained: fit on train, threshold on validation, test once",
      flush=True)
rows = []
rows.append(row("Levenshtein", yv, -data["validate"]["lev"], yt, -data["test"]["lev"]))
rows.append(row("Token Set Ratio", yv, data["validate"]["tsr"], yt, data["test"]["tsr"]))
sv, st = fuse(["tsr", "lev"])
rows.append(row("Levenshtein + Token Set Ratio", yv, sv, yt, st))
rows.append(row("VA-TE", yv, data["validate"]["cos"], yt, data["test"]["cos"]))
sv, st = fuse(["cos", "tsr", "lev"])
rows.append(row("VA-TE + String", yv, sv, yt, st))
for r in rows:
    print(f"   {r['model']:32s} ROC-AUC {r['roc_auc']:.4f}  acc {r['accuracy']:.4f}", flush=True)

vate = next(r for r in rows if r["model"] == "VA-TE")
print(f"\n{'='*70}")
print("  Table 11, .com-retained row (paste order: Lev, TokenSet, Lev+TS, VA-TE, VA-TE+String)")
print("  " + " & ".join(f"{r['roc_auc']:.4f}" for r in rows))
print(f"\n  Table 6, VA-TE row  (AUC, Acc %, Prec, Rec)")
print(f"  VA-TE (SigLIP + trained head) & {vate['roc_auc']:.3f} & {vate['accuracy']*100:.2f} "
      f"& {vate['precision']:.3f} & {vate['recall']:.3f} \\\\")
print(f"{'='*70}\n", flush=True)

json.dump({"head": os.path.basename(args.head), "rows": rows},
          open(os.path.join(OUT, "com_retained.json"), "w"), indent=2)
print(f"  saved -> {OUT}/com_retained.json", flush=True)
