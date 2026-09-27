#!/usr/bin/env python3
"""
Regenerate Figure 5 (roc_curves_all_methods.png) after the fusion refit.

Figure 5 draws ROC curves for all five methods in Table 7. Two of those five change
under the honest protocol, so the figure is stale the moment Table 7 is corrected —
the same defect Figure 4 already had.

run_local.py does not persist per-sample scores, only summary metrics. But it does
cache the embeddings, which is the expensive part. This script reloads that cache and
recomputes the five score vectors in a couple of minutes, then draws the figure and
saves the scores so nothing has to be recomputed again.

Run it AFTER run_local.py has finished.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

DATA_DIR  = "/Users/sashajh2/itau-group2/data/processed"
CACHE_DIR = "/Users/sashajh2/itau-group2/colab/vate_cache"
OUT_DIR   = "/Users/sashajh2/itau-group2/colab/vate_out"
OUT_PNG   = "roc_curves_all_methods.png"
SEED      = 42
GB_KW = dict(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED)

os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = {"train": ("train_pairs_ref.parquet", True),
          "test":  ("test_pairs_all.parquet",  False)}


def load(name, strip):
    df = pd.read_parquet(os.path.join(DATA_DIR, name))
    a = df["fraudulent_name"].astype(str)
    b = df["real_name"].astype(str)
    if strip:
        a, b = a.str.removesuffix(".com"), b.str.removesuffix(".com")
    return a.tolist(), b.tolist(), df["label"].values.astype(int)


data = {}
for k, (f, s) in SPLITS.items():
    a, b, y = load(f, s)
    data[k] = dict(a=a, b=b, y=y)

# Rebuild the same unique-string ordering run_local.py used, so cached rows line up.
# It included validation, so that split has to be loaded even though it is unused here.
va_a, va_b, _ = load("validate_pairs_ref.parquet", True)
uniques = sorted(set(data["train"]["a"] + data["train"]["b"] +
                     va_a + va_b +
                     data["test"]["a"] + data["test"]["b"]))
pos = {s: i for i, s in enumerate(uniques)}

E = np.load(os.path.join(CACHE_DIR, "emb.npy"))
if E.shape[0] != len(uniques):
    raise SystemExit(f"cache has {E.shape[0]} rows, expected {len(uniques)} — "
                     "the cache is from a different run, delete it and re-run run_local.py")

for k in ("train", "test"):
    ia = np.fromiter((pos[s] for s in data[k]["a"]), np.int64, len(data[k]["a"]))
    ib = np.fromiter((pos[s] for s in data[k]["b"]), np.int64, len(data[k]["b"]))
    data[k]["cos"] = np.einsum("ij,ij->i", E[ia], E[ib]).astype(np.float64)
    a, b = data[k]["a"], data[k]["b"]
    data[k]["lev"] = np.array([Levenshtein.distance(x, y) for x, y in zip(a, b)], float)
    data[k]["tsr"] = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(a, b)], float) / 100.
    print(f"  {k} features ready")

yt = data["test"]["y"]
X = lambda k, f: np.column_stack([data[k][c] for c in f])

# Two fused curves, fitted on train only — the honest protocol.
scores = {}
for label, feats in [("Levenshtein + Token Set Ratio", ["tsr", "lev"]),
                     ("VA-TE + String (full fusion)",  ["cos", "tsr", "lev"])]:
    m = GradientBoostingClassifier(**GB_KW).fit(X("train", feats), data["train"]["y"])
    scores[label] = m.predict_proba(X("test", feats))[:, 1]
    print(f"  fitted {label}")

scores["Levenshtein"] = -data["test"]["lev"]
scores["Token Set Ratio"] = data["test"]["tsr"]
scores["VA-TE (SigLIP-pair embedding similarity)"] = data["test"]["cos"]

ORDER = ["Levenshtein", "Token Set Ratio", "Levenshtein + Token Set Ratio",
         "VA-TE (SigLIP-pair embedding similarity)", "VA-TE + String (full fusion)"]

np.savez_compressed(os.path.join(OUT_DIR, "test_scores.npz"),
                    y=yt, **{k: scores[k] for k in ORDER})

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200)
for name in ORDER:
    fpr, tpr, _ = roc_curve(yt, scores[name])
    ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {roc_auc_score(yt, scores[name]):.4f})")
ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.003)
ax.legend(loc="lower right", fontsize=8, frameon=True)
ax.grid(alpha=0.25, lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, OUT_PNG), bbox_inches="tight", facecolor="white")

print(f"\nwrote {OUT_DIR}/{OUT_PNG}")
for name in ORDER:
    print(f"  {name:44s} {roc_auc_score(yt, scores[name]):.4f}")
print("\nThese AUCs must match Table 7 exactly — the caption says they are the same scores.")
