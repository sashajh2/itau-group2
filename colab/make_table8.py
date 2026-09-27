#!/usr/bin/env python3
"""
Recompute Table 8 (the combiner ablation) under the honest protocol.

run_local.py only refits the gradient-boosting combiner, because that is the one the
paper deploys. But Table 8 compares SEVEN combiners on the same three features, and its
caption has already been changed to say the learned rows are "fitted on the training
split and evaluated once on the held-out test split". Every learned row in that table is
still an out-of-fold number, so the caption and the body currently disagree.

This is cheap: the embeddings are already cached, so all seven combiners fit on three
scalar features in a few minutes. Run it after run_local.py.

Rows reproduced, in the paper's order:
  Weighted average (equal weights)      - no fitting
  Decision tree (depth 3)
  Weighted average (tuned on validation) - 3 weights grid-searched on validation
  Logistic regression
  Decision tree (depth 6)
  Random forest (100, depth 6)
  Gradient boosting (100, depth 6, lr 0.1)
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_DIR  = "/Users/sashajh2/itau-group2/data/processed"
CACHE_DIR = "/Users/sashajh2/itau-group2/colab/vate_cache"
OUT_DIR   = "/Users/sashajh2/itau-group2/colab/vate_out"
SEED      = 42
os.makedirs(OUT_DIR, exist_ok=True)


def load(name, strip):
    df = pd.read_parquet(os.path.join(DATA_DIR, name))
    a = df["fraudulent_name"].astype(str); b = df["real_name"].astype(str)
    if strip:
        a, b = a.str.removesuffix(".com"), b.str.removesuffix(".com")
    return a.tolist(), b.tolist(), df["label"].values.astype(int)


SPLITS = {"train": ("train_pairs_ref.parquet", True),
          "validate": ("validate_pairs_ref.parquet", True),
          "test": ("test_pairs_all.parquet", False)}
data = {k: dict(zip(("a", "b", "y"), load(f, s))) for k, (f, s) in SPLITS.items()}

# Same unique-string ordering run_local.py built, so the cache lines up.
uniques = sorted({s for k in ("train", "validate", "test")
                  for s in data[k]["a"] + data[k]["b"]})
pos = {s: i for i, s in enumerate(uniques)}
E = np.load(os.path.join(CACHE_DIR, "emb.npy"))
if E.shape[0] != len(uniques):
    raise SystemExit(f"cache has {E.shape[0]} rows, expected {len(uniques)} — rerun run_local.py")

for k in data:
    d = data[k]
    ia = np.fromiter((pos[s] for s in d["a"]), np.int64, len(d["a"]))
    ib = np.fromiter((pos[s] for s in d["b"]), np.int64, len(d["b"]))
    d["cos"] = np.einsum("ij,ij->i", E[ia], E[ib]).astype(np.float64)
    d["lev"] = np.array([Levenshtein.distance(x, y) for x, y in zip(d["a"], d["b"])], float)
    d["tsr"] = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(d["a"], d["b"])], float) / 100.
    print(f"  {k} ready")

F = ["cos", "tsr", "lev"]                       # Equation (3), in that order
X = lambda k: np.column_stack([data[k][c] for c in F])
Xtr, ytr = X("train"), data["train"]["y"]
Xva, yva = X("validate"), data["validate"]["y"]
Xte, yte = X("test"), data["test"]["y"]


def youden(y, s):
    f, t, th = roc_curve(y, s); return th[int(np.argmax(t - f))]


def fpr_at_fnr(y, s, tgt):
    f, t, _ = roc_curve(y, s); ok = np.where((1 - t) <= tgt)[0]
    return float(f[ok[0]]) if len(ok) else 1.0


def report(name, s_va, s_te):
    """Threshold on validation, metrics on test — the same rule as everywhere else."""
    th = youden(yva, s_va); p = (s_te >= th).astype(int)
    tp = int(((p == 1) & (yte == 1)).sum()); fp = int(((p == 1) & (yte == 0)).sum())
    fn = int(((p == 0) & (yte == 1)).sum()); tn = int(((p == 0) & (yte == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(model=name, roc_auc=roc_auc_score(yte, s_te),
                pr_auc=average_precision_score(yte, s_te),
                accuracy=(tp + tn) / len(yte), precision=pr, recall=rc,
                f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                fpr_at_fnr_5=fpr_at_fnr(yte, s_te, 0.05))


# --- min-max normalise for the two averaging rows, fitted on train only ---
lo, hi = Xtr.min(0), Xtr.max(0)
nrm = lambda Z: np.clip((Z - lo) / np.where(hi - lo == 0, 1, hi - lo), 0, 1)
# edit distance runs the other way: smaller is more spoof-like
flip = np.array([1.0, 1.0, -1.0])
Ntr, Nva, Nte = (nrm(Z) * flip for Z in (Xtr, Xva, Xte))

rows = []
rows.append(report("Weighted average (equal weights)", Nva.mean(1), Nte.mean(1)))

# tuned weights: grid-searched on validation, as the caption says
best, best_auc = None, -1
grid = np.arange(0, 1.05, 0.1)
for w in itertools.product(grid, repeat=3):
    if abs(sum(w) - 1.0) > 1e-9 or sum(w) == 0:
        continue
    auc = roc_auc_score(yva, Nva @ np.array(w))
    if auc > best_auc:
        best_auc, best = auc, np.array(w)
rows.append(report("Weighted average (tuned on validation)", Nva @ best, Nte @ best))
print(f"  tuned weights = {best.round(2).tolist()} (validation AUC {best_auc:.4f})")

MODELS = [
    ("Decision tree (depth 3)",  DecisionTreeClassifier(max_depth=3, random_state=SEED), False),
    ("Logistic regression",      LogisticRegression(max_iter=2000), True),
    ("Decision tree (depth 6)",  DecisionTreeClassifier(max_depth=6, random_state=SEED), False),
    ("Random forest (100, depth 6)",
     RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1), False),
    ("Gradient boosting (100, depth 6, lr 0.1)",
     GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                random_state=SEED), False),
]
for name, mdl, scale in MODELS:
    if scale:   # only the linear model needs it; trees are scale-invariant
        sc = StandardScaler().fit(Xtr)
        mdl.fit(sc.transform(Xtr), ytr)
        sv, st = (mdl.predict_proba(sc.transform(Z))[:, 1] for Z in (Xva, Xte))
    else:
        mdl.fit(Xtr, ytr)
        sv, st = (mdl.predict_proba(Z)[:, 1] for Z in (Xva, Xte))
    rows.append(report(name, sv, st))
    print(f"  fitted {name}")

ORDER = ["Weighted average (equal weights)", "Decision tree (depth 3)",
         "Weighted average (tuned on validation)", "Logistic regression",
         "Decision tree (depth 6)", "Random forest (100, depth 6)",
         "Gradient boosting (100, depth 6, lr 0.1)"]
by = {r["model"]: r for r in rows}

with open(os.path.join(OUT_DIR, "table8.json"), "w") as f:
    json.dump(rows, f, indent=2)

out = ["", "Table 8 rows, honest protocol — paste into template.tex", ""]
for m in ORDER:
    r = by[m]
    out.append(f"{m} & {r['roc_auc']:.4f} & {r['pr_auc']:.4f} & {r['accuracy']:.4f} & "
               f"{r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} & "
               f"{r['fpr_at_fnr_5']:.4f} \\\\")
out += ["", "Check before pasting: the paper bolds the gradient-boosting row as best. If some",
        "other combiner now leads, move the \\textbf{} and revise the surrounding prose at",
        "line 868, which currently reads '0.9648--0.9715' for the non-linear range."]
txt = "\n".join(out)
open(os.path.join(OUT_DIR, "table8.md"), "w").write(txt)
print(txt)
