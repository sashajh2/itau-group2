#!/usr/bin/env python3
"""
Epoch-budget sweep for the retrained projection head.

The curriculum schedule is a function of the TOTAL epoch budget (utils/curriculum.py
uses t = epoch/total_epochs), so each budget is a genuinely different training run and
a long run is not a superset of a short one. This trains one head per budget, sharing
the embedding load across all of them.

SELECTION RULE. The budget is chosen on VALIDATION ROC-AUC, and that choice alone
decides the number we report. Test ROC-AUC is recorded per budget as a diagnostic --
it answers "is the published 0.9450 reachable under this recipe at all?" -- but nothing
is selected from it. Choosing the budget by test ROC-AUC would be fitting a
hyperparameter on the test split, which is the exact defect reviewers R1-1/R1-6/R2-1
raised and the reason this revision exists.

Everything else follows the published configuration (Table 12, Sections 4.3/5.4).
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--budgets", type=str, default="5,10,15,20,30,40,50,60")
args = ap.parse_args()
BUDGETS = [int(x) for x in args.budgets.split(",")]

DATA = "/Users/sashajh2/itau-group2/data/processed"
CACHE = "/Users/sashajh2/itau-group2/colab/vate_cache"
OUT = "/Users/sashajh2/itau-group2/colab/vate_out"
os.makedirs(OUT, exist_ok=True)
MARGIN, LR, WD, BATCH = 0.3364, 3.8396e-5, 1.5673e-6, 64
PUBLISHED = 0.9450
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"device {DEV} | seed {args.seed} | budgets {BUDGETS}", flush=True)

uniques = np.load(os.path.join(CACHE, "raw_uniques.npy"), allow_pickle=True)
pos = {s: i for i, s in enumerate(uniques)}
E = np.load(os.path.join(CACHE, "raw_emb768.npy"), mmap_mode="r")


def load(fn, strip):
    df = pd.read_parquet(os.path.join(DATA, fn))
    a = df["fraudulent_name"].astype(str); b = df["real_name"].astype(str)
    if strip:
        a = a.str.removesuffix(".com"); b = b.str.removesuffix(".com")
    ia = np.fromiter((pos[s] for s in a), dtype=np.int64, count=len(a))
    ib = np.fromiter((pos[s] for s in b), dtype=np.int64, count=len(b))
    return ia, ib, df["label"].values.astype(np.float32)


tiers = {n: load(f"train_pairs_{n}_100k.parquet", False) for n in ("easy", "medium", "hard")}
va = load("validate_pairs_ref.parquet", True)
te = load("test_pairs_all.parquet", False)
need = np.unique(np.concatenate([t[i] for t in list(tiers.values()) + [va, te] for i in (0, 1)]))
remap = np.full(len(uniques), -1, dtype=np.int64); remap[need] = np.arange(len(need))
EM = torch.from_numpy(np.ascontiguousarray(E[need])).to(DEV)
print(f"  resident embeddings {tuple(EM.shape)} | val {len(va[2]):,} | test {len(te[2]):,}\n", flush=True)


def ratios(ep, tot):
    t = ep / tot
    e = 0.5 * (1 + np.cos(np.pi * t)) * (1 - t)
    h = 0.5 * (1 + np.cos(np.pi * (1 - t))) * t
    m = 1.0 - (e + h); s = e + m + h
    return e / s, m / s, h / s


def cosine_loss(z1, z2, y):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    d = 1 - F.cosine_similarity(z1, z2)
    return (y * d.pow(2) + (1 - y) * F.relu(MARGIN - d).pow(2)).mean()


def make_scorer(head):
    @torch.no_grad()
    def score(ia, ib, bs=16384):
        head.eval(); out = []
        for i in range(0, len(ia), bs):
            z1 = F.normalize(head(EM[torch.from_numpy(remap[ia[i:i+bs]]).to(DEV)]), dim=1)
            z2 = F.normalize(head(EM[torch.from_numpy(remap[ib[i:i+bs]]).to(DEV)]), dim=1)
            out.append(F.cosine_similarity(z1, z2).cpu().numpy())
        head.train(); return np.concatenate(out)
    return score


def train_budget(budget, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
    score = make_scorer(head)
    halfway = (budget - 1) // 2
    best_val, best_ep, best_state = -1, -1, None
    for ep in range(budget):
        re_, rm_, rh_ = ratios(ep, budget)
        total = len(tiers["hard"][2])
        parts = []
        for name, r in (("easy", re_), ("medium", rm_), ("hard", rh_)):
            ia, ib, y = tiers[name]; n = int(r * total)
            if n > 0:
                idx = np.random.choice(len(y), n, replace=False)
                parts.append((ia[idx], ib[idx], y[idx]))
        ia = np.concatenate([p[0] for p in parts]); ib = np.concatenate([p[1] for p in parts])
        y = np.concatenate([p[2] for p in parts])
        order = np.random.permutation(len(y)); ia, ib, y = ia[order], ib[order], y[order]
        for i in range(0, len(y), BATCH):
            b1 = EM[torch.from_numpy(remap[ia[i:i+BATCH]]).to(DEV)]
            b2 = EM[torch.from_numpy(remap[ib[i:i+BATCH]]).to(DEV)]
            yy = torch.from_numpy(y[i:i+BATCH]).to(DEV)
            loss = cosine_loss(head(b1), head(b2), yy)
            opt.zero_grad(); loss.backward(); opt.step()
        # model selection inside a run: validation at halfway and final (trainer.py:221-231)
        if ep == halfway or ep == budget - 1:
            v = roc_auc_score(va[2], score(va[0], va[1]))
            if v > best_val:
                best_val, best_ep = v, ep + 1
                best_state = {k: t.detach().cpu().clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    score = make_scorer(head)
    return best_val, best_ep, float(roc_auc_score(te[2], score(te[0], te[1]))), best_state


rows = []
t0 = time.time()
for b in BUDGETS:
    tb = time.time()
    v, ve, t_auc, st = train_budget(b, args.seed)
    rows.append({"budget": b, "val_auc": float(v), "selected_epoch": ve,
                 "test_auc_diagnostic": t_auc, "minutes": round((time.time()-tb)/60, 2)})
    print(f"  budget {b:3d}  val {v:.4f} (ep {ve:3d})   [test {t_auc:.4f}]   "
          f"{(time.time()-tb)/60:.1f} min", flush=True)
    torch.save({f"projector.{k}": x for k, x in st.items()},
               os.path.join(OUT, f"head_b{b}_seed{args.seed}.pt"))

best = max(rows, key=lambda r: r["val_auc"])
best_t = max(rows, key=lambda r: r["test_auc_diagnostic"])
print(f"\n{'='*72}")
print(f"  VALIDATION-SELECTED BUDGET : {best['budget']} epochs "
      f"(val ROC-AUC {best['val_auc']:.4f})")
print(f"  -> VA-TE TEST ROC-AUC      : {best['test_auc_diagnostic']:.4f}"
      f"   vs published {PUBLISHED:.4f}   delta {best['test_auc_diagnostic']-PUBLISHED:+.4f}")
print(f"\n  diagnostic only, NOT selectable:")
print(f"    best test anywhere in the sweep: {best_t['test_auc_diagnostic']:.4f} "
      f"at budget {best_t['budget']}")
print(f"    is {PUBLISHED} reachable?  "
      f"{'YES' if best_t['test_auc_diagnostic'] >= PUBLISHED else 'NO - recipe tops out below it'}")
print(f"{'='*72}\n", flush=True)

json.dump({"seed": args.seed, "published": PUBLISHED, "rows": rows,
           "validation_selected": best, "diagnostic_best_test": best_t,
           "total_minutes": round((time.time()-t0)/60, 2)},
          open(os.path.join(OUT, f"sweep_budgets_seed{args.seed}.json"), "w"), indent=2)
print(f"  saved to {OUT}/sweep_budgets_seed{args.seed}.json", flush=True)
