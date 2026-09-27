#!/usr/bin/env python3
"""
Pass B: retrain the 768->768->768 projection head from the PUBLISHED configuration.

The original checkpoint is unrecoverable. The backbone is frozen public SigLIP, so the
only trained component is the 1.18M-parameter head, and every hyperparameter for it is
stated in the manuscript (Table 12, Sections 4.3/5.4):

    head        Linear(768,768) -> ReLU -> Linear(768,768), L2-normalised output
    loss        CosineLoss (Hadsell et al. with cosine distance), margin 0.3364
    optim       Adam, lr 3.8396e-5, weight decay 1.5673e-6
    batch       64        epochs 5        seed 42
    curriculum  self-paced, easy/medium/hard 100k (utils/curriculum.py)
    selection   validation ROC-AUC at the halfway and final epochs (trainer.py:221-231)

Consumes the cached normalised raw pooler outputs from embed_raw.py, so this is CPU/MPS
minutes rather than an hour. Run with --seed N to repeat under a different seed.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=5)
args = ap.parse_args()

DATA = "/Users/sashajh2/itau-group2/data/processed"
CACHE = "/Users/sashajh2/itau-group2/colab/vate_cache"
OUT = "/Users/sashajh2/itau-group2/colab/vate_out"
os.makedirs(OUT, exist_ok=True)
MARGIN, LR, WD, BATCH = 0.3364, 3.8396e-5, 1.5673e-6, 64
GATE_TARGET, GATE_TOL = 0.9450, 0.002
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(args.seed); np.random.seed(args.seed)
print(f"device {DEV} | seed {args.seed} | epochs {args.epochs}", flush=True)

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
print(f"  curriculum {[len(t[2]) for t in tiers.values()]} | val {len(va[2]):,} | test {len(te[2]):,}", flush=True)

# gather only the rows we actually touch (much smaller than the full 2.79 GB cache)
need = np.unique(np.concatenate([t[i] for t in list(tiers.values()) + [va, te] for i in (0, 1)]))
remap = np.full(len(uniques), -1, dtype=np.int64); remap[need] = np.arange(len(need))
EM = torch.from_numpy(np.ascontiguousarray(E[need])).to(DEV)
print(f"  resident embeddings {tuple(EM.shape)} ({EM.element_size()*EM.nelement()/1e9:.2f} GB)", flush=True)


def ratios(ep, tot):
    t = ep / tot
    e = 0.5 * (1 + np.cos(np.pi * t)) * (1 - t)
    h = 0.5 * (1 + np.cos(np.pi * (1 - t))) * t
    m = 1.0 - (e + h); s = e + m + h
    return e / s, m / s, h / s


head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)).to(DEV)
print(f"  head parameters: {sum(p.numel() for p in head.parameters()):,}", flush=True)
opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)


def cosine_loss(z1, z2, y):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    d = 1 - F.cosine_similarity(z1, z2)
    return (y * d.pow(2) + (1 - y) * F.relu(MARGIN - d).pow(2)).mean()


@torch.no_grad()
def score(ia, ib, bs=8192):
    head.eval(); out = []
    for i in range(0, len(ia), bs):
        z1 = F.normalize(head(EM[torch.from_numpy(remap[ia[i:i+bs]]).to(DEV)]), dim=1)
        z2 = F.normalize(head(EM[torch.from_numpy(remap[ib[i:i+bs]]).to(DEV)]), dim=1)
        out.append(F.cosine_similarity(z1, z2).cpu().numpy())
    head.train(); return np.concatenate(out)


halfway = (args.epochs - 1) // 2
best_auc, best_state, best_ep = -1, None, -1
t0 = time.time()
for ep in range(args.epochs):
    re_, rm_, rh_ = ratios(ep, args.epochs)
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

    tot_loss = 0.0; nb = 0
    for i in range(0, len(y), BATCH):
        b1 = EM[torch.from_numpy(remap[ia[i:i+BATCH]]).to(DEV)]
        b2 = EM[torch.from_numpy(remap[ib[i:i+BATCH]]).to(DEV)]
        yy = torch.from_numpy(y[i:i+BATCH]).to(DEV)
        loss = cosine_loss(head(b1), head(b2), yy)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item(); nb += 1
    msg = (f"  epoch {ep+1}/{args.epochs}  ratios e/m/h "
           f"{re_:.3f}/{rm_:.3f}/{rh_:.3f}  loss {tot_loss/nb:.5f}")
    if ep == halfway or ep == args.epochs - 1:
        vauc = roc_auc_score(va[2], score(va[0], va[1]))
        msg += f"  | val ROC-AUC {vauc:.4f}"
        if vauc > best_auc:
            best_auc, best_ep = vauc, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    print(msg, flush=True)

head.load_state_dict(best_state)
print(f"\n  selected epoch {best_ep} on validation ROC-AUC {best_auc:.4f}", flush=True)
test_auc = roc_auc_score(te[2], score(te[0], te[1]))
delta = test_auc - GATE_TARGET
print(f"\n{'='*64}")
print(f"  VA-TE TEST ROC-AUC : {test_auc:.4f}")
print(f"  published           : {GATE_TARGET:.4f}")
print(f"  delta               : {delta:+.4f}   (gate tolerance +/-{GATE_TOL})")
print(f"  GATE                : {'PASS' if abs(delta) <= GATE_TOL else 'MISS'}")
print(f"{'='*64}\n", flush=True)

torch.save({f"projector.{k}": v for k, v in best_state.items()},
           os.path.join(OUT, f"retrained_head_seed{args.seed}.pt"))
json.dump({"seed": args.seed, "val_auc": float(best_auc), "selected_epoch": best_ep,
           "test_auc": float(test_auc), "delta": float(delta),
           "gate": "PASS" if abs(delta) <= GATE_TOL else "MISS",
           "minutes": round((time.time() - t0) / 60, 2)},
          open(os.path.join(OUT, f"retrain_seed{args.seed}.json"), "w"), indent=2)
print(f"  saved head + json to {OUT}", flush=True)
