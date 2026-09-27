#!/usr/bin/env python3
"""
Rebuild Figure 4 from Table 7's values.

Rafael, 24 Sep: "creating a new Figure 4 from values in Table 7 may solve
inconsistencies and ChatGPT problem."

Two changes from the original:
  - the ChatGPT bar is gone (no recoverable prompt, model version or parsing rule)
  - bars taken from prior work are marked with a dagger. Source for both is
    Vinayakumar & Soman, ICT Express 6(1):16-19, Table 1 (domain name spoofing).
    They evaluate on a 15,000-pair SUBSET of the Woodbridge data, not our
    256,886-pair test split, so they are not measured on the same evaluation set.
    Woodbridge et al. publish no numeric AUC at all - only ROC curves.

Re-run this after the fusion refit: edit the two values marked PROVISIONAL.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, roc_auc, quoted_from_prior_work)
BARS = [
    ("Fuzzy",          0.54,   False),
    ("BERT",           0.79,   False),
    ("Edit Dist.",     0.8137, False),   # Table 8, Levenshtein
    ("Token Set",      0.8350, False),   # Table 8, Token Set Ratio
    ("String Metrics", 0.8920, False),   # Table 8, Lev + Token Set (honest protocol)
    ("VATE",           0.9472, False),   # Table 8, VA-TE (retrained head)
    ("Siamese-CNN",    0.93,   True),    # Vinayakumar Table 1, domain spoofing (was 0.97 - wrong)
    ("Siamese-LSTM",   0.97,   True),    # Vinayakumar Table 1, domain spoofing
    ("Siamese-GRU",    0.98,   True),    # Vinayakumar Table 1, their best result
    ("VATE+String",    0.9639, False),   # Table 8, full fusion (honest protocol)
]

OUT = "image.png"
BLUE, EDGE = "#1f77b4", "#c8813c"

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
ax.set_facecolor("#eaeaf2")
for s in ax.spines.values():
    s.set_visible(False)

labels = [(l + r"$^\dagger$") if q else l for l, _, q in BARS]
vals   = [v for _, v, _ in BARS]

bars = ax.bar(range(len(BARS)), vals, width=0.78, color=BLUE, zorder=3)
bars[-1].set_edgecolor(EDGE)          # highlight the proposed method, as in the original
bars[-1].set_linewidth(1.4)

ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC-AUC", fontsize=11)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticks(range(len(BARS)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5)
ax.get_xticklabels()[-1].set_fontweight("bold")
ax.yaxis.grid(True, color="white", linewidth=1.1, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0, labelsize=9.5)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
for l, v, q in BARS:
    print(f"  {l:16s} {v:.4f}{'  (quoted from prior work)' if q else ''}")
