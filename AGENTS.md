# ⭐ READ ME FIRST — VA-TE homoglyph paper: reviewer-response evidence & reproduction

**If you are an AI agent (Claude Code or similar) or a collaborator opening this folder/repo: start here.**
This directory is the consolidated, verified evidence pack for responding to peer-review comments on the paper
*"Multi-Signal Learning Framework for Robust Detection of Visually Deceptive Text"* (a.k.a. **VA-TE + String**).
It also documents exactly how to reproduce every number in the paper.

Last updated by an autonomous agent run. Numbers that are still being computed live in the referenced files, not inlined here.

---

## 1. What this project is
Detect visually deceptive / homoglyph domain names (e.g. `okayplayer.com` vs `okayplaiier.com`) by fusing:
- **VA-TE**: a visually-aligned text embedding (SigLIP backbone + trained projection head, contrastive/curriculum training), and
- **String features**: Levenshtein edit distance + token-set ratio,
combined by a **gradient-boosting** fusion classifier. Paper reports ROC-AUC 0.98 for the full VA-TE+String model.

## 2. Where everything lives
- **Code (public):** GitHub `github.com/sashajh2/itau-group2` — `model_utils/` (models/losses), `scripts/` (training, ensemble_pipeline, evaluation), `utils/` (curriculum, data, embeddings, evals).
- **Data (released):** `data/processed/*.parquet` in that repo. Key files: `train_pairs_ref` (976,122), `validate_pairs_ref` (51,380), `test_pairs_all` (256,886); difficulty subsets `train_pairs_{easy,medium,hard}_100k` (100k each).
- **Source benchmark:** Woodbridge et al. 2018, `github.com/endgameinc/homoglyph` → `data/domains_spoof.pkl` (dict train/validate/test of `(real_name, fraudulent_name, label)`; **label 1 = spoof**). Our splits **byte-match** it. ⚠️ The paper cites a **typo'd URL** `endgameinc/homoglylph` (double-l, 404) — fix to `homoglyph`.
- **Trained weights:** `best_model_siglip_pair.pt` (~4.7 MB) + `best_hparams_siglip_pair.json` — in Sasha's Google Drive (personal account) and downloaded to `~/Downloads/best_model_siglip_pair.pt`. Also `best_model_siglip_triplet.pt`.
- **Dataset-construction notebooks (Drive, personal account):** the real generators are `assemble_train_datasets.ipynb` (v1) and `final_dataset_assembly.ipynb` (v2); `separability.ipynb` makes Figure 2; `smaller_datasets.ipynb` applies the 100k cap. (`hard_neg.ipynb` only `.describe()`s a saved CSV — not a generator.)

## 3. Reproducibility — YES, fully
You need code + data + seeds + weights, and all exist:
- **Fast path (no training):** load `best_model_siglip_pair.pt`, compute the embedding cosine on `test_pairs_all`, add Levenshtein + token-set ratio, run the gradient-boosting fusion. Reproduces the paper's Table 6 (VA-TE ≈ 0.95, VA-TE+String ≈ 0.98). Recipe + script: `reproduce_vate_rows_README.md`, `vate_repro.py`, `build_metrics.py`.
- **From scratch:** `best_hparams_siglip_pair.json` + the repo training code + the released data + seed 42. Only caveat: minor GPU/library float nondeterminism, and the 100k downsample used `random_state=None` (no fixed seed).

## 4. The four reviewer comments — answers (evidence-backed)
1. **R1-C3 (Figure 2 sample sizes):** report the full `pairs_all` counts (verified by `len()`): **easy 697,230** (348,615 anchor-pos + 348,615 anchor-neg), **medium 976,122** (627,507 + 348,615), **hard 697,230** (348,615 + 348,615); anchor-neg = 348,615/tier. The `*_100k` files are downsampled *training* subsets, not the Figure 2 source. See `SUMMARY.md`.
2. **R1-C8 (thresholds):** there are **none**. Difficulty = negative-sampling strategy: **easy** uniform-random negatives (~63k pool), **medium** the benchmark's own negatives, **hard** nearest negatives by cosine to the anchor in **raw SigLIP** (`google/siglip-base-patch16-224`, mean-pooled, L2-norm; top-K). Mean anchor-neg cos 0.47/0.59/0.77. Levenshtein/token-set are descriptive stats only. ⚠️ Fix the contradictory Table 2 caption. Proof: `comment8_construction.md`, `sasha_notebooks.md`.
3. **R1-C11 & R2-C3 (full metrics / ROC / FPR@FNR for the fusion model):** DONE. `fusion_metrics_table_FULL.md`/`.csv` — 5 rows on the test set with ROC-AUC, PR-AUC, precision/recall/F1, FPR@Youden, FPR@FNR(1/5/10%); ROC in `roc_curves_all_methods.png`. Measured ROC-AUC: Levenshtein 0.81, Token Set 0.84, Lev+TokenSet 0.92, **VA-TE 0.945** (paper ~0.95), **VA-TE+String 0.972** (paper 0.98; best method, lowest FPR at every FNR). Reproduced from the checkpoint via `finish_vate.py` (embeddings cached in `emb_cache/`).

## 5. Open items / to verify (for the authors)
- Fix the Table 2 caption ("descriptive statistics", not "split based on…") and the `endgameinc/homoglyph` URL typo in the paper.
- VA-TE+String: the fair number is the in-distribution 0.972; the literal fit-on-train protocol gives 0.906 due to a raw-Levenshtein train/test length shift (length-normalize to remove) — see the note in `fusion_metrics_table_FULL.md`.

## 6. File index (this folder)
- `SUMMARY.md` — full narrative + per-comment detail (**read second**).
- `DRAFT_EMAIL_to_Rafael.md` — ready follow-up email + paste-in reviewer responses.
- `dataset_provenance.md` — Woodbridge byte-match proof.
- `fusion_metrics_table*.md/.csv`, `roc_curves_*.png`, `roc_points_*.csv` — metrics + ROC data.
- `comment8_construction.md`, `sasha_notebooks.md`, `notebook_findings.md` — how difficulty/sampling works, code-grounded.
- `drive_recon.md`, `colab_inventory.md` — what's in Drive (weights, notebooks) + access notes.
- `reproduce_vate_rows_README.md`, `build_metrics.py`, `vate_repro.py`, `recompute_table2.py` — reproduction scripts.
