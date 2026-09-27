# Pre-flight — VA-TE fusion refit

**Nothing is retrained.** The projection head is loaded frozen. The only thing fitted is the
gradient-boosting fusion classifier, on the training split alone.

## What to upload
- `best_model_siglip_pair.pt` — ~2.6 MB, currently in `~/Downloads/`
- `vate_refit.py` — from this folder

Data is cloned from the public repo; nothing else to upload.

## Runtime
Runtime → Change runtime type → **T4 GPU**. Roughly 20–35 min: ~907k unique strings
through the frozen SigLIP text tower, then CPU-side fitting. Embedding is resumable —
if the session drops, rerun the cell and it picks up from `vate_cache/`.

## The one gate
The script aborts if VA-TE test ROC-AUC misses **0.9450** by more than 0.002. If the
checkpoint or encode path is wrong, every downstream number would be meaningless, so it
refuses to continue rather than producing plausible garbage. A failed gate means the
checkpoint in `~/Downloads/` is not the one that produced the paper.

## What to send back
1. The **projection head** line printed near the start — settles whether Table 12 says
   512 or 768, and whether §4.3 / §5.4 should say 0.66M or 1.18M parameters.
2. Whether the gate passed.
3. `vate_out/corrected_tables.md` — LaTeX rows ready to paste.
4. The **bootstrap** block. Watch `d_vs_vate`: if its 95% CI still excludes zero, the
   paper's central claim survives as written. If it straddles zero, the claim needs
   rewording and the 4-feature variant becomes the fallback.

## Config
All tunables are at the top of `vate_refit.py`. Drop `BATCH_SIZE` to 256 if the T4 runs
out of memory. Set `RUN_COM_RETAINED = False` to skip the Table 10 variant and save ~8 min.

## Running locally instead of on Colab

`vate_refit.py` auto-detects `cuda -> mps -> cpu`, so it runs on your Mac unchanged.
You need: `pandas pyarrow scikit-learn rapidfuzz torch transformers sentencepiece protobuf`.
`protobuf` is easy to miss — SiglipTokenizer imports but fails without it.

Two things about sizing the job:

**Padding is not negotiable.** Strings average 9.5 characters but `padding="max_length"`
pads every one to 64 tokens. That looks like wasted compute, but HuggingFace's
`SiglipTextTransformer` pools with `last_hidden_state[:, -1, :]` — the last position, which
under max-length padding is a pad token. That is what SigLIP was trained with and what the
checkpoint was fitted against. Switch to dynamic padding and every embedding changes, the
cosines change, and the 0.9450 gate fails.

**Train is where the cost is.** Unique strings by split: test 181,783 · validation 36,356 ·
train 689,697 · all three 907,143. Train is 76% of the job. Use `TRAIN_SUBSAMPLE` to cut it
(200k rows -> 46% of the full job). Never subsample test or validation.

### Measured throughput (this Mac, MPS, fp32)

239 strings/sec, and batch size makes no difference (236/sec at 256, 239/sec at 512 — MPS
is already saturated, so there is nothing to tune).

| Job | Unique strings | Wall clock |
|---|---|---|
| test + validation only | 218,109 | ~15 min |
| + 200k train rows | 413,375 | ~29 min |
| + 300k train rows | 478,036 | ~33 min |
| full splits | 907,143 | ~63 min |
| `RUN_COM_RETAINED = True` adds | ~182,000 | +13 min |

A Colab T4 is roughly 4–6x faster, so ~10–15 min for the full job. That saves under an
hour and costs an upload, a session that can time out, and Rafael's $100.
