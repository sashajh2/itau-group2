# VA-TE round-2 refit

Everything needed to reproduce the results in the revised manuscript
(Electronics 4535789), under the corrected evaluation protocol:

* the fusion classifier is fitted on the **training** split only
* every decision threshold is selected on the **validation** split
* the held-out **test** split is read once, for reporting

The previous version fitted the fusion classifier by 5-fold out-of-fold
prediction *within* the test split. That is what reviewers R1-1, R1-6 and R2-1
identified, and it is what these scripts replace.

## The projection head

The checkpoint behind the original submission was stored on infrastructure that
has since been decommissioned and could not be recovered. Because the SigLIP
backbone is frozen and public, the only trained component is the
1,181,184-parameter projection head, and every hyperparameter for it is stated
in the manuscript (Table 5, Sections 4.3 and 5.4). `retrain_head.py` and
`sweep_epochs.py` rebuild it from that configuration.

The head used for every reported figure is committed here as
`projection_head_seed42_40ep.pt`, so it cannot be lost a second time.

**The retrain reproduces the original.** Running the *previous* out-of-fold
procedure on the retrained head returns 0.9707 against the published 0.9715 for
the fused model, and 0.9153 against 0.9150 for the string-only combination. The
difference between 0.9715 and the corrected 0.9639 is the protocol change, not
the retraining.

## Running it

Order matters; each step consumes the previous one's cache.

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `embed_raw.py` | Cache normalised raw SigLIP pooler outputs for all 907,143 unique strings. Head-independent, so it is done once. | ~65 min |
| 2 | `sweep_epochs.py` | Train the head at each epoch budget; select the budget on **validation** ROC-AUC. Picks 40. | ~2 h |
| 3 | `run_final.py --head <selected>` | Signal ablation (Table 8), its prose figures, the paired bootstrap, and the complementarity diagnostic (Table 12). | ~25 min |
| 4 | `make_table8.py` | Combiner ablation (Table 9). Reuses the projected cache written by step 3. | ~10 min |
| 5 | `embed_raw_com.py` then `run_com.py --head <selected>` | The `.com`-retained pass: suffix ablation (Table 11) and Table 6's VA-TE row. | ~80 min |
| 6 | `make_figure4.py`, `make_figure5.py` | Figures 4 and 5, generated from the same scores as Table 8. | ~3 min |

`retrain_head.py` trains a single budget and is the quickest way to check the
pipeline end to end (~1 min once step 1 has run).

`run_final.py` is generated from `run_local.py` by `make_run_final.py`, which
replaces only the embedding source. Everything downstream is unchanged, so the
diff between the two files is exactly the change of model provenance.

## Two guards worth leaving in place

**The gate.** `run_final.py` aborts if VA-TE test ROC-AUC misses the published
0.9450 by more than 0.005. It is the reason a wrong checkpoint was caught rather
than silently producing plausible numbers: a 768→512→512 head found on disk
scored 0.9398 and failed it. The tolerance was widened from 0.002 once the head
became a retrain rather than a recovered file; 0.9398 still fails at 0.005.

**Cache verification.** Every script checks its embedding cache against the
actual splits and fails loudly on mismatch. The scripts previously checked shape
but not provenance, which is how a stale cache from the wrong head was reused
without anyone noticing.

## Environment

```
python3 -m venv venv && ./venv/bin/pip install \
    pandas pyarrow scikit-learn rapidfuzz torch transformers \
    sentencepiece protobuf matplotlib tqdm
```

`protobuf` is easy to miss: `SiglipTokenizer` imports without it and then fails
at use.
