
## Table 7, corrected - paste into template.tex

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Levenshtein | 0.8137 | 0.8391 | 0.8170 | 0.7930 | 0.9681 | 0.8718 |
| Token Set Ratio | 0.8350 | 0.9083 | 0.7836 | 0.8680 | 0.7824 | 0.8230 |
| Levenshtein + Token Set Ratio | 0.8920 | 0.9276 | 0.8139 | 0.8887 | 0.8121 | 0.8487 |
| VA-TE (SigLIP-pair embedding similarity) | 0.9472 | 0.9663 | 0.8860 | 0.9041 | 0.9204 | 0.9122 |
| VA-TE + String (3-feature, train) | 0.9639 | 0.9757 | 0.9084 | 0.9179 | 0.9417 | 0.9297 |

| Model | FPR@Youden | FPR@FNR=1% | FPR@FNR=5% | FPR@FNR=10% |
|---|---|---|---|---|
| Levenshtein | 0.4550 | 0.6551 | 0.4550 | 0.4550 |
| Token Set Ratio | 0.2142 | 0.9996 | 0.9345 | 0.6008 |
| Levenshtein + Token Set Ratio | 0.1830 | 0.5483 | 0.4166 | 0.3503 |
| VA-TE (SigLIP-pair embedding similarity) | 0.1758 | 0.4835 | 0.2397 | 0.1472 |
| VA-TE + String (3-feature, train) | 0.1516 | 0.3272 | 0.1664 | 0.1031 |

### LaTeX rows for Table 7 (first block)
```
Levenshtein & 0.8137 & 0.8391 & 0.8170 & 0.7930 & 0.9681 & 0.8718 \\
Token Set Ratio & 0.8350 & 0.9083 & 0.7836 & 0.8680 & 0.7824 & 0.8230 \\
Levenshtein + Token Set Ratio & 0.8920 & 0.9276 & 0.8139 & 0.8887 & 0.8121 & 0.8487 \\
VA-TE (SigLIP-pair embedding similarity) & 0.9472 & 0.9663 & 0.8860 & 0.9041 & 0.9204 & 0.9122 \\
VA-TE + String (3-feature, train) & 0.9639 & 0.9757 & 0.9084 & 0.9179 & 0.9417 & 0.9297 \\
```

### LaTeX rows for Table 7 (FPR block)
```
Levenshtein & 0.4550 & 0.6551 & 0.4550 & 0.4550 \\
Token Set Ratio & 0.2142 & 0.9996 & 0.9345 & 0.6008 \\
Levenshtein + Token Set Ratio & 0.1830 & 0.5483 & 0.4166 & 0.3503 \\
VA-TE (SigLIP-pair embedding similarity) & 0.1758 & 0.4835 & 0.2397 & 0.1472 \\
VA-TE + String (3-feature, train) & 0.1516 & 0.3272 & 0.1664 & 0.1031 \\
```