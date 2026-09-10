# 🕵️‍♂️ Spoof Detection with CLIP Embeddings - Itaú Group 2

Fraudsters are creating visually similar spoof accounts to impersonate trusted companies, posing a serious risk to financial institutions like Itaú Unibanco. This repository contains the code and data used to train, test, and evaluate a spoof detection system based on CLIP embeddings, combined with Cosine and Euclidean similarity metrics. Our approach begins by training the model on a dataset of spoofed names, then testing it on a ~1,800-name German dataset. We evaluate performance using confusion matrices, accuracy, and precision.

---

## 📁 Repository Structure


---

## 🧾 Data Overview

- `data/raw/`: Contains the original testing dataset of ~1,800 German company names — including those with special characters.
- `data/processed/`: Includes cleaned intermediate and final CSV/PKL files:
  - `merged_data.pkl`: Fully processed pairwise data (`name1`, `name2`, `label`) used for contrastive loss training.
  - `fraud_triplets.pkl`: Fully processed triplet data (`fraud_name`, `real_name`, `negative_name`) used for triplet loss training.
  - `german_merged_dataset.csv`: Testing data containing normalized German company name and label.
  - `german_companies_after_500.csv`: Reference set containing real German company names used for testing.
    - Spoof versions of these names are integrated into `german_merged_dataset.csv`.

---

## 📓 Notebook Breakdown

- `notebooks/full_run_v2/`: Iteratively tests different parameters for both pairwise/triplet datsets depending on function input.
- `notebooks/test_raw_clip/`: Testing the effectiveness of pre-trained CLIP embeddings on identifying spoof.

---

## 🧠 Models

- `models/models.py`: Contains different variations of the SiameseCLIP model:
  - `BaseSiameseCLIP`: Core model that wraps a frozen or trainable CLIP encoder and a 2-layer projector to produce normalized text embeddings.
  - `SiameseCLIPModelPairs`: Extends `BaseSiameseCLIP`; returns embeddings for a pair of inputs, used with contrastive loss.
  - `SiameseCLIPTriplet`: Extends `BaseSiameseCLIP`; returns embeddings for anchor, positive, and negative inputs, used with triplet loss.

---

## ⚙️ Scripts

- `scripts/grid_search.py`: Performs grid search across hyperparameters (e.g., learning rate, batch size, margin) to train and evaluate Siamese models using either contrastive or triplet loss.
- `scripts/train.py`: Contains training loops for both contrastive (`train_pair`) and triplet (`train_triplet`) loss models.
- `scripts/test.py`: Computes model predictions by comparing test names to a reference set using Cosine/Euclidean similarity of projected embeddings.
- `scripts/eval.py`: Evaluates model predictions using ROC curve, confusion matrix, and threshold-based accuracy metrics.
- `scripts/test_raw_clip.py`: Evaluates performance using raw (untrained) CLIP embeddings for baseline comparison.

---

## 🛠️ Utils
- `utils/embeddings.py`: Defines helper functions for generating CLIP-based embeddings, including `EmbeddingExtractor`, `batched_embedding`.
- `utils/loss.py`: Defines multiple loss functions including contrastive, triplet (Cosine & Euclidean), and hybrid variations.
- `utils/data.py`: Contains PyTorch dataset classes for handling contrastive and triplet data formats.
- `utils/evals.py`: Provides plotting utilities and threshold optimization functions for model evaluation.

---

## 🔍 Split Integrity / Data Leakage Check

`scripts/split_overlap_check.py` verifies, directly from the released split files in
`data/processed/`, that no legitimate domain (anchor) identity is shared across the
training, validation, and test splits, and that negative sampling and hard-negative
mining never reach outside their own split. Run it with:

```bash
python scripts/split_overlap_check.py
```

Results on the released data:

| | train | validate | test |
|---|---|---|---|
| pairs | 976,122 | 51,380 | 256,886 |
| unique legitimate anchors | 69,723 | 3,670 | 18,349 |
| anchor-positive / anchor-negative | 627,507 / 348,615 | 33,030 / 18,350 | 165,141 / 91,745 |
| negative-sampling pool | 63,081 | 3,290 | 16,560 |

- **Anchor overlap across splits is exactly zero** for train/validate, train/test, and
  validate/test. The three anchor sets sum to the 91,742 domains in their union, so the
  splits are a strict partition of the legitimate-domain population. There are also no
  duplicated (variant, legitimate) pairs across splits.
- **Negatives are strictly within-split.** In every split, 100% of negative partners are
  legitimate domains from that split's own anchor pool and 0% come from either other split.
- **Hard-negative mining uses training identities only.** Across all nine mined-negative
  training sets (triplet / InfoNCE / SupCon at easy, medium, and hard difficulty), 100% of
  mined negatives are training-split identities and 0.000000% are validation or test
  identities.
- One incidental effect is reported for completeness. In 212 cases a training-split spoof
  variant coincides as a character string with a legitimate domain in the validation or test
  split, because that domain is itself within spoofing distance of a *different* legitimate
  domain. For example, `iboats` appears in training as the spoof member of the pair
  (`boats`, `iboats`) and in test as the legitimate anchor of its own family of 14 spoof
  pairs — both labels are correct. This affects 233 of 976,122 training rows (0.024%), each
  colliding string appearing in a single training row, and it reuses no anchor identity
  across splits: the anchor-level intersections above remain exactly zero.

This script was added to answer a Round 1 reviewer comment on the paper
*"Multi-Signal Learning Framework for Robust Detection of Visually Deceptive Text"*
asking for the overlap of anchor/domain identities across the three splits and for
clarification that hard-negative mining draws only on training identities.

## 📦 Installation

We recommend using a Python 3.10 virtual environment.

```bash
git clone https://github.com/sashajh2/itau-group2.git
cd itau-group2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
