"""
Split-overlap verification for the homoglyph benchmark.

Reports, directly from the released split files in data/processed/:
  1. unique legitimate-domain (anchor) identities per split,
  2. pairwise and three-way anchor-identity intersections across splits,
  3. exact duplicate (variant, legitimate) pairs across splits,
  4. the provenance of every negative partner and every mined hard negative,
     i.e. whether it is an identity from its own split or from another split,
  5. incidental spoof-string collisions with out-of-split anchors.

Written to answer reviewer comment R1-C4 (Round 1): whether the same
legitimate domain or anchor can appear across splits through different spoof
variants or negative pairs, and whether hard-negative mining is confined to
training identities.

Usage:
    python scripts/split_overlap_check.py [--data-dir data/processed]
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = {
    "train": "train_pairs_ref.parquet",
    "validate": "validate_pairs_ref.parquet",
    "test": "test_pairs_ref.parquet",
}

# Mined-negative training sets. Each entry is (filename, column) where the
# column holds either a single negative name or a list of them.
MINED_NEGATIVES = [
    ("train_triplet_easy_100k.parquet", "negative_name"),
    ("train_triplet_medium_100k.parquet", "negative_name"),
    ("train_triplet_hard_100k.parquet", "negative_name"),
    ("train_infonce_easy_100k.parquet", "negative_names"),
    ("train_infonce_medium_100k.parquet", "negative_names"),
    ("train_infonce_hard_100k.parquet", "negative_names"),
    ("train_supcon_easy.parquet", "negative_names"),
    ("train_supcon_medium.parquet", "negative_names"),
    ("train_supcon_hard.parquet", "negative_names"),
]


def strip_tld(s):
    """Drop the constant '.com' suffix so files that keep it and files that
    don't are compared on the same footing."""
    return s.astype(str).str.replace(r"\.com$", "", regex=True)


def flatten(column):
    """Return a flat Series of names from a column of names or lists of names."""
    values = column.tolist()
    if len(values) and isinstance(values[0], (list, tuple, np.ndarray)):
        values = np.concatenate([np.asarray(v, dtype=object) for v in values])
    return strip_tld(pd.Series(values, dtype=object))


def main(data_dir):
    data_dir = Path(data_dir)
    splits = {name: pd.read_parquet(data_dir / f) for name, f in SPLITS.items()}
    anchors = {name: set(strip_tld(df["real_name"])) for name, df in splits.items()}
    spoofs = {
        name: set(strip_tld(df.loc[df["label"] == 1, "fraudulent_name"]))
        for name, df in splits.items()
    }

    print("=" * 72)
    print("1. SPLIT SIZES AND ANCHOR IDENTITIES")
    print("=" * 72)
    for name, df in splits.items():
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        print(
            f"  {name:9s} pairs={len(df):>9,}  unique anchors={len(anchors[name]):>7,}"
            f"  anchor-pos={pos:>9,}  anchor-neg={neg:>9,}"
        )
    union = set().union(*anchors.values())
    total = sum(len(a) for a in anchors.values())
    print(f"\n  sum of per-split anchors = {total:,}")
    print(f"  size of their union      = {len(union):,}")
    print(f"  strict partition of the anchor set: {total == len(union)}")

    print("\n" + "=" * 72)
    print("2. CROSS-SPLIT ANCHOR-IDENTITY OVERLAP")
    print("=" * 72)
    for a, b in itertools.combinations(splits, 2):
        shared = anchors[a] & anchors[b]
        jaccard = len(shared) / len(anchors[a] | anchors[b])
        print(f"  anchors({a}) n anchors({b}) = {len(shared):>6,}   Jaccard = {jaccard:.6f}")
    print(f"  three-way intersection            = "
          f"{len(anchors['train'] & anchors['validate'] & anchors['test']):>6,}")

    print("\n" + "=" * 72)
    print("3. EXACT DUPLICATE (variant, legitimate) PAIRS ACROSS SPLITS")
    print("=" * 72)
    pairs = {
        name: set(
            zip(strip_tld(df["fraudulent_name"]), strip_tld(df["real_name"]))
        )
        for name, df in splits.items()
    }
    for a, b in itertools.combinations(splits, 2):
        print(f"  pairs({a}) n pairs({b}) = {len(pairs[a] & pairs[b]):,}")

    print("\n" + "=" * 72)
    print("4a. NEGATIVE-PAIR PROVENANCE (in-split pairs files)")
    print("=" * 72)
    for name, df in splits.items():
        partners = strip_tld(df.loc[df["label"] == 0, "fraudulent_name"])
        print(f"  {name}: {len(partners):,} negative rows, pool = {partners.nunique():,} unique legitimate domains")
        for other in splits:
            tag = "OWN split" if other == name else f"{other} split"
            print(f"      drawn from {tag:15s}: {partners.isin(anchors[other]).mean():.6%}")

    print("\n" + "=" * 72)
    print("4b. MINED HARD-NEGATIVE PROVENANCE (training sets)")
    print("=" * 72)
    for fname, col in MINED_NEGATIVES:
        path = data_dir / fname
        if not path.exists():
            print(f"  {fname:34s} [not present]")
            continue
        df = pd.read_parquet(path)
        if col not in df.columns:
            print(f"  {fname:34s} [no column '{col}']")
            continue
        neg = flatten(df[col])
        print(
            f"  {fname:34s} {len(neg):>7,} negatives, {neg.nunique():>6,} unique"
            f" | from train={neg.isin(anchors['train']).mean():.6%}"
            f"  validate={neg.isin(anchors['validate']).mean():.6%}"
            f"  test={neg.isin(anchors['test']).mean():.6%}"
        )

    print("\n" + "=" * 72)
    print("5. INCIDENTAL SPOOF-STRING COLLISIONS WITH OUT-OF-SPLIT ANCHORS")
    print("=" * 72)
    print("  (a generated variant coincides as a string with a legitimate domain")
    print("   in another split because that domain is itself within spoofing")
    print("   distance of a different legitimate domain -- e.g. \'iboats\' is a spoof")
    print("   of \'boats\' in train and the legitimate anchor of its own family in")
    print("   test. Both labels are correct; no anchor identity is reused.)")
    for name, df in splits.items():
        foreign = set().union(*(anchors[o] for o in splits if o != name))
        variants = strip_tld(df.loc[df["label"] == 1, "fraudulent_name"])
        hits = variants[variants.isin(foreign)]
        uniq = hits.drop_duplicates()
        lengths = uniq.str.len()
        print(
            f"  {name:9s} {uniq.nunique():>4,} unique colliding strings"
            f" -> {len(hits):>5,} rows ({len(hits) / len(df):.4%})"
            + (f", length median={lengths.median():.0f} max={lengths.max():.0f}"
               f", e.g. {', '.join(uniq.head(6))}" if len(uniq) else "")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    main(parser.parse_args().data_dir)
