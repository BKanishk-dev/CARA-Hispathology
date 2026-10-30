#!/usr/bin/env python
"""Create the train / validation / test split of PanNuke (3 folds).

    python scripts/prepare_pannuke.py                       # uses data/pannuke
    python scripts/prepare_pannuke.py --root /path/to/pannuke --seed 42
    python scripts/prepare_pannuke.py --samples             # (re)generate demo sample images only

For the full dataset (2,656 + 2,523 + 2,722 = 7,901 images) the split is exactly
5,688 train / 632 validation / 1,581 test, stratified by tissue type (19 classes).
The result is written to <root>/splits.json and contains indices into the concatenation
fold1 + fold2 + fold3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.dataset import (SPLITS_FILE, find_fold_files, load_fold_arrays, split_sizes,  # noqa: E402
                         stratified_split)
from app.inference import ensure_samples  # noqa: E402
from app.utils import CLASS_NAMES, FOLD_SIZES, PANNUKE_DIR, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=PANNUKE_DIR, help="folder that contains the 3 PanNuke folds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", action="store_true", help="only (re)generate synthetic gallery samples")
    args = ap.parse_args()

    if args.samples:
        items = ensure_samples(force=True)
        print(f"Wrote {len(items)} sample images")
        return 0

    for f in (1, 2, 3):
        found = find_fold_files(args.root, f)
        if found is None:
            print(f"ERROR: fold {f} (images.npy + types.npy) not found under {args.root}.\n"
                  f"       Expected e.g. '{args.root}/Fold {f}/images/fold{f}/images.npy'. See README.")
            return 1
        print(f"fold {f}: {found[0]}")

    images, labels = load_fold_arrays(args.root)
    counts = [len(x) for x in images]
    for f, n in zip((1, 2, 3), counts):
        flag = "" if n == FOLD_SIZES[f] else f"  (official: {FOLD_SIZES[f]})"
        print(f"  fold {f}: {n} images{flag}")
    all_labels = np.concatenate(labels)
    sizes = split_sizes(len(all_labels))
    print(f"total {len(all_labels)} images -> train {sizes['train']} / val {sizes['val']} / test {sizes['test']}")

    split = stratified_split(all_labels, sizes, args.seed)
    assert len(set(split["train"]) & set(split["val"])) == 0
    assert len(set(split["train"]) & set(split["test"])) == 0
    assert len(set(split["val"]) & set(split["test"])) == 0

    out = {
        "seed": args.seed,
        "fold_sizes": counts,
        "class_names": CLASS_NAMES,
        "sizes": {k: int(len(v)) for k, v in split.items()},
        "class_counts": {k: np.bincount(all_labels[v], minlength=len(CLASS_NAMES)).tolist()
                         for k, v in split.items()},
        "splits": {k: v.tolist() for k, v in split.items()},
    }
    save_json(args.root / SPLITS_FILE, out)
    print(f"Saved {args.root / SPLITS_FILE}")
    print(f"{'class':<14}{'train':>7}{'val':>6}{'test':>6}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:<14}{out['class_counts']['train'][i]:>7}{out['class_counts']['val'][i]:>6}"
              f"{out['class_counts']['test'][i]:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
