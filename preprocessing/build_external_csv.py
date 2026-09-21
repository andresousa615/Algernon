# -*- coding: utf-8 -*-
"""
Build the CSV for an external set without ground truth.

Walks a directory in which every sub-folder is an exam containing a NIfTI
volume, and writes a CSV with the column image_path only. No mask_path is
written: without annotations there is no Dice to compute.

The exam name is the sub-folder name, which is also how inference.py names
its output files.

Usage:
    python preprocessing/build_external_csv.py --root /path/to/exams --out data/external.csv \
        [--include PREFIX] [--limit 40]
"""
import argparse
import os

import pandas as pd


def collect(root: str, include: str | None = None) -> pd.DataFrame:
    rows = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        if include and not name.startswith(include):
            continue
        # First NIfTI that is neither a mask nor an already-defaced volume.
        candidates = sorted(f for f in os.listdir(folder)
                            if f.endswith(".nii.gz") and "mask" not in f.lower() and "defaced" not in f.lower())
        if not candidates:
            continue
        rows.append({"image_path": os.path.join(folder, candidates[0])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include", default=None, help="Folder-name prefix, to select one cohort")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = collect(args.root, args.include)
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        raise SystemExit(f"No exams found under {args.root}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} exams -> {args.out}")
    for p in df["image_path"].head(3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
