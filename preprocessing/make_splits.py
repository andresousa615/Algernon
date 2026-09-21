# -*- coding: utf-8 -*-
"""
Build train / val / test CSVs from a dataset directory, splitting by subject.

Expected layout (one folder per exam; several exams may share a subject):

    <root>/<subject>__<session>/raw.nii.gz
    <root>/<subject>__<session>/training_masks/mask_4_classes.nii.gz

The subject id is the folder name up to the first "__" (the whole name when
there is no separator), so all exams of a subject land in the same split.

Usage:
    python preprocessing/make_splits.py --root /path/to/dataset_128 --out data \
        [--test_size 0.2] [--val_size 0.25] [--seed 42] [--test_only]

--test_only writes a single test.csv with every exam (for external test sets).
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_FILENAME = "raw.nii.gz"
MASK_FILENAME = os.path.join("training_masks", "mask_4_classes.nii.gz")


def build_exam_records(root: str, require_mask: bool = True) -> pd.DataFrame:
    records = []
    # sorted() matters: os.listdir has no guaranteed order and the order of
    # the subject list decides the split for a given seed.
    for exam_folder in sorted(os.listdir(root)):
        exam_path = os.path.join(root, exam_folder)
        if not os.path.isdir(exam_path):
            continue
        image_path = os.path.join(exam_path, IMAGE_FILENAME)
        mask_path = os.path.join(exam_path, MASK_FILENAME)
        if not os.path.exists(image_path) or (require_mask and not os.path.exists(mask_path)):
            continue
        records.append({
            "subject_id": exam_folder.split("__")[0],
            "image_path": image_path,
            "mask_path": mask_path if os.path.exists(mask_path) else "",
        })
    return pd.DataFrame(records)


def split_by_subject(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Split at the subject level so no subject appears in two sets."""
    unique_subjects = sorted(set(df["subject_id"].tolist()))
    train_val, test = train_test_split(unique_subjects, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)

    def subset(subjects):
        return df[df["subject_id"].isin(set(subjects))].reset_index(drop=True)

    return subset(train), subset(val), subset(test)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Dataset directory (one folder per exam)")
    ap.add_argument("--out", required=True, help="Output directory for the CSVs")
    ap.add_argument("--test_size", type=float, default=0.20, help="Fraction of subjects held out for test")
    ap.add_argument("--val_size", type=float, default=0.25, help="Fraction of the remainder used for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_only", action="store_true", help="Write every exam to test.csv (no split)")
    args = ap.parse_args()

    df = build_exam_records(args.root, require_mask=not args.test_only)
    if df.empty:
        raise SystemExit(f"No exams found under {args.root}")
    print(f"{len(df)} exams, {df['subject_id'].nunique()} subjects")

    os.makedirs(args.out, exist_ok=True)
    cols = ["image_path", "mask_path"]

    if args.test_only:
        df[cols].to_csv(os.path.join(args.out, "test.csv"), index=False)
        print(f"test.csv written to {args.out}")
        return

    df_train, df_val, df_test = split_by_subject(df, args.test_size, args.val_size, args.seed)
    df_train[cols].to_csv(os.path.join(args.out, "train.csv"), index=False)
    df_val[cols].to_csv(os.path.join(args.out, "val.csv"), index=False)
    df_test[cols].to_csv(os.path.join(args.out, "test.csv"), index=False)

    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        print(f"  {name:5s}: {len(part):4d} exams, {part['subject_id'].nunique():3d} subjects")
    print(f"CSVs written to {args.out}")


if __name__ == "__main__":
    main()
