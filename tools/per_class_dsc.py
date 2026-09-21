# -*- coding: utf-8 -*-
"""
Per-class Dice (ears, mouth, nose, eyes) over the test predictions.

Usage:
    python tools/per_class_dsc.py --inference_dir <run>/inference --test_csv data/test.csv [--show_worst 5]

Predictions are read as <inference_dir>/<exam_id>_pred_mask.nii.gz; the GT
comes from the CSV's mask_path column and is remapped {0,2,3,4,5} -> {0..4}.
"""

import argparse
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd

CLASS_NAMES = {1: "ears", 2: "mouth", 3: "nose", 4: "eyes"}
REMAP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def remap_labels(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for orig, new in REMAP.items():
        out[mask == orig] = new
    return out


def dice_binary(gt_k: np.ndarray, pred_k: np.ndarray):
    intersection = (gt_k & pred_k).sum()
    denom = gt_k.sum() + pred_k.sum()
    if denom == 0:
        return None  # class absent from this exam — excluded from the mean
    return float(2 * intersection) / float(denom)


def analyze(inference_dir: str, test_csv: str, show_worst: int = 0) -> None:
    df = pd.read_csv(test_csv)
    scores: dict[int, list[tuple[str, float]]] = defaultdict(list)
    missing_pred, missing_gt = [], []

    for _, row in df.iterrows():
        exam_id = os.path.basename(os.path.dirname(row["image_path"]))
        pred_path = os.path.join(inference_dir, f"{exam_id}_pred_mask.nii.gz")
        gt_path = row["mask_path"]

        if not os.path.exists(pred_path):
            missing_pred.append(exam_id)
            continue
        if not isinstance(gt_path, str) or not os.path.exists(gt_path):
            missing_gt.append(exam_id)
            continue

        pred = nib.load(pred_path).get_fdata().astype(np.int32)
        gt = remap_labels(np.round(nib.load(gt_path).get_fdata()).astype(np.int32))

        if pred.shape != gt.shape:   # resize prediction to the GT grid if needed
            from scipy.ndimage import zoom
            factors = tuple(g / p for g, p in zip(gt.shape, pred.shape))
            pred = zoom(pred, factors, order=0).astype(np.int32)

        for k in CLASS_NAMES:
            dsc = dice_binary(gt == k, pred == k)
            if dsc is not None:
                scores[k].append((exam_id, dsc))

    n_exams = len(df) - len(missing_pred) - len(missing_gt)
    print(f"\nDataset : {os.path.basename(test_csv)}")
    print(f"Exams   : {n_exams} processed ({len(missing_pred)} predictions missing, {len(missing_gt)} GT missing)")
    print(f"\n{'Class':<12} {'N exams':>9}  {'Mean DSC':>10}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 60)

    overall = []
    for k, name in CLASS_NAMES.items():
        pairs = scores[k]
        if not pairs:
            print(f"  {name:<10}  {'—':>9}  {'—':>10}  {'—':>8}  {'—':>8}  {'—':>8}")
            continue
        arr = np.array([v for _, v in pairs])
        overall.extend(arr.tolist())
        print(f"  {name:<10}  {len(arr):>9}  {arr.mean():>10.4f}  {arr.std():>8.4f}  {arr.min():>8.4f}  {arr.max():>8.4f}")

    print("-" * 60)
    if overall:
        print(f"  {'macro avg':<10}  {'':>9}  {np.mean(overall):>10.4f}")
    print()

    if show_worst > 0:
        print("=" * 60)
        print(f" WORST {show_worst} EXAMS PER CLASS")
        print("=" * 60)
        for k, name in CLASS_NAMES.items():
            if not scores[k]:
                continue
            print(f"\n  {name.upper()}:")
            print(f"  {'Exam ID':<50}  {'DSC':>8}")
            print(f"  {'-' * 60}")
            for exam_id, dsc in sorted(scores[k], key=lambda x: x[1])[:show_worst]:
                print(f"  {exam_id:<50}  {dsc:>8.4f}")
        print()

    if missing_pred:
        print(f"WARNING: {len(missing_pred)} predictions missing:")
        for e in missing_pred[:5]:
            print(f"  {e}")
        if len(missing_pred) > 5:
            print(f"  ... and {len(missing_pred) - 5} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inference_dir", required=True, help="Folder with the *_pred_mask.nii.gz files")
    parser.add_argument("--test_csv", required=True, help="Test CSV with image_path and mask_path")
    parser.add_argument("--show_worst", type=int, default=0, help="List the N worst exams per class")
    args = parser.parse_args()
    analyze(args.inference_dir, args.test_csv, show_worst=args.show_worst)
