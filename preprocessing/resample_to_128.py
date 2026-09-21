# -*- coding: utf-8 -*-
"""
Resample every exam of a dataset to the network input size (default 128^3).

Reads from <source_dir> and writes to <dest_dir> keeping the folder layout:

    <dest_dir>/<exam>/raw.nii.gz
    <dest_dir>/<exam>/training_masks/mask_4_classes.nii.gz   (if present)

Images are resampled with trilinear interpolation (order=1), label masks with
nearest neighbour (order=0). The affine is rescaled so voxel spacing stays
consistent with the new grid. Exams already at the target size are copied.

The mask is optional: exams without one are still resampled (inference-only
sets), unless --require_mask is given.

Usage:
    python preprocessing/resample_to_128.py <source_dir> <dest_dir> [--size 128 128 128]
"""
import argparse
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

IMAGE_FILENAME = "raw.nii.gz"
MASK_RELPATH = os.path.join("training_masks", "mask_4_classes.nii.gz")


def zoom_volume(data: np.ndarray, target: tuple, order: int) -> np.ndarray:
    if data.shape[:3] == target:
        return data
    factors = tuple(target[i] / data.shape[i] for i in range(3))
    if data.ndim == 4:
        channels = [ndimage.zoom(data[..., c], factors, order=order) for c in range(data.shape[3])]
        return np.stack(channels, axis=-1)
    return ndimage.zoom(data, factors, order=order)


def process_exam(exam: str, src_dir: str, dst_dir: str, target: tuple, require_mask: bool) -> str:
    src_exam = os.path.join(src_dir, exam)
    dst_exam = os.path.join(dst_dir, exam)

    img_src = os.path.join(src_exam, IMAGE_FILENAME)
    msk_src = os.path.join(src_exam, MASK_RELPATH)
    if not os.path.exists(img_src):
        return "SKIP (no image)"
    if require_mask and not os.path.exists(msk_src):
        return "SKIP (no mask)"

    os.makedirs(dst_exam, exist_ok=True)

    img_obj = nib.load(img_src)
    img_data = img_obj.get_fdata(dtype=np.float32)
    orig_shape = img_data.shape[:3]

    img_resampled = zoom_volume(img_data, target, order=1)
    new_affine = np.copy(img_obj.affine)
    for i in range(3):
        new_affine[i, i] *= orig_shape[i] / target[i]
    nib.save(nib.Nifti1Image(img_resampled, new_affine), os.path.join(dst_exam, IMAGE_FILENAME))

    if os.path.exists(msk_src):
        msk_data = nib.load(msk_src).get_fdata(dtype=np.float32)
        msk_resampled = zoom_volume(np.round(msk_data), target, order=0)
        dst_msk = os.path.join(dst_exam, MASK_RELPATH)
        os.makedirs(os.path.dirname(dst_msk), exist_ok=True)
        nib.save(nib.Nifti1Image(msk_resampled, new_affine), dst_msk)

    return "copied (already at target size)" if orig_shape == target else f"{orig_shape} -> {target}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_dir")
    ap.add_argument("dest_dir")
    ap.add_argument("--size", type=int, nargs=3, default=(128, 128, 128), metavar=("D", "H", "W"))
    ap.add_argument("--require_mask", action="store_true", help="Skip exams without a ground-truth mask")
    args = ap.parse_args()

    target = tuple(args.size)
    os.makedirs(args.dest_dir, exist_ok=True)

    exams = sorted(f for f in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir, f)))
    print(f"Source : {args.source_dir}\nDest   : {args.dest_dir}\nExams  : {len(exams)}\n")

    ok = skip = 0
    for i, exam in enumerate(exams, 1):
        result = process_exam(exam, args.source_dir, args.dest_dir, target, args.require_mask)
        if result.startswith("SKIP"):
            skip += 1
        else:
            ok += 1
        print(f"[{i:>3}/{len(exams)}] {exam:<50} {result}")

    print(f"\nDone: {ok} converted, {skip} skipped. Output in {args.dest_dir}")


if __name__ == "__main__":
    main()
