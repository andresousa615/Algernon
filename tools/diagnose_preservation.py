# -*- coding: utf-8 -*-
"""
Diagnose region preservation in the anonymisation.

Answers one question: when a region is asked to be preserved, which operation
still touches its voxels, and how many. It runs the AnonymizationOfficer once
per isolated region, compares the output with the original volume voxel by
voxel, and crosses the changed voxels with the predicted label map.

Usage, on an exam already processed by inference.py:

    python tools/diagnose_preservation.py \
        --image     /path/to/raw.nii.gz \
        --pred_mask <inference_dir>/<exam_id>_pred_mask.nii.gz \
        --preserve  eyes --margin 0 [--anon <inference_dir>/<exam_id>_anon.nii.gz]
"""

import argparse

import nibabel as nib
import numpy as np
from scipy import ndimage

from anonymization.officer import ALL_REGIONS, REGION_LABELS, AnonymizationOfficer, resolve_regions
from data.dataset import resample


def bbox(mask):
    if not mask.any():
        return None
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    return tuple(f"{a}-{b}" for a, b in zip(lo, hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Original volume (NIfTI)")
    ap.add_argument("--pred_mask", required=True, help="<exam>_pred_mask.nii.gz from inference.py")
    ap.add_argument("--preserve", default="eyes", help="Comma-separated regions to preserve")
    ap.add_argument("--margin", type=int, default=0)
    ap.add_argument("--anon", default=None,
                    help="Optional <exam>_anon.nii.gz written by the run, to check that it matches this code")
    args = ap.parse_args()

    img = resample(nib.load(args.image)).get_fdata()
    pred = nib.load(args.pred_mask).get_fdata().astype(np.uint8)

    if img.shape != pred.shape:
        print(f"!! SHAPE MISMATCH — image {img.shape} vs mask {pred.shape}")
        print("   The pred_mask must have the same orientation and resolution as the image.")
        return

    print("=" * 70)
    print("PREDICTED LABEL MAP")
    print("=" * 70)
    total = img.size
    for name in ALL_REGIONS:
        lbl = REGION_LABELS[name]
        m = (pred == lbl)
        print(f"  {name:<6s} (label {lbl}): {m.sum():>9d} voxels ({100 * m.sum() / total:.4f}%)  bbox X/Y/Z = {bbox(m)}")
    print(f"  background (label 0): {(pred == 0).sum():>9d} voxels")

    preserved = [r.strip().lower() for r in args.preserve.split(",") if r.strip()]
    keep_mask = np.zeros(pred.shape, dtype=bool)
    for r in preserved:
        keep_mask |= (pred == REGION_LABELS[r])

    if not keep_mask.any():
        print(f"\n!! Region(s) {preserved} DO NOT EXIST in this exam's prediction.")
        print("   Nothing to preserve — whatever looks removed has another label.")

    print()
    print("=" * 70)
    print(f"WHICH OPERATION TOUCHES THE VOXELS OF: {', '.join(preserved)}")
    print("=" * 70)
    print("  Each line runs the anonymisation with ONE region active and counts how")
    print("  many voxels of the preserved region differ from the original.")
    print()

    for name in ALL_REGIONS:
        out = AnonymizationOfficer(img, pred, regions=(name,), protect_preserved=False).anonymize()
        changed = (out != img)
        hit = changed & keep_mask
        flag = "  <-- this one" if hit.sum() else ""
        print(f"  only {name:<6s}: changes {changed.sum():>8d} voxels in total, "
              f"of which {hit.sum():>7d} belong to the preserved region{flag}")

    print()
    print("=" * 70)
    print("RESULT WITH PRESERVATION ACTIVE")
    print("=" * 70)
    regions = resolve_regions(preserve=",".join(preserved))
    out = AnonymizationOfficer(img, pred, regions=regions, protect_preserved=True,
                               preserve_margin=args.margin).anonymize()
    changed = (out != img)
    hit = changed & keep_mask

    print(f"  anonymised regions   : {', '.join(regions) if regions else 'none'}")
    print(f"  protection margin    : {args.margin} voxels")
    print(f"  voxels changed       : {changed.sum()}")
    print(f"  of which labelled as the preserved region: {hit.sum()}")
    if hit.sum() == 0:
        print("  -> label-based preservation works. If the region still looks altered,")
        print("     the voxels in question carry ANOTHER label — see the table above.")
    else:
        print("  -> BUG: voxels of the preserved region are still altered.")

    if args.anon:
        print()
        print("=" * 70)
        print("DOES THE FILE WRITTEN BY THE RUN MATCH THIS CODE?")
        print("=" * 70)
        written = resample(nib.load(args.anon)).get_fdata()
        if written.shape != img.shape:
            print(f"  shape mismatch: {written.shape} vs {img.shape} — not comparable")
        else:
            diff = (written != out)
            w_hit = (written != img) & keep_mask
            print(f"  voxels where the written file differs from the recomputed one: {diff.sum()}")
            print(f"  preserved-region voxels altered IN THE WRITTEN FILE: {w_hit.sum()}")
            if diff.sum() == 0:
                print("  -> the run used exactly this code.")
            elif w_hit.sum() > 0 and hit.sum() == 0:
                print("  -> the written file removed the preserved region but this code does not:")
                print("     the run used a different version of anonymization/officer.py.")
            else:
                print("  -> difference for another reason (the ear noise is random between runs;")
                print("     ignore if the count is of the order of the ear cover).")

    print()
    print("=" * 70)
    print("CONTEXT — what surrounds the preserved region")
    print("=" * 70)
    for it in (1, 2, 3, 5):
        ring = ndimage.binary_dilation(keep_mask, iterations=it) & ~keep_mask
        if not ring.any():
            continue
        vals, counts = np.unique(pred[ring], return_counts=True)
        desc = ", ".join(
            f"{'background' if v == 0 else [k for k, l in REGION_LABELS.items() if l == v][0]}={c}"
            for v, c in zip(vals, counts))
        print(f"  at {it} voxel(s) distance: {desc}")
    print()
    print("  Lots of 'nose' or 'ears' next to the preserved region means that is the")
    print("  neighbourhood the margin has to cover to protect it fully.")


if __name__ == "__main__":
    main()
