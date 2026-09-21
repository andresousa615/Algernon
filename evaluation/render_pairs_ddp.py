# -*- coding: utf-8 -*-
"""
Phase 4 of the pipeline: build (original, anonymised) pairs for every exam and
render frontal 2D views of both, plus a 4-view collage of the anonymised one.

The PNGs feed the defacing score (evaluation/defacing_score_ddp.py) and the
re-identification analysis (evaluation/reidentification.py).

Output layout:
    <dir_pairs>/<exam_id>/<exam_id>_original.nii.gz
    <dir_pairs>/<exam_id>/<exam_id>_defaced.nii.gz
    <dir_pairs>/<exam_id>/<exam_id>_original_image.png
    <dir_pairs>/<exam_id>/<exam_id>_defaced_image.png
    <dir_pairs>/<exam_id>/<exam_id>_compiled_image.png

Runs standalone or under torchrun (gloo backend; exams split across ranks).
Needs an off-screen display: wrap with `xvfb-run -a` on headless nodes.

Usage:
    torchrun --nproc_per_node=2 evaluation/render_pairs_ddp.py \
        --dir_defaced <run>/inference --test_csvs data/test.csv --dir_pairs <run>/pairs_2d
"""
import argparse
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch.distributed as dist

from evaluation.render_3d import load_volume, render_compiled_image, render_view

log = logging.getLogger("render_pairs")

# Fixed frontal camera used for the single-view renders (kept identical across
# exams so the face detector sees the same framing).
CAMERA_POSITION = [
    (95.40000379085541, 887.52, 84.39690110118934),
    (95.40000379085541, 119.5, 127.5),
    (0.0, 0.0, 1.0),
]
FRONTAL_WINDOW = 1200


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def build_original_lookup(test_csvs: list) -> dict:
    """{exam_id: image_path} from the test CSVs (exam_id = parent folder name)."""
    lookup = {}
    for csv_path in test_csvs:
        for _, row in pd.read_csv(csv_path).iterrows():
            lookup[os.path.basename(os.path.dirname(row["image_path"]))] = row["image_path"]
    return lookup


def prepare_pairs(dir_defaced, dir_pairs, original_lookup=None, dir_original=None) -> int:
    """
    Rank 0 only. For every *_anon.nii(.gz) in dir_defaced, locate the original
    volume and copy both into <dir_pairs>/<exam_id>/.

    The original is resolved from the test CSVs (preferred) or, as a fallback,
    from <dir_original>/<exam_id>/raw.nii.gz.
    """
    os.makedirs(dir_pairs, exist_ok=True)
    count = 0

    for f in sorted(os.listdir(dir_defaced)):
        if f.endswith("_anon.nii.gz"):
            exam_id, ext = f[: -len("_anon.nii.gz")], ".nii.gz"
        elif f.endswith("_anon.nii"):
            exam_id, ext = f[: -len("_anon.nii")], ".nii"
        else:
            continue

        path_orig = None
        if original_lookup and exam_id in original_lookup and os.path.exists(original_lookup[exam_id]):
            path_orig = original_lookup[exam_id]
        elif dir_original:
            for cand in ("raw.nii.gz", "raw.nii", "image.nii.gz", "image.nii"):
                p = os.path.join(dir_original, exam_id, cand)
                if os.path.exists(p):
                    path_orig = p
                    break

        if path_orig is None:
            log.warning(f"Original not found for {exam_id}; skipped.")
            continue

        target = os.path.join(dir_pairs, exam_id)
        os.makedirs(target, exist_ok=True)
        orig_ext = ".nii.gz" if path_orig.endswith(".nii.gz") else ".nii"
        dst_defaced = os.path.join(target, f"{exam_id}_defaced{ext}")
        dst_original = os.path.join(target, f"{exam_id}_original{orig_ext}")
        if not os.path.exists(dst_defaced):
            shutil.copy2(os.path.join(dir_defaced, f), dst_defaced)
        if not os.path.exists(dst_original):
            shutil.copy2(path_orig, dst_original)
        count += 1

    log.info(f"{count} pairs ready in {dir_pairs}")
    return count


def render_all(dir_pairs: str) -> None:
    rank, world_size = get_rank(), get_world_size()

    exams = sorted(d for d in os.listdir(dir_pairs) if os.path.isdir(os.path.join(dir_pairs, d)))
    if not exams:
        if rank == 0:
            log.warning("No exams found to render.")
        return

    my_exams = np.array_split(exams, world_size)[rank]
    log.info(f"[rank {rank}] rendering {len(my_exams)} exams")

    for exam_id in my_exams:
        folder = os.path.join(dir_pairs, exam_id)
        for f in os.listdir(folder):
            if not f.endswith((".nii", ".nii.gz")) or "_mask" in f:
                continue
            nifti_path = os.path.join(folder, f)
            base = f[:-7] if f.endswith(".nii.gz") else f[:-4]
            output_path = os.path.join(folder, f"{base}_image.png")

            try:
                grid, params = load_volume(nifti_path)
                img = render_view(grid, CAMERA_POSITION, params, window_size=FRONTAL_WINDOW)
                from PIL import Image
                Image.fromarray(img).save(output_path)
            except Exception as e:
                log.error(f"[rank {rank}] failed on {f}: {e}")

            if "_defaced" in f:
                try:
                    render_compiled_image(nifti_path, os.path.join(folder, f"{exam_id}_compiled_image.png"))
                except Exception as e:
                    log.error(f"[rank {rank}] collage failed for {exam_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir_defaced", required=True, help="Folder with the *_anon.nii.gz volumes")
    parser.add_argument("--dir_pairs", required=True, help="Output folder for pairs and PNGs")
    parser.add_argument("--test_csvs", nargs="*", default=[], help="Test CSVs with image_path (preferred)")
    parser.add_argument("--dir_original", default=None, help="Fallback: folder with <exam_id>/raw.nii.gz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="gloo")

    if get_rank() == 0:
        lookup = build_original_lookup(args.test_csvs) if args.test_csvs else None
        prepare_pairs(args.dir_defaced, args.dir_pairs, original_lookup=lookup, dir_original=args.dir_original)

    if is_dist():
        dist.barrier()

    render_all(args.dir_pairs)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
