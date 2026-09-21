# -*- coding: utf-8 -*-
"""
Phase 5 of the pipeline: defacing score with a CNN face detector (dlib via
face_recognition).

For every exam pair rendered by render_pairs_ddp.py:
  1. the ORIGINAL render must contain a face with the central landmarks
     (nose bridge/tip, both eyes, upper lip) — otherwise the exam is rejected
     as a control failure and does not count;
  2. the ANONYMISED render is considered successfully defaced when no such face
     is detected, or when a face is detected but its encoding is farther than
     TOLERANCE from the original.

    defacing score = successfully defaced / validated originals  (%)

Runs standalone or under torchrun (exams split across ranks).

Usage:
    torchrun --nproc_per_node=2 evaluation/defacing_score_ddp.py \
        --dir_pairs <run>/pairs_2d --output_report <run>/defacing_report.txt
"""
import argparse
import logging
import os
import time
from datetime import timedelta

import face_recognition
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group

TOLERANCE = 0.6          # face-distance threshold (lower = stricter)
DETECTION_MODEL = "cnn"  # 'cnn' (GPU) or 'hog' (CPU)
CENTRAL_LANDMARKS = ("nose_bridge", "nose_tip", "left_eye", "right_eye", "top_lip")

log = logging.getLogger("defacing_score")


def ddp_setup():
    """Initialise the process group; falls back to a single process without torchrun."""
    if "WORLD_SIZE" not in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)   # makes dlib's CNN use the right GPU
    init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return local_rank, global_rank, world_size


def detect_real_face(image_path: str):
    """
    Return (found, encoding). A detection only counts when the central facial
    landmarks are present, which filters out boxes fired on the head outline.
    """
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=DETECTION_MODEL)
    if not face_locations:
        return False, None

    landmarks_list = face_recognition.face_landmarks(image, face_locations, model="large")
    for i, landmarks in enumerate(landmarks_list):
        if all(feature in landmarks for feature in CENTRAL_LANDMARKS):
            encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[i]])[0]
            return True, encoding
    return False, None


def calculate_defacing_score(dir_pairs: str, output_report: str) -> None:
    _, global_rank, world_size = ddp_setup()
    start_time = time.time()

    exams = sorted(d for d in os.listdir(dir_pairs) if os.path.isdir(os.path.join(dir_pairs, d)))
    my_exams = np.array_split(exams, world_size)[global_rank]
    log.info(f"[rank {global_rank}] processing {len(my_exams)} exams")

    local = {"tested": 0, "orig_with_face": 0, "orig_without_face": 0,
             "defaced_ok": 0, "defaced_fail": 0, "control_failures": [], "defacing_failures": []}

    for exam_id in my_exams:
        folder = os.path.join(dir_pairs, exam_id)
        path_orig = os.path.join(folder, f"{exam_id}_original_image.png")
        path_def = os.path.join(folder, f"{exam_id}_defaced_image.png")
        if not (os.path.exists(path_orig) and os.path.exists(path_def)):
            continue
        local["tested"] += 1

        try:
            face_orig, enc_orig = detect_real_face(path_orig)
            if not face_orig:
                local["orig_without_face"] += 1
                local["control_failures"].append(exam_id)
                continue
            local["orig_with_face"] += 1

            face_def, enc_def = detect_real_face(path_def)
            if not face_def or face_recognition.face_distance([enc_orig], enc_def)[0] > TOLERANCE:
                local["defaced_ok"] += 1
            else:
                local["defaced_fail"] += 1
                local["defacing_failures"].append(exam_id)
        except Exception as e:
            log.error(f"[rank {global_rank}] {exam_id}: {e}")

    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.gather_object(local, gathered if global_rank == 0 else None, dst=0)
    else:
        gathered = [local]

    if global_rank == 0:
        final = {k: sum(s[k] for s in gathered) for k in
                 ("tested", "orig_with_face", "orig_without_face", "defaced_ok", "defaced_fail")}
        final["control_failures"] = [x for s in gathered for x in s["control_failures"]]
        final["defacing_failures"] = [x for s in gathered for x in s["defacing_failures"]]

        elapsed = time.time() - start_time
        score = 100.0 * final["defaced_ok"] / final["orig_with_face"] if final["orig_with_face"] else 0.0

        report = (
            "=======================================================\n"
            "DEFACING SCORE REPORT\n"
            "=======================================================\n"
            f"Pairs processed:                    {final['tested']}\n"
            f"Originals validated (landmarks):    {final['orig_with_face']}\n"
            f"Originals rejected (no face found): {final['orig_without_face']}\n"
            f"Rejected originals: {final['control_failures']}\n\n"
            "-------------------------------------------------------\n"
            f"Successfully defaced:               {final['defaced_ok']}\n"
            f"Identity still detectable:          {final['defaced_fail']}\n"
            f"Exams with a face still detected: {final['defacing_failures']}\n\n"
            f"DEFACING SCORE: {score:.2f}%\n\n"
            f"Total time: {timedelta(seconds=int(elapsed))}\n"
            "=======================================================\n"
        )
        print("\n" + report)
        with open(output_report, "w", encoding="utf-8") as f:
            f.write(report)
        log.info(f"Report saved to {output_report}")

    if world_size > 1:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir_pairs", required=True, help="Folder with the exam pairs and rendered PNGs")
    parser.add_argument("--output_report", required=True, help="Path of the text report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    calculate_defacing_score(args.dir_pairs, args.output_report)
