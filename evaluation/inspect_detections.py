# -*- coding: utf-8 -*-
"""
Show what the face detector considered a face on the anonymised renders.

The re-identification evaluation counts how many anonymised exams still have
a detectable face, and that number only means something if the detections
are real. A detector trained on photographs can fire on the shape of a head
in a render without any facial trait — the embedding built on it is then
meaningless.

For every exam where a face was detected on the anonymised render, writes a
side-by-side image (original | anonymised) with the boxes drawn and the
cosine distance annotated. File names are prefixed by the distance, so the
first files to inspect are the ones the evaluation considers most risky.

Usage:
    python evaluation/inspect_detections.py \
        --pairs_dir <run>/pairs_2d --out_dir <run>/detections --detector retinaface
"""

import argparse
import glob
import os

import cv2
import numpy as np


def load_deepface():
    from deepface import DeepFace
    return DeepFace


def detect(DeepFace, path, detector):
    """List of (x, y, w, h, confidence) for the detected faces."""
    try:
        faces = DeepFace.extract_faces(img_path=path, detector_backend=detector, enforce_detection=True)
    except Exception:
        return []
    return [(a.get("x", 0), a.get("y", 0), a.get("w", 0), a.get("h", 0), float(f.get("confidence", 0.0)))
            for f in faces for a in [f.get("facial_area", {})]]


def embed(DeepFace, path, model, detector):
    try:
        reps = DeepFace.represent(img_path=path, model_name=model, detector_backend=detector,
                                  enforce_detection=True)
        if not reps:
            return None
        v = np.asarray(reps[0]["embedding"], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    except Exception:
        return None


def annotate(img, boxes, label):
    img = img.copy()
    for (x, y, w, h, conf) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f"{conf:.2f}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--detector", default="retinaface")
    ap.add_argument("--model", default="ArcFace")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    DeepFace = load_deepface()
    os.makedirs(args.out_dir, exist_ok=True)

    exams = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.pairs_dir, "*")) if os.path.isdir(p))
    if args.limit:
        exams = exams[: args.limit]

    n_det, n_prot, results = 0, 0, []

    for exam in exams:
        d = os.path.join(args.pairs_dir, exam)
        p_o = os.path.join(d, f"{exam}_original_image.png")
        p_a = os.path.join(d, f"{exam}_defaced_image.png")
        if not (os.path.exists(p_o) and os.path.exists(p_a)):
            continue

        boxes_a = detect(DeepFace, p_a, args.detector)
        if not boxes_a:
            n_prot += 1
            continue
        n_det += 1

        v_o = embed(DeepFace, p_o, args.model, args.detector)
        v_a = embed(DeepFace, p_a, args.model, args.detector)
        dist = float(1.0 - np.dot(v_o, v_a)) if (v_o is not None and v_a is not None) else float("nan")

        boxes_o = detect(DeepFace, p_o, args.detector)
        img_o, img_a = cv2.imread(p_o), cv2.imread(p_a)
        if img_o is None or img_a is None:
            continue
        if img_o.shape[0] != img_a.shape[0]:
            img_a = cv2.resize(img_a, (img_o.shape[1], img_o.shape[0]))

        pair = np.hstack([annotate(img_o, boxes_o, "ORIGINAL"), annotate(img_a, boxes_a, "ANONYMISED")])
        cv2.putText(pair, f"dist={dist:.3f}  {exam}", (8, pair.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        name = f"{dist:06.3f}_{exam}.png" if dist == dist else f"nan_{exam}.png"
        cv2.imwrite(os.path.join(args.out_dir, name), pair)
        results.append((dist, exam, max(b[4] for b in boxes_a)))

    print(f"\nexams with a face detected on the anonymised render : {n_det}")
    print(f"exams without a detected face (protected)           : {n_prot}")
    print(f"images written to                                   : {args.out_dir}")

    if results:
        results.sort()
        print("\n10 cases most similar to the original (inspect first):")
        print(f"  {'dist':>7s}  {'conf':>5s}  exam")
        for dist, exam, conf in results[:10]:
            print(f"  {dist:7.3f}  {conf:5.2f}  {exam}")
        confs = [c for _, _, c in results]
        print(f"\ndetection confidence: mean {np.mean(confs):.2f} | min {np.min(confs):.2f} | max {np.max(confs):.2f}")
        print("Low, uniform confidences suggest spurious detections; high ones suggest real facial structure.")


if __name__ == "__main__":
    main()
