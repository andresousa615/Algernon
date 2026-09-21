# -*- coding: utf-8 -*-
"""
Re-identification risk on the 2D renders produced by render_pairs_ddp.py.

The defacing score only answers "is a face still detectable?". This script
measures identity: how confidently a face-recognition model links an
anonymised exam to the subject it came from. Three parts, in order:

  1. THRESHOLD CALIBRATION on the renders of this dataset.
     Face-recognition models are trained on photographs, not on MRI surface
     renders, so a literature threshold does not transfer. When subjects have
     several sessions, same-subject pairs (genuine) and different-subject
     pairs (impostor) of ORIGINAL renders are compared and the threshold that
     best separates them is chosen.

  2. VERIFICATION — original vs. anonymised render of the same exam (the
     protocol used in the refacing literature). Reports the mean distance and
     the fraction of potentially identifiable exams.

  3. IDENTIFICATION — anonymised render against a gallery of all originals.
     Closer to a real attack: "whose face is this?". Reports the rank-1 hit
     rate, which is the number that actually measures risk.

Exam / subject naming: the subject id is the exam id up to the first "__"
(<subject>__<session>...); exams without the separator are one session per
subject.

Usage:
    python evaluation/reidentification.py --pairs_dir <run>/pairs_2d \
        --out <run>/reidentification.txt --csv <run>/reidentification.csv \
        [--model ArcFace] [--detector retinaface] [--exclude_file exclusions.txt]

Requires deepface (see evaluation/README.md).
"""

import argparse
import csv
import glob
import itertools
import os

import numpy as np

MODEL_NAME = "ArcFace"

# The detector matters as much as the recognition model: surface renders are
# not photographs, and a weak detector misses originals with a perfectly
# visible face, silently dropping them from the evaluation. If the report
# shows many originals without a detected face, switch to 'retinaface'/'mtcnn'.
DETECTOR = "retinaface"


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def subject_of(exam_id: str) -> str:
    return exam_id.split("__")[0] if "__" in exam_id else exam_id


def restrict(d: dict, prefix: str | None) -> dict:
    """Keep only exams whose id starts with `prefix` (None = keep all)."""
    return d if not prefix else {k: v for k, v in d.items() if k.upper().startswith(prefix.upper())}


def load_exclusions(path: str | None) -> tuple[dict, dict]:
    """
    Read an exclusion file: one `<id>: <reason>` per line (`#` comments allowed).
    Ids containing "__" are treated as exam ids, the others as subject ids.

    Exclusions are for DATA defects, not results: renders where the facial
    surface is not reconstructed (skull, no skin) make the detector fire on
    nothing and produce a degenerate embedding that sits close to everyone.
    Use the "hub" warning in the report to find candidates, confirm them in
    the images, then list them here.
    """
    subjects, exams = {}, {}
    if not path:
        return subjects, exams
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, reason = line.partition(":")
            key, reason = key.strip(), reason.strip() or "excluded"
            (exams if "__" in key else subjects)[key] = reason
    return subjects, exams


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def load_model():
    """Import deepface, showing the real cause on failure."""
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception:
        import traceback
        print("\n" + "=" * 72)
        print("FAILED TO IMPORT deepface — original error below")
        print("=" * 72)
        traceback.print_exc()
        print("=" * 72)
        print("See evaluation/README.md for installation notes.")
        raise SystemExit(1)


def embed(DeepFace, path):
    """
    Return (vector, True) if a face was detected, (None, False) otherwise.
    A failed detection is NOT an error: on a well-anonymised exam it is the
    expected outcome and is counted as such.
    """
    try:
        reps = DeepFace.represent(img_path=path, model_name=MODEL_NAME,
                                  detector_backend=DETECTOR, enforce_detection=True)
        if not reps:
            return None, False
        v = np.asarray(reps[0]["embedding"], dtype=np.float64)
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v), True
    except Exception:
        return None, False


def cosine_distance(a, b):
    return float(1.0 - np.dot(a, b))


def collect(pairs_dir, cache_path, limit=None):
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        print(f"[cache] embeddings loaded from {cache_path}")
        return d["orig"].item(), d["anon"].item(), d["fail"].item()

    DeepFace = load_model()
    orig, anon, fail = {}, {}, {}

    exams = sorted(os.path.basename(p) for p in glob.glob(os.path.join(pairs_dir, "*")) if os.path.isdir(p))
    if limit:
        exams = exams[:limit]
    print(f"Processing {len(exams)} exams from {pairs_dir}\n")

    for i, exam in enumerate(exams, 1):
        d = os.path.join(pairs_dir, exam)
        p_o = os.path.join(d, f"{exam}_original_image.png")
        p_a = os.path.join(d, f"{exam}_defaced_image.png")
        if not (os.path.exists(p_o) and os.path.exists(p_a)):
            continue

        v_o, ok_o = embed(DeepFace, p_o)
        v_a, ok_a = embed(DeepFace, p_a)

        if ok_o:
            orig[exam] = v_o
        else:
            fail[exam] = "no face detected in the ORIGINAL — excluded"
            continue

        if ok_a:
            anon[exam] = v_a
        else:
            fail[exam] = "no face detected in the anonymised render — protected"

        if i % 20 == 0:
            print(f"  {i}/{len(exams)}")

    if cache_path:
        np.savez(cache_path, orig=orig, anon=anon, fail=fail)
        print(f"\n[cache] embeddings saved to {cache_path}")
    return orig, anon, fail


# ---------------------------------------------------------------------------
# 0. Quality control
# ---------------------------------------------------------------------------

def apply_exclusions(orig, anon, fail, excluded_subjects, excluded_exams, lines):
    """Drop excluded subjects/exams from every dictionary (including `fail`)."""
    out = [e for e in list(orig) + list(fail) if subject_of(e) in excluded_subjects or e in excluded_exams]
    if not out:
        return orig, anon, fail
    lines.append("=" * 72)
    lines.append("0. EXCLUSIONS")
    lines.append("=" * 72)
    n0 = len(orig)
    for s in sorted(excluded_subjects):
        n = sum(1 for e in out if subject_of(e) == s)
        if n:
            lines.append(f"  subject {s}: {n} exam(s) — {excluded_subjects[s]}")
    for e in sorted(excluded_exams):
        if e in out:
            lines.append(f"  exam {e} — {excluded_exams[e]}")
    for e in out:
        orig.pop(e, None)
        anon.pop(e, None)
        fail.pop(e, None)
    lines.append(f"  gallery: {n0} -> {len(orig)} originals")
    lines.append("")
    return orig, anon, fail


def warn_hubs(orig, anon, lines, factor=10.0):
    """
    Flag originals that are the nearest neighbour of far too many queries.
    Such a render is not measuring similarity — its face was not reconstructed.
    Nothing is excluded here; confirm in the image and add to the exclusion file.
    """
    ids = sorted(orig)
    if len(ids) < 10 or len(anon) < 10:
        return
    M = np.stack([orig[e] for e in ids])
    count = {}
    for v in anon.values():
        i = int(np.argmin(1.0 - M @ v))
        count[ids[i]] = count.get(ids[i], 0) + 1
    expected = len(anon) / len(ids)
    limit = max(3, factor * expected)
    bad = sorted((e for e, c in count.items() if c >= limit), key=lambda e: -count[e])
    if not bad:
        return
    lines.append("  !! WARNING — originals suspected of not encoding identity:")
    for e in bad:
        lines.append(f"     {count[e]:>4d}x nearest neighbour (expected {expected:.2f})  {e}")
    lines.append("     Confirm in the image and, if defective, add to the exclusion file.")
    lines.append("")


# ---------------------------------------------------------------------------
# 1. Threshold calibration
# ---------------------------------------------------------------------------

def calibrate(orig, lines):
    same, diff = [], []
    for a, b in itertools.combinations(sorted(orig), 2):
        d = cosine_distance(orig[a], orig[b])
        (same if subject_of(a) == subject_of(b) else diff).append(d)

    lines.append("=" * 72)
    lines.append("1. THRESHOLD CALIBRATION ON THESE RENDERS")
    lines.append("=" * 72)

    if len(same) < 5:
        lines.append(f"  Only {len(same)} same-subject pairs — not enough to calibrate.")
        lines.append("  Without calibration no literature threshold should be adopted (it")
        lines.append("  depends on the renderer). Only the distance distribution is reported.")
        return None

    same, diff = np.array(same), np.array(diff)
    lines.append(f"  same-subject pairs       : {len(same):>6d} | distance {same.mean():.4f} ± {same.std():.4f}")
    lines.append(f"  different-subject pairs  : {len(diff):>6d} | distance {diff.mean():.4f} ± {diff.std():.4f}")

    # Threshold minimising the sum of error RATES, not counts: the classes are
    # very unbalanced (hundreds of times more impostor pairs), and minimising
    # counts would push the threshold towards zero and under-estimate the risk.
    grid = np.linspace(min(same.min(), diff.min()), max(same.max(), diff.max()), 2000)
    fnr = np.array([(same > t).mean() for t in grid])   # genuine, not recognised
    fpr = np.array([(diff <= t).mean() for t in grid])  # impostor, confused
    thr = float(grid[int(np.argmin(fnr + fpr))])

    fn = float((same > thr).mean())
    fp = float((diff <= thr).mean())
    sep = (diff.mean() - same.mean()) / np.sqrt(0.5 * (same.var() + diff.var()))

    lines.append(f"  distribution separation     : d' = {sep:.2f}")
    lines.append(f"  -> chosen threshold         : {thr:.4f}")
    lines.append(f"     false negatives          : {100 * fn:.2f} %  (same subject, not recognised)")
    lines.append(f"     false positives          : {100 * fp:.2f} %  (different subjects, confused)")
    lines.append("")
    lines.append("  False positives are the method's own error: the fraction of cases that")
    lines.append("  would be counted as identifiable without any identity leak.")
    lines.append("  d' above 2 means the model separates identities on these renders;")
    lines.append("  below that, nothing that follows should be interpreted.")
    return thr, fp


# ---------------------------------------------------------------------------
# 2. Verification   3. Identification
# ---------------------------------------------------------------------------

def verification(orig, anon, fail, thr, lines, rows, fpr=None):
    lines.append("")
    lines.append("=" * 72)
    lines.append("2. VERIFICATION — original vs. anonymised render of the same exam")
    lines.append("=" * 72)

    n_prot = sum(1 for v in fail.values() if "protected" in v)

    gal_ids = sorted(orig)
    gal = np.stack([orig[e] for e in gal_ids])
    gal_sub = [subject_of(e) for e in gal_ids]
    i_gal = {e: i for i, e in enumerate(gal_ids)}

    # The distance to the own original is read FROM THE SAME matrix product
    # that produces the ranking: a separate np.dot sums in a different order
    # and the last digits differ, which can push the own original to rank 2.
    ds = []
    for exam, v_a in anon.items():
        all_d = 1.0 - gal @ v_a
        d = float(all_d[i_gal[exam]])
        ds.append(d)

        rank = int((all_d < d).sum()) + 1
        others = [i for i, s in enumerate(gal_sub) if s != subject_of(exam)]
        i_min = min(others, key=lambda i: all_d[i]) if others else None
        d_other = float(all_d[i_min]) if i_min is not None else float("nan")

        rows.append({"Exam_ID": exam,
                     "cos_dist_orig_vs_anon": round(d, 4),
                     "rank_of_own_original": rank,
                     "dist_to_nearest_other_subject": round(d_other, 4),
                     "nearest_stranger": gal_ids[i_min] if i_min is not None else "",
                     "stranger_closer_than_own": int(d_other < d),
                     "identifiable_1to1": int(thr is not None and d <= thr),
                     "identifiable_1toN": int(rank == 1)})

    total = len(ds) + n_prot
    lines.append(f"  exams evaluated                       : {total}")
    lines.append(f"  no face detected in anonymised render : {n_prot}  ({100 * n_prot / total if total else 0:.1f} %)")
    if ds:
        ds = np.array(ds)
        lines.append(f"  cosine distance (where a face exists) : {ds.mean():.4f} ± {ds.std():.4f}")
        if thr is not None:
            ident = int((ds <= thr).sum())
            lines.append("")
            lines.append(f"  potentially identifiable              : {ident}")
            lines.append(f"    of {total} exams evaluated           : {100 * ident / total:.1f} %")
            lines.append(f"    of {len(ds)} with a detectable face    : {100 * ident / len(ds):.1f} %")
            if fpr is not None:
                lines.append(f"    expected by chance alone            : {fpr * len(ds):.1f}"
                             f"  (false-positive rate of the threshold)")
                lines.append("")
                lines.append("  Both denominators matter: the first is the fraction of the test set at")
                lines.append("  risk, the second the failure rate conditioned on a face being left")
                lines.append("  detectable. The chance value separates identity residue from a")
                lines.append("  threshold artefact.")
    if rows:
        ranks = [r["rank_of_own_original"] for r in rows]
        top1 = sum(1 for r in ranks if r == 1)
        med = sorted(ranks)[len(ranks) // 2]
        worse = sum(r["stranger_closer_than_own"] for r in rows)
        d_out = sorted(r["dist_to_nearest_other_subject"] for r in rows)
        med_out = d_out[len(d_out) // 2]
        under = sum(1 for x in d_out if thr is not None and x <= thr)

        lines.append("")
        lines.append("  --- Is the 1:1 criterion valid on these data? ---")
        lines.append(f"  median distance to own original           : {np.median(ds):.3f}")
        lines.append(f"  median distance to nearest stranger       : {med_out:.3f}")
        lines.append(f"  exams with a STRANGER closer than the own original: {worse} of {len(rows)}"
                     f"  ({100 * worse / len(rows):.0f} %)")
        if thr is not None:
            lines.append(f"  exams with at least one stranger below the threshold: {under} of {len(rows)}"
                         f"  ({100 * under / len(rows):.0f} %)")
        lines.append(f"  rank of own original: median {med} of {len(gal_ids)} | rank 1: {top1}")
        lines.append("")
        lines.append("  Reading: if most exams have strangers below the threshold and closer than")
        lines.append("  their own original, a small distance to the own original is not identity")
        lines.append("  residue — the embeddings no longer encode who the person is. The 1:1")
        lines.append("  count is then a false alarm and the gallery identification below is the")
        lines.append("  number that counts.")

        strangers = {}
        for r in rows:
            strangers[r["nearest_stranger"]] = strangers.get(r["nearest_stranger"], 0) + 1
        top = sorted(strangers.items(), key=lambda kv: -kv[1])[:3]
        if top and top[0][1] > 1:
            lines.append("")
            lines.append("  Originals that repeatedly appear as the nearest stranger:")
            for e, c in top:
                lines.append(f"    {c:>3d}x  {e}")
            lines.append("  High repetition indicates degenerate embeddings, not real similarity.")

    lines.append("")
    lines.append("  Larger distance = better protected. An exam with no detectable face is")
    lines.append("  the best possible outcome and does not enter the distance mean.")


def identification(orig, anon, fail, thr, lines):
    lines.append("")
    lines.append("=" * 72)
    lines.append("3. IDENTIFICATION — anonymised render against the gallery of originals")
    lines.append("=" * 72)

    gal_ids = sorted(orig)
    if len(gal_ids) < 2 or not anon:
        lines.append("  Gallery too small.")
        return
    gal = np.stack([orig[e] for e in gal_ids])
    gal_sub = [subject_of(e) for e in gal_ids]
    n_sub_exams = {}
    for e in gal_ids:
        n_sub_exams[subject_of(e)] = n_sub_exams.get(subject_of(e), 0) + 1

    lines.append(f"  gallery (originals): {len(gal_ids)} exams, {len(set(gal_sub))} subjects")
    lines.append(f"  chance             : {100.0 / len(set(gal_sub)):.1f} %")
    lines.append("")

    # Scenario A: the attacker holds the target's ORIGINAL exam
    hits_a, who_a = 0, []
    for exam, v in anon.items():
        d = 1.0 - gal @ v
        if gal_sub[int(np.argmin(d))] == subject_of(exam):
            hits_a += 1
            who_a.append(exam)
    n_a = len(anon)
    lines.append("  A) The attacker holds the target's ORIGINAL exam")
    lines.append(f"     queries        : {n_a}")
    lines.append(f"     rank-1 hits    : {hits_a}  ({100 * hits_a / n_a if n_a else 0:.1f} %)")
    for e in who_a:
        lines.append(f"       -> {e}")
    if who_a:
        lines.append("     Inspect each hit: it may be real identity or two defective renders")
        lines.append("     meeting. Confirm in the images.")
    lines.append("     This is the easiest scenario: the original was shared or leaked before")
    lines.append("     anonymisation.")
    lines.append("")

    # Scenario B: the attacker holds ANOTHER session of the target. Only
    # evaluable for subjects with 2+ exams in the gallery.
    queries_b = [e for e in anon if n_sub_exams.get(subject_of(e), 0) >= 2]
    lines.append("  B) The attacker holds ANOTHER SESSION of the target")
    if not queries_b:
        lines.append("     No subject with more than one exam in the gallery — not evaluable.")
    else:
        hits_b = 0
        for exam in queries_b:
            d = 1.0 - gal @ anon[exam]
            for i, e in enumerate(gal_ids):
                if e == exam:
                    d[i] = np.inf
            if gal_sub[int(np.argmin(d))] == subject_of(exam):
                hits_b += 1
        n_b = len(queries_b)

        # An exam where no face was detected is the attack failing at step one
        # — the best outcome — so it belongs in the denominator.
        blocked_b = [e for e in fail if "protected" in fail[e] and n_sub_exams.get(subject_of(e), 0) >= 2]
        total_b = n_b + len(blocked_b)

        lines.append(f"     exams of subjects with 2+ sessions   : {total_b}")
        lines.append(f"       no face detected, attack blocked   : {len(blocked_b)}")
        lines.append(f"       reached the comparison             : {n_b}")
        lines.append(f"     rank-1 hits   : {hits_b}")
        lines.append(f"       of {total_b} exams in the scenario  : {100 * hits_b / total_b:.1f} %")
        lines.append(f"       of {n_b} that reached comparison   : {100 * hits_b / n_b:.1f} %")
        lines.append("     The exam itself is removed from the gallery, so a hit requires")
        lines.append("     recognising the subject in a different acquisition. This is the")
        lines.append("     most demanding scenario and the closest to a photo-based attack.")


def main():
    global MODEL_NAME, DETECTOR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs_dir", required=True, help="pairs_2d folder from render_pairs_ddp.py")
    ap.add_argument("--out", default=None, help="Text report")
    ap.add_argument("--csv", default=None, help="Per-exam distances (CSV)")
    ap.add_argument("--cache", default=None, help=".npz file to cache the embeddings")
    ap.add_argument("--model", default=MODEL_NAME, help="ArcFace, Facenet512, VGG-Face, Dlib")
    ap.add_argument("--detector", default=DETECTOR, help="retinaface, mtcnn, opencv, ssd")
    ap.add_argument("--exclude_file", default=None,
                    help="File with `<subject or exam id>: <reason>` lines to exclude (data defects)")
    ap.add_argument("--attack_prefix", default=None,
                    help="Evaluate the attack only on exams whose id starts with this prefix. "
                         "Calibration always uses every render (it needs repeated sessions).")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N exams (dry run)")
    args = ap.parse_args()

    MODEL_NAME, DETECTOR = args.model, args.detector
    print(f"model={MODEL_NAME} | detector={DETECTOR}")

    orig, anon, fail = collect(args.pairs_dir, args.cache, args.limit)
    if not orig:
        raise SystemExit("No original render with a detected face — nothing to evaluate.")

    lines, rows = [], []
    lines.append(f"RE-IDENTIFICATION RISK — model {MODEL_NAME}, detector {DETECTOR}")
    excluded_subjects, excluded_exams = load_exclusions(args.exclude_file)
    orig, anon, fail = apply_exclusions(orig, anon, fail, excluded_subjects, excluded_exams, lines)
    warn_hubs(orig, anon, lines)
    cal = calibrate(orig, lines)
    thr, fp_rate = cal if cal else (None, None)

    # The restriction is applied AFTER calibration and BEFORE the attack on
    # purpose: calibration needs repeated sessions, which only some cohorts
    # have, while the attack is only interpretable where the original renders
    # resolve the facial surface.
    if args.attack_prefix:
        orig, anon, fail = restrict(orig, args.attack_prefix), restrict(anon, args.attack_prefix), \
            restrict(fail, args.attack_prefix)
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"RESTRICTION — the attack is evaluated only on exams starting with '{args.attack_prefix}'")
        lines.append("=" * 72)
        lines.append(f"  queries and gallery : {len(orig)} originals, {len(set(subject_of(e) for e in orig))} subjects")
        lines.append("  Section 1 calibration still uses ALL renders.")
        if not orig:
            raise SystemExit(f"No exam matches prefix {args.attack_prefix}.")

    verification(orig, anon, fail, thr, lines, rows, fp_rate)
    identification(orig, anon, fail, thr, lines)

    report = "\n".join(lines)
    print("\n" + report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nReport: {args.out}")
    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Per-exam distances: {args.csv}")


if __name__ == "__main__":
    main()
