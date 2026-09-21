# -*- coding: utf-8 -*-
"""
Summarise the per-exam test metrics written by inference.py, with an optional,
documented list of excluded exams.

The measurement itself is never altered: the CSV keeps every exam evaluated,
and exclusions (e.g. exams with a known acquisition defect) are applied and
justified here, in the open.

Usage:
    python tools/summarise_test_results.py --csv <run>/test_metrics_a.csv <run>/test_metrics_b.csv \
        [--exclude_file exclusions.txt]

The exclusion file has one `<exam_id>: <reason>` per line (`#` comments allowed).
"""

import argparse
import os
import statistics as st
import csv


def load_exclusions(path):
    excluded = {}
    if not path:
        return excluded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, reason = line.partition(":")
            excluded[key.strip()] = reason.strip() or "excluded"
    return excluded


def summarise(csv_path: str, excluded: dict) -> dict:
    label = os.path.splitext(os.path.basename(csv_path))[0]
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = [r for r in rows if r["Exam_ID"] not in excluded]
    dropped = [r for r in rows if r["Exam_ID"] in excluded]

    def col(rs, name):
        return [float(r[name]) for r in rs if r.get(name) not in (None, "", "nan")]

    d_net = col(kept, "Dice_Score")
    d_orig = col(kept, "Dice_Score_orig_res") or d_net

    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    print(f"  Exams evaluated  : {len(rows)}")
    if dropped:
        print(f"  Exams excluded   : {len(dropped)}")
        for r in dropped:
            print(f"      {r['Exam_ID']}  (measured Dice {r['Dice_Score']})")
            print(f"        reason: {excluded[r['Exam_ID']]}")
    print(f"  Exams reported   : {len(kept)}")
    if not kept or not d_net:
        return {}

    for name, vals in (("network resolution", d_net), ("original resolution", d_orig)):
        print(f"  Dice ({name:<19s}): mean {st.mean(vals):.4f} | median {st.median(vals):.4f} | "
              f"std {st.stdev(vals) if len(vals) > 1 else 0:.4f} | min {min(vals):.4f} | max {max(vals):.4f}")

    return {"label": label, "n_total": len(rows), "n_kept": len(kept), "n_dropped": len(dropped),
            "dice_net": st.mean(d_net), "dice_orig": st.mean(d_orig)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="One or more test_metrics CSVs")
    ap.add_argument("--exclude_file", default=None, help="File with `<exam_id>: <reason>` lines")
    args = ap.parse_args()

    excluded = load_exclusions(args.exclude_file)
    results = [r for r in (summarise(p, excluded) for p in args.csv) if r]

    if results:
        print(f"\n{'=' * 66}\nSUMMARY\n{'=' * 66}")
        print(f"  {'Set':<28s} {'Exams':>9s} {'Dice':>8s}")
        for r in results:
            n = f"{r['n_kept']}" + (f" (-{r['n_dropped']})" if r["n_dropped"] else "")
            print(f"  {r['label']:<28s} {n:>9s} {r['dice_orig']:>8.4f}")
        print("=" * 66)


if __name__ == "__main__":
    main()
