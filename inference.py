# -*- coding: utf-8 -*-
"""
Distributed inference, evaluation and anonymisation at the exam's native resolution.

For every exam listed in the test CSV(s):
  1. reorient to RAS, resize to the network input size, min-max normalise;
  2. forward pass (FP16 autocast, torch.compile);
  3. Dice at network resolution and, after trilinear restore + argmax, Dice at
     the original resolution (only when a ground-truth mask is available);
  4. AnonymizationOfficer applies the per-region transforms to the original
     volume; the predicted mask and the anonymised volume are written as NIfTI.

Launch with torchrun:

    torchrun --nproc_per_node=2 inference.py \
        --config configs/train_128.yaml --weights <run>/best_*.pt \
        --test_csv data/test_a.csv data/test_b.csv \
        --out_dir <run>/inference --metrics_csv <run>/test_metrics.csv \
        --metrics_txt <run>/test_metrics.txt [--preserve_regions eyes]

The CSV must have an image_path column; mask_path is optional (no Dice without it).
"""
import argparse
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from anonymization.officer import ALL_REGIONS, AnonymizationOfficer, resolve_regions
from data.dataset import DEFAULT_INPUT_SIZE, remap_labels, resample
from models import build_network
from profiling.mfu import InferenceMFUTracker, compute_model_flops, verify_flop_convention
from utils.validation import macro_dice_iou

GPU_PEAK_TFLOPS = 312.0
GPU_PEAK_TFLOPS_SECONDARY = 156.0

log = logging.getLogger("inference")


def calculate_hard_metrics(gt_data: np.ndarray, pred_data: np.ndarray, num_classes: int) -> float:
    """
    Macro Dice at the ORIGINAL resolution, over label maps.

    Computed per facial class (1..num_classes-1, background excluded) and then
    averaged — the same convention as the network-resolution metric. A class
    absent from both GT and prediction is skipped; a class predicted but absent
    from the GT counts as 0.
    """
    dices = []
    for c in range(1, num_classes):
        gt_bin, pred_bin = (gt_data == c), (pred_data == c)
        gt_sum, pred_sum = gt_bin.sum(), pred_bin.sum()
        if gt_sum == 0 and pred_sum == 0:
            continue
        intersection = np.logical_and(gt_bin, pred_bin).sum()
        dices.append((2.0 * intersection) / (gt_sum + pred_sum + 1e-8))
    return float(np.mean(dices)) if dices else 0.0


def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")
    return local_rank, global_rank, world_size


class PhysicalEvalDataset(Dataset):
    """
    Pre-processing (I/O + RAS + resize + normalise) on DataLoader workers, so it
    overlaps with the GPU forward pass of the previous exam.
    """

    def __init__(self, df_chunk: pd.DataFrame, target_size):
        self.df = df_chunk.reset_index(drop=True)
        self.target_size = tuple(target_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        exam_id = os.path.basename(os.path.dirname(row["image_path"]))

        img_ras = resample(nib.load(row["image_path"]))

        # The mask is optional: without it only anonymisation and timing run.
        mask_path = row["mask_path"] if "mask_path" in row.index else None
        has_gt = bool(mask_path) and isinstance(mask_path, str) and os.path.exists(mask_path)
        gt_ras = resample(nib.load(mask_path)) if has_gt else None

        orig_data_ras = img_ras.get_fdata().copy()
        orig_shape_ras = orig_data_ras.shape
        orig_affine_ras = img_ras.affine.copy()

        img_raw = torch.tensor(orig_data_ras, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if has_gt:
            gt_raw = torch.tensor(gt_ras.get_fdata().copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            gt_raw = remap_labels(gt_raw)
        else:
            gt_raw = torch.zeros_like(img_raw)

        orig_gt_ras = gt_raw.squeeze(0).squeeze(0).numpy().astype(np.uint8)

        img_t = F.interpolate(img_raw, size=self.target_size, mode="trilinear", align_corners=False).squeeze(0)
        gt_t = F.interpolate(gt_raw, size=self.target_size, mode="nearest").squeeze(0)

        t_min, t_max = img_t.min(), img_t.max()
        if t_max > 0:
            img_t = (img_t - t_min) / (t_max - t_min)

        return {
            "img": img_t,                    # (1, *target_size), normalised
            "gt": gt_t,                      # (1, *target_size)
            "orig_data": orig_data_ras,      # numpy (W,H,D) — for anonymisation
            "orig_gt": orig_gt_ras,          # numpy (W,H,D) — original-resolution label map
            "orig_shape": orig_shape_ras,
            "orig_affine": orig_affine_ras,
            "exam_id": exam_id,
            "has_gt": int(has_gt),
        }


def _collate(batch):
    """Collate for batch_size=1 with variable-size numpy arrays."""
    item = batch[0]
    out = dict(item)
    out["img"] = item["img"].unsqueeze(0)
    out["gt"] = item["gt"].unsqueeze(0)
    return out


def _evaluate_dataset(label, df, model, device, global_rank, world_size, out_dir, out_channels,
                      target_size, forward_flops, profile, no_amp, timed=True, write_outputs=True,
                      anon_regions=None, preserve_margin=0):
    """
    Run inference over a set of exams. Returns this rank's per-exam results,
    the wall-clock time, the MFU tracker (rank 0) and the phase profile.

    timed=True  -> barriers on both ends so the time covers the whole DDP set,
                   and MFU is recorded.
    timed=False -> quality only; no barriers, no MFU.
    write_outputs=False suppresses writing the mask and anonymised volume.
    """
    my_chunk = np.array_split(df, world_size)[global_rank]

    dataloader = DataLoader(
        PhysicalEvalDataset(my_chunk, target_size),
        batch_size=1, shuffle=False, num_workers=4, prefetch_factor=2,
        pin_memory=True, collate_fn=_collate,
    )

    mfu_inf = None
    if timed and global_rank == 0 and forward_flops:
        n_mine = len(my_chunk)
        warmup = 2 if n_mine >= 8 else (1 if n_mine >= 3 else 0)
        mfu_inf = InferenceMFUTracker(
            forward_flops=forward_flops, gpu_peak_tflops=GPU_PEAK_TFLOPS, warmup_samples=warmup,
            gpu_peak_tflops_secondary=GPU_PEAK_TFLOPS_SECONDARY, world_size=world_size,
        )
        log.info(f"[MFU] {label}: {n_mine} exams on rank 0, warmup={warmup} "
                 f"-> {max(0, n_mine - warmup)} measured samples")

    exam_results = []
    phase_times = defaultdict(list)   # populated only when profile=True
    executor = ThreadPoolExecutor(max_workers=2)
    loop = tqdm(dataloader, disable=(global_rank != 0), desc=f"{label} (rank 0)")

    # CUDA is asynchronous: the phase timer needs explicit syncs (rank 0, --profile only).
    _sync = (lambda: torch.cuda.synchronize()) if (profile and global_rank == 0) else (lambda: None)
    _t = time.perf_counter

    if timed:
        dist.barrier()
    t_start = time.time()

    with torch.no_grad():
        for batch in loop:
            exam_start = time.time()

            if profile: _sync(); t0 = _t()
            img_tensor = batch["img"].to(device, non_blocking=True)
            gt_tensor = batch["gt"].to(device, non_blocking=True)
            orig_data_ras = batch["orig_data"]
            orig_gt_ras = batch["orig_gt"]
            orig_shape_ras = batch["orig_shape"]
            orig_affine_ras = batch["orig_affine"]
            exam_id = batch["exam_id"]
            has_gt = bool(int(batch["has_gt"]))
            if profile: _sync(); phase_times["1_H2D_ms"].append((_t() - t0) * 1e3)

            # Forward
            if profile: _sync(); t0 = _t()
            _mfu_start = _mfu_end = None
            _mfu_skip = True
            if mfu_inf is not None:
                _mfu_start, _mfu_end, _mfu_skip = mfu_inf.new_sample_events()
                if not _mfu_skip:
                    _mfu_start.record()
            if no_amp:
                logits = model(img_tensor.float())
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(img_tensor.float())
            if mfu_inf is not None and not _mfu_skip:
                _mfu_end.record()
            predictions = logits.float()
            if profile: _sync(); phase_times["2_forward_ms"].append((_t() - t0) * 1e3)

            # Network-resolution metric (macro over facial classes, background ignored)
            if profile: _sync(); t0 = _t()
            dsc = macro_dice_iou(predictions, gt_tensor.squeeze(1), out_channels)[0] if has_gt else None
            if profile: _sync(); phase_times["3_metrics_ms"].append((_t() - t0) * 1e3)

            # Restore to original resolution on the GPU
            if profile: _sync(); t0 = _t()
            probs_restored = F.interpolate(predictions, size=orig_shape_ras, mode="trilinear", align_corners=False)
            if profile: _sync(); phase_times["4_interp_post_GPU_ms"].append((_t() - t0) * 1e3)

            if profile: _sync(); t0 = _t()
            pred_mask_orig = torch.argmax(probs_restored, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            if profile: phase_times["5_D2H_argmax_ms"].append((_t() - t0) * 1e3)

            # Original-resolution metric, over the mask that is actually applied
            if profile: t0 = _t()
            dsc_orig = calculate_hard_metrics(orig_gt_ras, pred_mask_orig, out_channels) if has_gt else float("nan")
            if profile: phase_times["6_metrics_orig_res_ms"].append((_t() - t0) * 1e3)

            if profile: t0 = _t()
            officer = AnonymizationOfficer(orig_data_ras, pred_mask_orig, regions=anon_regions,
                                           preserve_margin=preserve_margin)
            defaced = officer.anonymize()
            if profile: phase_times["7_anonymisation_ms"].append((_t() - t0) * 1e3)

            exam_results.append({
                "Exam_ID": exam_id,
                "Dice_Score": round(dsc.item(), 4) if has_gt else float("nan"),
                "Dice_Score_orig_res": round(dsc_orig, 4) if has_gt else float("nan"),
                "Inference_Time_s": round(time.time() - exam_start, 3),
            })

            if write_outputs:
                _pred = pred_mask_orig.copy()
                _anon = defaced.astype(orig_data_ras.dtype).copy()
                _affine = orig_affine_ras.copy()
                executor.submit(nib.save, nib.Nifti1Image(_pred, _affine),
                                os.path.join(out_dir, f"{exam_id}_pred_mask.nii.gz"))
                executor.submit(nib.save, nib.Nifti1Image(_anon, _affine),
                                os.path.join(out_dir, f"{exam_id}_anon.nii.gz"))

    executor.shutdown(wait=True)
    if timed:
        dist.barrier()
    total_time = time.time() - t_start

    return exam_results, total_time, mfu_inf, phase_times


def _report_dataset(label, gathered_quality, n_timed, total_time, mfu_inf, phase_times,
                    csv_path, txt_path, profile, target_size, anon_regions=None):
    """Write the per-exam CSV and the text summary of one test set (rank 0 only)."""
    all_results = [item for sublist in gathered_quality for item in sublist]
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(csv_path, index=False)

    n_ex = len(df_out)
    n_gt = int(df_out["Dice_Score_orig_res"].notna().sum()) if n_ex else 0
    # Per-exam time = wall clock / timed exams; includes I/O and pre-processing,
    # which the in-loop Inference_Time_s under-estimates because of prefetching.
    avg_per_exam = total_time / n_timed if n_timed > 0 else 0.0
    throughput = n_timed / total_time if total_time > 0 else 0.0

    regions = tuple(anon_regions) if anon_regions is not None else ALL_REGIONS
    preserved = tuple(r for r in ALL_REGIONS if r not in regions)

    summary = (
        f"RESULTS {label} — input size {tuple(target_size)}\n"
        f"{'=' * 63}\n"
        f"Anonymised regions:                    {', '.join(regions) if regions else 'none'}\n"
        f"Preserved regions:                     {', '.join(preserved) if preserved else 'none'}\n"
        f"{'=' * 63}\n"
        + (
            f"QUALITY — over {n_gt} exams\n"
            f"{'-' * 63}\n"
            f"Mean DSC (network resolution):         {df_out['Dice_Score'].mean():.4f}\n"
            f"Mean DSC (original resolution):        {df_out['Dice_Score_orig_res'].mean():.4f}\n"
            if n_gt else
            f"QUALITY — not evaluated\n"
            f"{'-' * 63}\n"
            f"No ground truth in this set: no Dice computed.\n"
        )
        + f"{'=' * 63}\n"
        f"EFFICIENCY — over {n_timed} timed exams\n"
        f"{'-' * 63}\n"
        f"Total inference time:                  {total_time:.2f} s\n"
        f"Mean time per exam (incl. I/O):        {avg_per_exam:.4f} s   [= total / N]\n"
        f"Throughput:                            {throughput:.2f} exams/s\n"
    )

    if mfu_inf is not None:
        mfu_inf.record_wallclock(n_samples=n_timed, total_seconds=total_time)
        mfu = mfu_inf.summarize()
        if mfu:
            summary += (
                f"{'-' * 63}\n"
                f"Samples measured for MFU:              {mfu.get('n_samples', 0)}\n"
                f"Mean forward time:                     {mfu.get('avg_fwd_ms', 0):.1f} ms\n"
                f"MFU forward-only (peak 312 TFLOP/s):   {mfu.get('mfu_fwd_pure_pct', 0):.2f} %\n"
                f"MFU forward-only (peak 156 TFLOP/s):   {mfu.get('mfu_fwd_pure_sec_pct', 0):.2f} %\n"
                f"MFU end-to-end    (peak 312 TFLOP/s):  {mfu.get('mfu_e2e_pct', 0):.2f} %\n"
                f"MFU end-to-end    (peak 156 TFLOP/s):  {mfu.get('mfu_e2e_sec_pct', 0):.2f} %\n"
                f"Compute fraction (fwd / end-to-end):   {mfu.get('compute_fraction', 0) * 100:.1f} %\n"
            )
        mfu_inf.log_summary(label=f"{label} — DDP FP16")
        mfu_inf.save_csv(os.path.join(os.path.dirname(csv_path), f"mfu_inference_{label.lower()}.csv"))

    summary += f"{'=' * 63}\n"
    with open(txt_path, "w") as f:
        f.write(summary)
    print("\n" + summary)

    if profile and phase_times:
        print("=" * 63)
        print(f"PHASE PROFILE ({label}) — means excluding the first exam (CUDA warm-up)")
        print("  Note: I/O + resize run on the workers (not measured here)")
        print("=" * 63)
        total_steady = 0.0
        for phase, times in sorted(phase_times.items()):
            steady = times[1:] if len(times) > 1 else times
            avg = sum(steady) / len(steady) if steady else 0.0
            print(f"  {phase:<48s}: {avg:6.1f} ms")
            total_steady += avg
        print(f"  {'─' * 56}")
        print(f"  {'TOTAL measured phases (I/O covered by prefetch)':<48s}: {total_steady:6.1f} ms")
        print("=" * 63)

    return df_out, avg_per_exam, throughput


def run_physical_evaluation(weights_path, out_dir, metrics_csv, metrics_txt, out_channels, target_size,
                            model_name="mednext", profile=False, no_amp=False, test_csvs=(),
                            n_exams=None, timed_exams=None, anonymise="all", anon_regions=None,
                            preserve_margin=0):
    local_rank, global_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")

    regions = tuple(anon_regions) if anon_regions is not None else ALL_REGIONS
    preserved = tuple(r for r in ALL_REGIONS if r not in regions)

    if global_rank == 0:
        log.info("Starting distributed inference")
        log.info(f"Input size (must match training): {tuple(target_size)}")
        log.info(f"[ANONYMISATION] regions anonymised: {', '.join(regions) if regions else 'NONE'}")
        if preserved:
            log.info(f"[ANONYMISATION] regions PRESERVED: {', '.join(preserved)}")
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()

    model = build_network(model_name, out_channels)
    state_dict = torch.load(weights_path, map_location=device)
    # torch.compile prefixes every key with "_orig_mod."; strip it to match the bare model.
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    model = torch.compile(model)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Warm-up: compile (or load the cached graph) before the timer starts.
    dist.barrier()
    _t0 = time.time()
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, *target_size, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(_dummy.float())
        torch.cuda.synchronize()
    if global_rank == 0:
        log.info(f"[torch.compile] warm-up done in {time.time() - _t0:.1f}s")

    forward_flops = 0
    if global_rank == 0:
        try:
            raw = build_network(model_name, out_channels).to(device)
            _dummy = torch.zeros(1, 1, *target_size, device=device)
            forward_flops = compute_model_flops(raw, _dummy.float())
            del raw, _dummy
            torch.cuda.empty_cache()
            verify_flop_convention(device=device)
            log.info(f"[MFU] Forward FLOPs={forward_flops / 1e9:.1f} GFLOPs | "
                     f"GPU peak={GPU_PEAK_TFLOPS:.0f} TFLOP/s (FP16) | world_size={world_size}")
        except Exception as exc:
            log.warning(f"[MFU] not initialised: {exc}")
            forward_flops = 0

    datasets = []
    for csv_path in test_csvs:
        name = os.path.splitext(os.path.basename(csv_path))[0].upper()
        suffix = f"_{name.lower()}" if len(test_csvs) > 1 else ""
        datasets.append((name, pd.read_csv(csv_path),
                         metrics_csv.replace(".csv", f"{suffix}.csv"),
                         metrics_txt.replace(".txt", f"{suffix}.txt")))

    # --n_exams is a hard limit: exams beyond it are never read (dry runs).
    if n_exams is not None and n_exams > 0:
        datasets = [(name, df.head(n_exams), c, t) for name, df, c, t in datasets]

    if global_rank == 0:
        log.info("Exams to process: " + " + ".join(f"{len(df)} ({name})" for name, df, _, _ in datasets)
                 + f" = {sum(len(df) for _, df, _, _ in datasets)}")
        if timed_exams:
            log.info(f"[TIMING] efficiency measured on the first {timed_exams} exams of each set; "
                     f"quality on all processed exams. Anonymised outputs written for: {anonymise}.")

    overall = []
    for label, df, csv_path, txt_path in datasets:
        # Efficiency is measured on a subset, quality on everything. Each exam is
        # processed once: the first part is timed, the second is not. The split
        # depends only on the CSV, so every rank sees the same collectives.
        if timed_exams and 0 < timed_exams < len(df):
            df_timed, df_rest = df.head(timed_exams), df.iloc[timed_exams:]
        else:
            df_timed, df_rest = df, df.iloc[0:0]

        results, total_time, mfu_inf, phase_times = _evaluate_dataset(
            label, df_timed, model, device, global_rank, world_size, out_dir, out_channels,
            target_size, forward_flops, profile, no_amp, timed=True, write_outputs=True,
            anon_regions=anon_regions, preserve_margin=preserve_margin,
        )
        if len(df_rest) > 0:
            rest_results, _, _, _ = _evaluate_dataset(
                f"{label} (rest)", df_rest, model, device, global_rank, world_size, out_dir,
                out_channels, target_size, forward_flops, profile, no_amp, timed=False,
                write_outputs=(anonymise == "all"), anon_regions=anon_regions,
                preserve_margin=preserve_margin,
            )
            results = results + rest_results

        gathered = [None for _ in range(world_size)]
        dist.gather_object(results, gathered if global_rank == 0 else None, dst=0)

        if global_rank == 0:
            df_out, per_exam, thr = _report_dataset(
                label, gathered, len(df_timed), total_time, mfu_inf, phase_times,
                csv_path, txt_path, profile, target_size, anon_regions=anon_regions,
            )
            overall.append((label, len(df_out), len(df_timed), total_time, per_exam, thr))
        dist.barrier()

    if global_rank == 0 and overall:
        n_qual = sum(n for _, n, _, _, _, _ in overall)
        n_timed = sum(k for _, _, k, _, _, _ in overall)
        t_total = sum(t for _, _, _, t, _, _ in overall)
        lines = [
            "=" * 76,
            "INFERENCE SUMMARY",
            "  Quality over the full sets; efficiency over the timed exams.",
            "=" * 76,
            f"  {'Set':<10s} {'Eval.':>7s} {'Timed':>8s} {'Total (s)':>11s} {'Per exam (s)':>15s} {'Exams/s':>10s}",
        ]
        for label, n, k, t, per_exam, thr in overall:
            lines.append(f"  {label:<10s} {n:>7d} {k:>8d} {t:>11.2f} {per_exam:>15.4f} {thr:>10.2f}")
        lines.append(f"  {'-' * 70}")
        lines.append(f"  {'TOTAL':<10s} {n_qual:>7d} {n_timed:>8d} {t_total:>11.2f} "
                     f"{t_total / n_timed if n_timed else 0:>15.4f} {n_timed / t_total if t_total else 0:>10.2f}")
        lines.append("=" * 76)
        report = "\n".join(lines)
        print("\n" + report)
        with open(metrics_txt.replace(".txt", "_summary.txt"), "w") as f:
            f.write(report + "\n")

    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="YAML config used for training")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--test_csv", type=str, nargs="+", required=True,
                        help="One or more CSVs with image_path (and optionally mask_path)")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--metrics_csv", type=str, required=True)
    parser.add_argument("--metrics_txt", type=str, required=True)
    parser.add_argument("--profile", action="store_true",
                        help="Time each phase with CUDA syncs on rank 0 (perturbs throughput)")
    parser.add_argument("--no-amp", action="store_true", dest="no_amp", help="Run the forward pass in FP32")
    parser.add_argument("--n_exams", type=int, default=None,
                        help="Hard limit: only the first N exams of each set are processed")
    parser.add_argument("--timed_exams", type=int, default=None,
                        help="Time only the first N exams of each set; the rest still count for quality")
    parser.add_argument("--preserve_margin", type=int, default=0,
                        help="Extra voxels protected around preserved regions")
    parser.add_argument("--anonymise_regions", type=str, default=None,
                        help="Comma-separated regions to anonymise: nose, eyes, ears, mouth ('all' = default)")
    parser.add_argument("--preserve_regions", type=str, default=None,
                        help="Comma-separated regions to PRESERVE; everything else is anonymised")
    parser.add_argument("--anonymise", type=str, default="all", choices=["all", "timed"],
                        help="Which exams get their mask/anonymised volume written")
    args = parser.parse_args()

    if int(os.environ.get("RANK", "0")) == 0:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)
    out_channels = config["out_channels"]
    model_name = config.get("model", "mednext")
    target_size = tuple(config.get("input_size", DEFAULT_INPUT_SIZE))

    # Resolved before any work so a typo in a region name fails fast instead of
    # silently producing less-anonymised exams. Identical on every rank.
    anon_regions = resolve_regions(anonymise=args.anonymise_regions, preserve=args.preserve_regions)

    run_physical_evaluation(
        args.weights, args.out_dir, args.metrics_csv, args.metrics_txt,
        out_channels=out_channels, target_size=target_size, model_name=model_name,
        profile=args.profile, no_amp=args.no_amp, test_csvs=args.test_csv,
        n_exams=args.n_exams, timed_exams=args.timed_exams, anonymise=args.anonymise,
        anon_regions=anon_regions, preserve_margin=args.preserve_margin,
    )
