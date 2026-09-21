# -*- coding: utf-8 -*-
"""
MFU (Model FLOP Utilisation) tracker for training and inference.

Timing uses CUDA events with a single synchronisation per epoch (training) or
at the end (inference), so there is no per-step overhead.

Methodology notes
=================

1. COUNTING CONVENTION: FLOPs, NOT MACs.
   A multiply-accumulate counts as 1 MAC but 2 FLOPs. The MFU formula assumes
   FLOPs. torch.utils.flop_counter.FlopCounterMode reports FLOPs (the x2 is
   included); verify_flop_convention() checks this against the closed-form
   FLOP count of a single conv3d rather than trusting it blindly.

2. THE x3 HEURISTIC (training).
   A training step does forward + backward, and backward costs roughly 2x the
   forward (one pass for activation gradients, one for weight gradients), so
   step ~= 3 x forward. This is the standard heuristic (PaLM, Kaplan et al.)
   and an approximation: it ignores the optimizer and fused ops. For
   transparency the forward-only (x1) MFU is reported as well.

3. DENOMINATOR = peak of the GPUs in the ALLOCATION, at the precision used —
   not the peak of the whole cluster. NVIDIA A100-40GB:
       FP32 (CUDA cores)        : 19.5 TFLOP/s
       TF32 (tensor cores)      : 156  TFLOP/s  <- PyTorch default with allow_tf32
       FP16/BF16 (tensor cores) : 312  TFLOP/s  <- under AMP
   Multiply by N for N GPUs. Under AMP FP16 the primary peak is 312xN and the
   secondary 156xN (part of the ops still run in TF32/FP32, so 312 is an
   optimistic bound and 156 a realistic one).

4. INTERPRETATION. 3D convolutions over large volumes are typically
   memory-bound, not compute-bound: an MFU of 5-25 % is normal for this
   workload. MFU_compute (fwd+bwd only) vs MFU_wall (whole step) measures the
   overhead of everything else (loader, all-reduce, optimizer). In inference,
   forward-only vs end-to-end tells whether the bottleneck is the GPU or the
   I/O / pre-processing pipeline.

5. MEASUREMENT. CUDA is asynchronous: without a synchronisation the clock only
   measures kernel enqueue time. torch.cuda.Event(enable_timing=True) +
   torch.cuda.synchronize() are used, and the first epochs/samples (cudnn
   autotuning, torch.compile, allocator warm-up) are excluded from the
   steady-state summary.

Usage — training:
    tracker = MFUTracker(forward_flops, gpu_peak_tflops=312.0, warmup_epochs=4)
    ev = tracker.new_step_events(); ev.fwd_start.record()
    ... forward ...;   ev.fwd_end.record()
    ... backward ...;  ev.bwd_end.record()
    ... optimizer ...; ev.step_end.record()
    record = tracker.finalize_epoch(epoch); tracker.log_epoch(record)
    tracker.log_summary(); tracker.save_csv(path)

Usage — inference:
    tracker = InferenceMFUTracker(forward_flops, gpu_peak_tflops=312.0)
    start, end, skip = tracker.new_sample_events()
    if not skip: start.record()
    ... forward ...
    if not skip: end.record()
    tracker.log_summary(); tracker.save_csv(path)
"""

import csv
import logging
import statistics
from dataclasses import dataclass
from typing import List

import torch


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------

def compute_model_flops(model: torch.nn.Module, dummy_input: torch.Tensor) -> int:
    """
    Count the FLOPs of one forward pass.

    Tries torch.utils.flop_counter.FlopCounterMode first (PyTorch 2.0+), then
    fvcore's FlopCountAnalysis. dummy_input must be on the same device as the
    model. Returns 0 if both fail (times are still recorded, MFU shows 0.0 %).
    """
    was_training = model.training
    model.eval()

    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(model, display=False) as fcm:
            with torch.no_grad():
                model(dummy_input.float())
        total = int(fcm.get_total_flops())
        if total > 0:
            if was_training:
                model.train()
            return total
    except Exception:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fa = FlopCountAnalysis(model, dummy_input.float())
            fa.unsupported_ops_warnings(False)
            total = int(fa.total())
        if total > 0:
            if was_training:
                model.train()
            return total
    except Exception:
        pass

    logging.warning("[MFU] compute_model_flops: FlopCounterMode and fvcore both failed. MFU=0.0%%.")
    if was_training:
        model.train()
    return 0


def verify_flop_convention(device=None, tol: float = 0.02) -> dict:
    """
    Cross-check the counting convention (FLOPs vs MACs).

    Builds one isolated Conv3d of known size, computes its FLOPs analytically,
        FLOPs = 2 x Cout x (Cin/groups) x K^3 x (D_out x H_out x W_out)
    (bias ignored: < 0.01 %) and compares with compute_model_flops().

    ratio = counter / analytic:
        ~1.0 -> the counter reports FLOPs (x2). Correct for MFU.
        ~0.5 -> the counter reports MACs. MFU would be under-estimated by 2x.
    Logs a WARNING when the ratio is not ~1.0.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Cin, Cout, K = 8, 16, 3
    D = H = W = 16
    stride, padding = 1, 1

    conv = torch.nn.Conv3d(Cin, Cout, kernel_size=K, stride=stride, padding=padding, bias=False).to(device)
    x = torch.randn(1, Cin, D, H, W, device=device)

    D_out = (D + 2 * padding - K) // stride + 1
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    flops_analytic = 2 * Cout * Cin * (K ** 3) * (D_out * H_out * W_out)
    flops_counter = compute_model_flops(conv, x)
    ratio = (flops_counter / flops_analytic) if flops_analytic > 0 else 0.0

    if abs(ratio - 1.0) <= tol:
        verdict, ok = "FLOPs (x2 multiply-add) — correct for MFU", True
    elif abs(ratio - 0.5) <= tol:
        verdict, ok = "MACs (no x2) — MFU would be under-estimated by 2x", False
    else:
        verdict, ok = f"unexpected ratio ({ratio:.3f}) — check the counter", False

    result = {"flops_analytic": flops_analytic, "flops_counter": flops_counter,
              "ratio": round(ratio, 4), "convention_ok": ok, "verdict": verdict}
    log = logging.info if ok else logging.warning
    log("[MFU] verify_flop_convention: analytic=%d counter=%d ratio=%.3f -> %s",
        flops_analytic, flops_counter, ratio, verdict)
    return result


# ---------------------------------------------------------------------------
# Training MFU
# ---------------------------------------------------------------------------

@dataclass
class StepEvents:
    fwd_start: torch.cuda.Event
    fwd_end: torch.cuda.Event
    bwd_end: torch.cuda.Event
    step_end: torch.cuda.Event


class MFUTracker:
    """
    Training MFU via CUDA events (one synchronisation per epoch).

    Args:
        forward_flops:    FLOPs of one forward pass (from compute_model_flops).
        gpu_peak_tflops:  PRIMARY GPU peak at the precision in use
                          (A100: 156 TF32/FP32, 312 FP16 AMP).
        warmup_epochs:    Epochs excluded from the steady-state summary.
        world_size:       Number of GPUs (for aggregate TFLOP/s).
        gpu_peak_tflops_secondary: Optional SECONDARY peak reported alongside
                          (e.g. 156 under AMP FP16, 19.5 for FP32 with TF32).
        samples_per_step: Batch size per GPU (for images/s).
    """

    def __init__(self, forward_flops: int, gpu_peak_tflops: float, warmup_epochs: int = 4,
                 world_size: int = 1, gpu_peak_tflops_secondary: float = None,
                 samples_per_step: int = 1):
        self.forward_flops = forward_flops
        self.gpu_peak_flops = gpu_peak_tflops * 1e12
        self.gpu_peak_flops_secondary = gpu_peak_tflops_secondary * 1e12 if gpu_peak_tflops_secondary else None
        self.warmup_epochs = warmup_epochs
        self.world_size = world_size
        self.samples_per_step = samples_per_step
        self._events: List[StepEvents] = []
        self.epoch_records = []

    def new_step_events(self) -> StepEvents:
        ev = StepEvents(
            fwd_start=torch.cuda.Event(enable_timing=True),
            fwd_end=torch.cuda.Event(enable_timing=True),
            bwd_end=torch.cuda.Event(enable_timing=True),
            step_end=torch.cuda.Event(enable_timing=True),
        )
        self._events.append(ev)
        return ev

    def finalize_epoch(self, epoch: int) -> dict:
        """Synchronise the events and compute the mean MFU of the epoch."""
        if not self._events:
            return {}

        torch.cuda.synchronize()

        t_fwd = t_bwd = t_step = 0.0
        n = len(self._events)
        for ev in self._events:
            t_fwd += ev.fwd_start.elapsed_time(ev.fwd_end)
            t_bwd += ev.fwd_end.elapsed_time(ev.bwd_end)
            t_step += ev.fwd_start.elapsed_time(ev.step_end)
        self._events.clear()

        avg_fwd = t_fwd / n / 1000.0
        avg_bwd = t_bwd / n / 1000.0
        avg_step = t_step / n / 1000.0

        flops3x = self.forward_flops * 3   # fwd + bwd ~= 3 x fwd
        flops1x = self.forward_flops       # forward only
        t_compute = avg_fwd + avg_bwd

        def _mfu(flops, t, peak):
            return (flops / (t * peak) * 100.0) if (t > 0 and peak) else 0.0

        mfu_c = _mfu(flops3x, t_compute, self.gpu_peak_flops)     # compute (fwd+bwd)
        mfu_w = _mfu(flops3x, avg_step, self.gpu_peak_flops)      # wall (whole step, x3)
        mfu_w_fwd = _mfu(flops1x, avg_step, self.gpu_peak_flops)  # wall, forward only (x1)

        tg = flops3x / avg_step / 1e12 if avg_step > 0 else 0.0
        ta = tg * self.world_size

        ips_gpu = (self.samples_per_step / avg_step) if avg_step > 0 else 0.0
        ips_agg = ips_gpu * self.world_size

        record = {
            "epoch": epoch,
            "t_fwd_ms": round(avg_fwd * 1000, 1),
            "t_bwd_ms": round(avg_bwd * 1000, 1),
            "t_step_ms": round(avg_step * 1000, 1),
            "imgs_per_s_gpu": round(ips_gpu, 3),
            "imgs_per_s_agg": round(ips_agg, 3),
            "mfu_compute": round(mfu_c, 2),
            "mfu_wall": round(mfu_w, 2),
            "mfu_wall_fwd": round(mfu_w_fwd, 2),
            "tflops_gpu": round(tg, 2),
            "tflops_agg": round(ta, 2),
        }
        if self.gpu_peak_flops_secondary:
            record["mfu_compute_sec"] = round(_mfu(flops3x, t_compute, self.gpu_peak_flops_secondary), 2)
            record["mfu_wall_sec"] = round(_mfu(flops3x, avg_step, self.gpu_peak_flops_secondary), 2)

        self.epoch_records.append(record)
        return record

    def log_epoch(self, record: dict) -> None:
        if not record:
            return
        tag = "[WARMUP]" if record["epoch"] < self.warmup_epochs else "[MFU]  "
        sec = f" | MFU_wall(sec)={record['mfu_wall_sec']:.1f}%" if "mfu_wall_sec" in record else ""
        logging.info(
            "%s Epoch %03d | t_fwd=%dms t_bwd=%dms t_step=%dms | "
            "MFU_compute=%.1f%% MFU_wall=%.1f%% (fwd-only=%.1f%%)%s | TFLOP/s_gpu=%.1f TFLOP/s_agg=%.1f",
            tag, record["epoch"], round(record["t_fwd_ms"]), round(record["t_bwd_ms"]),
            round(record["t_step_ms"]), record["mfu_compute"], record["mfu_wall"],
            record["mfu_wall_fwd"], sec, record["tflops_gpu"], record["tflops_agg"],
        )

    def log_summary(self) -> None:
        steady = [r for r in self.epoch_records if r["epoch"] >= self.warmup_epochs]
        if not steady:
            logging.info("[MFU] No steady-state epochs (warmup=%d epochs).", self.warmup_epochs)
            return

        def _s(key):
            v = [r[key] for r in steady]
            return sum(v) / len(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

        mc, mcs = _s("mfu_compute")
        mw, mws = _s("mfu_wall")
        mwf, mwfs = _s("mfu_wall_fwd")
        tg, tgs = _s("tflops_gpu")
        ta, tas = _s("tflops_agg")
        e0 = min(r["epoch"] for r in steady)
        e1 = max(r["epoch"] for r in steady)

        sep = "=" * 62
        logging.info(sep)
        logging.info(" MFU SUMMARY -- steady-state epochs [%d..%d]", e0, e1)
        logging.info(" Forward FLOPs : %.1f GFLOPs", self.forward_flops / 1e9)
        logging.info(" GPU peak      : %.0f TFLOP/s (primary)", self.gpu_peak_flops / 1e12)
        if self.gpu_peak_flops_secondary:
            logging.info(" GPU peak (sec): %.0f TFLOP/s (secondary)", self.gpu_peak_flops_secondary / 1e12)
        logging.info(" World size    : %d", self.world_size)
        logging.info(" Step convention: x3 (fwd+bwd). MFU_wall_fwd_only uses x1.")
        logging.info("-" * 62)
        logging.info("  MFU_compute (x3, fwd+bwd) : %.1f%% +/- %.1f%%", mc, mcs)
        logging.info("  MFU_wall    (x3, step)    : %.1f%% +/- %.1f%%", mw, mws)
        logging.info("  MFU_wall_fwd_only (x1)    : %.1f%% +/- %.1f%%", mwf, mwfs)
        if self.gpu_peak_flops_secondary:
            mcx, mcxs = _s("mfu_compute_sec")
            mwx, mwxs = _s("mfu_wall_sec")
            logging.info("  -- vs secondary peak (%.0f TFLOP/s) --", self.gpu_peak_flops_secondary / 1e12)
            logging.info("  MFU_compute (sec)         : %.1f%% +/- %.1f%%", mcx, mcxs)
            logging.info("  MFU_wall    (sec)         : %.1f%% +/- %.1f%%", mwx, mwxs)
        logging.info("-" * 62)
        logging.info("  TFLOP/s per GPU: %.1f +/- %.1f", tg, tgs)
        if self.world_size > 1:
            logging.info("  TFLOP/s agg    : %.1f +/- %.1f  [x%d GPUs]", ta, tas, self.world_size)
        ig, _ = _s("imgs_per_s_gpu")
        ia, _ = _s("imgs_per_s_agg")
        logging.info("  Throughput     : %.2f img/s per GPU | %.2f img/s aggregate", ig, ia)
        logging.info(sep)

    def save_csv(self, path: str) -> None:
        if not self.epoch_records:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.epoch_records[0].keys()))
            writer.writeheader()
            writer.writerows(self.epoch_records)
        logging.info("[MFU] CSV saved: %s", path)


# ---------------------------------------------------------------------------
# Inference MFU
# ---------------------------------------------------------------------------

class InferenceMFUTracker:
    """
    Lightweight inference MFU tracker (forward only). One CUDA event pair per
    sample, one synchronisation at the end.

    Reports two complementary numbers:
      - forward-only : time of model(x) alone, from CUDA events — efficiency of
                       the isolated kernel;
      - end-to-end   : real loop throughput (I/O + pre-processing included),
                       via record_wallclock() — efficiency of the PIPELINE.
    The gap between them says whether the bottleneck is the GPU or the I/O.
    """

    def __init__(self, forward_flops: int, gpu_peak_tflops: float, warmup_samples: int = 5,
                 gpu_peak_tflops_secondary: float = None, world_size: int = 1):
        self.forward_flops = forward_flops
        self.gpu_peak_flops = gpu_peak_tflops * 1e12
        self.gpu_peak_flops_secondary = gpu_peak_tflops_secondary * 1e12 if gpu_peak_tflops_secondary else None
        self.warmup_samples = warmup_samples
        self.world_size = world_size
        self._events = []
        self._counter = 0
        self._e2e_samples = 0
        self._e2e_seconds = 0.0

    def new_sample_events(self):
        """Return (start, end, skip). skip=True during warm-up: not recorded."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        skip = self._counter < self.warmup_samples
        self._counter += 1
        if not skip:
            self._events.append((start, end))
        return start, end, skip

    def record_wallclock(self, n_samples: int, total_seconds: float) -> None:
        """
        Record the END-TO-END throughput of the inference loop: samples
        processed and total wall-clock time (I/O and pre-processing included).
        Under DDP pass the aggregate over all ranks; world_size scales the peak.
        """
        self._e2e_samples = n_samples
        self._e2e_seconds = total_seconds

    def summarize(self) -> dict:
        if not self._events:
            return {}
        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in self._events]
        avg_ms = sum(times_ms) / len(times_ms)
        avg_s = avg_ms / 1000.0

        def _mfu(t, peak):
            return (self.forward_flops / (t * peak) * 100.0) if (t > 0 and peak) else 0.0

        out = {
            "n_samples": len(times_ms),
            "avg_fwd_ms": round(avg_ms, 1),
            "mfu_fwd_pure_pct": round(_mfu(avg_s, self.gpu_peak_flops), 2),
            "tflops_gpu": round(self.forward_flops / avg_s / 1e12 if avg_s > 0 else 0.0, 2),
        }
        if self.gpu_peak_flops_secondary:
            out["mfu_fwd_pure_sec_pct"] = round(_mfu(avg_s, self.gpu_peak_flops_secondary), 2)

        if self._e2e_samples > 0 and self._e2e_seconds > 0:
            s_per_sample = self._e2e_seconds / self._e2e_samples
            peak_e2e = self.gpu_peak_flops * self.world_size   # the loop runs on world_size GPUs
            out["e2e_samples"] = self._e2e_samples
            out["e2e_s_per_sample"] = round(s_per_sample, 4)
            out["mfu_e2e_pct"] = round(self.forward_flops * self._e2e_samples
                                       / (self._e2e_seconds * peak_e2e) * 100.0, 2)
            out["e2e_samples_per_s"] = round(self._e2e_samples / self._e2e_seconds, 3)
            if self.gpu_peak_flops_secondary:
                peak_e2e_sec = self.gpu_peak_flops_secondary * self.world_size
                out["mfu_e2e_sec_pct"] = round(self.forward_flops * self._e2e_samples
                                               / (self._e2e_seconds * peak_e2e_sec) * 100.0, 2)
            out["compute_fraction"] = round(avg_s / s_per_sample, 3) if s_per_sample > 0 else 0.0
        return out

    def log_summary(self, label: str = "INFERENCE") -> None:
        r = self.summarize()
        if not r:
            logging.info("[MFU-%s] No samples to summarise.", label)
            return
        sep = "=" * 62
        logging.info(sep)
        logging.info(" MFU SUMMARY -- %s (warmup=%d samples)", label, self.warmup_samples)
        logging.info(" Forward FLOPs : %.1f GFLOPs", self.forward_flops / 1e9)
        logging.info(" GPU peak      : %.0f TFLOP/s (primary)", self.gpu_peak_flops / 1e12)
        if self.gpu_peak_flops_secondary:
            logging.info(" GPU peak (sec): %.0f TFLOP/s (secondary)", self.gpu_peak_flops_secondary / 1e12)
        logging.info("-" * 62)
        logging.info("  Samples measured   : %d", r["n_samples"])
        logging.info("  Mean forward time  : %.1f ms/sample", r["avg_fwd_ms"])
        logging.info("  MFU forward-only   : %.1f%%", r["mfu_fwd_pure_pct"])
        if "mfu_fwd_pure_sec_pct" in r:
            logging.info("  MFU forward (sec)  : %.1f%%", r["mfu_fwd_pure_sec_pct"])
        logging.info("  TFLOP/s per GPU    : %.1f", r["tflops_gpu"])
        if "mfu_e2e_pct" in r:
            logging.info("-" * 62)
            logging.info("  -- END-TO-END (I/O + pre-processing included) --")
            logging.info("  Samples (e2e)      : %d  [world_size=%d]", r["e2e_samples"], self.world_size)
            logging.info("  Time/sample e2e    : %.1f ms", r["e2e_s_per_sample"] * 1000)
            logging.info("  Global throughput  : %.3f exams/s", r["e2e_samples_per_s"])
            logging.info("  MFU end-to-end     : %.2f%%", r["mfu_e2e_pct"])
            if "mfu_e2e_sec_pct" in r:
                logging.info("  MFU end-to-end(sec): %.2f%%", r["mfu_e2e_sec_pct"])
            logging.info("  Compute fraction   : %.1f%%  (forward / end-to-end)", r["compute_fraction"] * 100)
            logging.info("  -> Dominant bottleneck: %s",
                         "I/O / pre-processing" if r["compute_fraction"] < 0.5 else "compute (GPU)")
        logging.info(sep)

    def save_csv(self, path: str) -> None:
        r = self.summarize()
        if not r:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(r.keys()))
            writer.writeheader()
            writer.writerow(r)
        logging.info("[MFU] Inference CSV saved: %s", path)
