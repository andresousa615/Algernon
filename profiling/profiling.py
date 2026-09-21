# -*- coding: utf-8 -*-
"""
profiling/profiling.py — DDP training monitor
=============================================
Measures four things that matter when diagnosing distributed training:
  1. VRAM        — peak memory per phase (forward, backward, optimizer step)
  2. Communication — all-reduce overhead via a DDP comm hook
  3. Starvation  — time the GPU spends waiting for the data loader
  4. Timeline    — per-phase breakdown of the training loop

Usage:
  from profiling.profiling import TrainingProfiler
  profiler = TrainingProfiler(save_dir, global_rank, enabled=True)

  with profiler.profile_epoch(epoch):
      for batch in profiler.iter_dataloader(loader):
          with profiler.phase("forward"):   ...
          with profiler.phase("backward"):  ...
          with profiler.phase("optimizer"): ...

  profiler.save_final_report()

With enabled=False every method is a no-op, so the training script needs no
conditionals.
"""

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.distributed as dist


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseStats:
    """Accumulates the metrics of one phase over a whole epoch."""
    name: str
    total_time_ms: float = 0.0
    call_count: int = 0
    peak_vram_mb: float = 0.0    # peak during this phase (MB)
    vram_delta_mb: float = 0.0   # net allocation introduced by this phase (MB)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.call_count, 1)

    def to_dict(self) -> dict:
        return {
            "phase": self.name,
            "total_ms": round(self.total_time_ms, 2),
            "avg_ms": round(self.avg_time_ms, 2),
            "calls": self.call_count,
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "vram_delta_mb": round(self.vram_delta_mb, 1),
        }


@dataclass
class EpochReport:
    epoch: int
    phases: dict = field(default_factory=dict)   # {name: PhaseStats}
    loader_wait_ms: float = 0.0   # total time waiting for data (starvation)
    total_epoch_ms: float = 0.0
    nccl_ms: float = 0.0          # estimated time spent in gradient all-reduces
    batches: int = 0

    @property
    def gpu_utilization_pct(self) -> float:
        """Fraction of the epoch spent in compute phases (excludes starvation and comm)."""
        compute = sum(p.total_time_ms for p in self.phases.values())
        return 100.0 * compute / max(self.total_epoch_ms, 1)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Epoch {self.epoch:03d} | Batches: {self.batches} | Total: {self.total_epoch_ms / 1000:.1f}s",
            f"  GPU utilisation (compute): {self.gpu_utilization_pct:.1f}%",
            f"  DataLoader starvation:     {self.loader_wait_ms:.1f} ms  "
            f"({100 * self.loader_wait_ms / max(self.total_epoch_ms, 1):.1f}%)",
            f"  NCCL / comm overhead:      {self.nccl_ms:.1f} ms  "
            f"({100 * self.nccl_ms / max(self.total_epoch_ms, 1):.1f}%)",
        ]
        for ps in self.phases.values():
            lines.append(
                f"  Phase [{ps.name:12s}]  avg {ps.avg_time_ms:7.1f} ms/batch | "
                f"peak VRAM {ps.peak_vram_mb:6.0f} MB | Δ {ps.vram_delta_mb:+6.0f} MB"
            )
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# 2. DDP HOOK — times every gradient all-reduce
# ─────────────────────────────────────────────────────────────────────────────

class _CommTimerHook:
    """
    Registered as a DDP comm_hook to time each gradient all-reduce bucket.

    Only gradient communication passes through this hook. All-reduces issued
    by SyncBatchNorm in the forward pass do not; they show up in the
    torch.profiler trace as ncclAllReduce kernels instead.
    """

    def __init__(self):
        self.accumulated_ms: float = 0.0

    def hook(self, process_group, bucket):
        t0 = time.perf_counter()
        fut = dist.all_reduce(bucket.buffer(), group=process_group, async_op=True).get_future()

        def _done(fut):
            self.accumulated_ms += (time.perf_counter() - t0) * 1000.0
            result = fut.value()
            return result[0] if isinstance(result, list) else result

        return fut.then(_done)

    def reset(self):
        self.accumulated_ms = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TrainingProfiler:
    """
    Non-invasive instrumentation for DDP training.

    Parameters
    ----------
    save_dir            : directory for the reports
    global_rank         : global rank of this process
    enabled             : False turns every method into a no-op
    profile_torch       : enable torch.profiler (disable when running under nsys)
    profile_epoch_start : first epoch captured by torch.profiler
    profile_epoch_count : number of consecutive epochs to capture (0 = none)

    Output (per captured epoch):
      chrome_trace_epoch{N}_rank{R}.json  — per-rank trace
      chrome_trace_epoch{N}_merged.json   — all ranks on one timeline (Perfetto)
      op_table_epoch{N}.txt               — top-30 ops by CUDA time (rank 0)
      nccl_ops_epoch{N}.txt               — NCCL / all-reduce rows (rank 0)
      memory_table_epoch{N}.txt           — top-20 ops by memory (rank 0)
    """

    def __init__(self, save_dir: str, global_rank: int, enabled: bool = True,
                 profile_torch: bool = True, profile_epoch_start: int = 2,
                 profile_epoch_count: int = 0):
        self.save_dir = Path(save_dir)
        self.rank = global_rank
        self.enabled = enabled
        self.do_torch_prof = profile_torch
        self.profile_epoch_start = profile_epoch_start
        self.profile_epoch_end = profile_epoch_start + profile_epoch_count

        self._current_epoch: Optional[EpochReport] = None
        self._epoch_t0: Optional[float] = None
        self._comm_hook: Optional[_CommTimerHook] = None
        self._torch_profiler = None
        self._all_reports: list[EpochReport] = []

        if self.enabled and self.rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.save_dir / "profiling_summary.csv"
            with open(self._csv_path, "w") as f:
                f.write("epoch,phase,avg_ms,total_ms,peak_vram_mb,vram_delta_mb,"
                        "loader_starvation_ms,nccl_ms,gpu_util_pct\n")

    # ── DDP model registration ─────────────────────────────────────────────

    def register_ddp_model(self, ddp_model) -> None:
        if not self.enabled:
            return
        self._comm_hook = _CommTimerHook()
        ddp_model.register_comm_hook(state=dist.GroupMember.WORLD, hook=self._comm_hook.hook)
        if self.rank == 0:
            logging.info("[Profiler] Comm hook registered on the DDP model.")

    # ── per-epoch context manager ──────────────────────────────────────────

    @contextlib.contextmanager
    def profile_epoch(self, epoch: int):
        """Wrap a whole epoch; the report is flushed at the end."""
        if not self.enabled:
            yield
            return

        self._current_epoch = EpochReport(epoch=epoch)
        self._epoch_t0 = time.perf_counter()
        if self._comm_hook:
            self._comm_hook.reset()

        use_torch_prof = self.do_torch_prof and self.profile_epoch_start <= epoch < self.profile_epoch_end
        if use_torch_prof:
            # No schedule: the whole epoch is captured. Shapes/stack/memory are
            # off to keep the JSON manageable (~20-80 MB per rank).
            self._torch_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                on_trace_ready=self._on_trace_ready,
            )
            self._torch_profiler.__enter__()
            if self.rank == 0:
                logging.info(f"[Profiler] torch.profiler ACTIVE for epoch {epoch} "
                             f"(epochs {self.profile_epoch_start}–{self.profile_epoch_end - 1} captured).")

        try:
            yield
        finally:
            if use_torch_prof and self._torch_profiler:
                self._torch_profiler.__exit__(None, None, None)   # triggers _on_trace_ready
                self._torch_profiler = None
                if dist.is_initialized():
                    dist.barrier()   # every rank must have written its trace before the merge
                if self.rank == 0:
                    self._merge_rank_traces(epoch)

            self._current_epoch.total_epoch_ms = (time.perf_counter() - self._epoch_t0) * 1000.0
            if self._comm_hook:
                self._current_epoch.nccl_ms = self._comm_hook.accumulated_ms

            self._all_reports.append(self._current_epoch)
            self._flush_epoch_report(self._current_epoch)

    def _on_trace_ready(self, prof):
        """Called at the end of each captured epoch — every rank writes its trace."""
        epoch = self._current_epoch.epoch if self._current_epoch else 0

        trace_path = self.save_dir / f"chrome_trace_epoch{epoch:03d}_rank{self.rank}.json"
        prof.export_chrome_trace(str(trace_path))
        logging.info(f"[Profiler] Chrome trace rank{self.rank} -> {trace_path}")

        if self.rank != 0:
            return

        key_table = prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", row_limit=30)
        table_path = self.save_dir / f"op_table_epoch{epoch:03d}.txt"
        table_path.write_text(key_table)
        logging.info(f"[Profiler] Op table -> {table_path}")

        nccl_lines = [l for l in key_table.split("\n")
                      if any(k in l.lower() for k in ["nccl", "allreduce", "all_reduce", "sync_batch", "syncbatch"])]
        if nccl_lines:
            nccl_path = self.save_dir / f"nccl_ops_epoch{epoch:03d}.txt"
            nccl_path.write_text("NCCL / SyncBatchNorm operations:\n" + "\n".join(nccl_lines))
            logging.info(f"[Profiler] NCCL ops -> {nccl_path}")

        mem_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20)
        mem_path = self.save_dir / f"memory_table_epoch{epoch:03d}.txt"
        mem_path.write_text(mem_table)
        logging.info(f"[Profiler] Memory table -> {mem_path}")

    def _merge_rank_traces(self, epoch: int) -> None:
        """
        Merge the traces of every rank into one Perfetto file.

        torch.profiler timestamps come from CLOCK_MONOTONIC (via libkineto),
        which is shared by processes on the same node, so no offset is needed.
        Each rank becomes a separate PID (pid=r -> rank r).
        """
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        merged: list = [
            {"name": "process_name", "ph": "M", "pid": r, "tid": 0, "args": {"name": f"Rank {r} — GPU {r}"}}
            for r in range(world_size)
        ]

        for r in range(world_size):
            trace_path = self.save_dir / f"chrome_trace_epoch{epoch:03d}_rank{r}.json"
            if not trace_path.exists():
                logging.warning(f"[Profiler] Trace missing for merge: {trace_path}")
                continue
            with open(trace_path) as fh:
                data = json.load(fh)
            events = data if isinstance(data, list) else data.get("traceEvents", [])
            for ev in events:
                ev = dict(ev)
                ev["pid"] = r
                merged.append(ev)

        merged_path = self.save_dir / f"chrome_trace_epoch{epoch:03d}_merged.json"
        with open(merged_path, "w") as fh:
            json.dump(merged, fh)

        size_mb = merged_path.stat().st_size / 1024 / 1024
        logging.info(f"[Profiler] Merged trace ({world_size} ranks) -> {merged_path} "
                     f"({size_mb:.1f} MB) | open at https://ui.perfetto.dev")

    # ── data-loader iterator with starvation measurement ───────────────────

    def iter_dataloader(self, loader) -> Iterator:
        if not self.enabled or self._current_epoch is None:
            yield from loader
            return

        precise = self._torch_profiler is not None
        wait_total_ms = 0.0
        batch_count = 0
        iterator = iter(loader)

        while True:
            t0 = time.perf_counter()
            torch.cuda.nvtx.range_push("algernon/loader_wait")
            try:
                batch = next(iterator)   # the CPU blocks here waiting for the workers
            except StopIteration:
                torch.cuda.nvtx.range_pop()
                break
            torch.cuda.nvtx.range_pop()

            wait_total_ms += (time.perf_counter() - t0) * 1000.0
            batch_count += 1

            yield batch

            if precise and torch.cuda.is_available():
                torch.cuda.synchronize()

        self._current_epoch.loader_wait_ms += wait_total_ms
        self._current_epoch.batches += batch_count

        if batch_count > 0 and self.rank == 0:
            avg_stall = wait_total_ms / batch_count
            if avg_stall > 50:
                logging.warning(f"[Profiler] High starvation: {avg_stall:.0f} ms/batch")

    # ── per-phase context manager ──────────────────────────────────────────

    @contextlib.contextmanager
    def phase(self, name: str):
        if not self.enabled or self._current_epoch is None:
            yield
            return

        precise = self._torch_profiler is not None
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if precise:
            torch.cuda.synchronize(device)

        mem_before_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()

        torch.cuda.nvtx.range_push(f"algernon/{name}")
        with torch.profiler.record_function(f"algernon/{name}") if self._torch_profiler else contextlib.nullcontext():
            yield
        torch.cuda.nvtx.range_pop()

        if precise:
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mem_after_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        ep = self._current_epoch
        ps = ep.phases.setdefault(name, PhaseStats(name=name))
        ps.total_time_ms += elapsed_ms
        ps.call_count += 1
        ps.peak_vram_mb = max(ps.peak_vram_mb, peak_mb)
        ps.vram_delta_mb += (mem_after_mb - mem_before_mb)

        if name == "optimizer" and self._torch_profiler:
            self._torch_profiler.step()

    # ── instantaneous memory snapshot ──────────────────────────────────────

    def log_memory_checkpoint(self, label: str = "") -> dict:
        if not self.enabled:
            return {}
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        allocated = torch.cuda.memory_allocated(dev) / 1024 ** 2
        reserved = torch.cuda.memory_reserved(dev) / 1024 ** 2
        free = (torch.cuda.get_device_properties(dev).total_memory - torch.cuda.memory_reserved(dev)) / 1024 ** 2
        stats = {"label": label, "allocated_mb": round(allocated, 1),
                 "reserved_mb": round(reserved, 1), "free_mb": round(free, 1)}
        if self.rank == 0:
            logging.info(f"[VRAM] {label:30s} | allocated: {allocated:6.0f} MB | "
                         f"reserved: {reserved:6.0f} MB | free: {free:6.0f} MB")
        return stats

    # ── epoch report flush ─────────────────────────────────────────────────

    def _flush_epoch_report(self, report: EpochReport) -> None:
        if self.rank != 0:
            return

        for line in report.summary_lines():
            logging.info(line)

        with open(self.save_dir / "profiling_log.txt", "a") as f:
            f.write("\n".join(report.summary_lines()) + "\n\n")

        with open(self._csv_path, "a") as f:
            for ps in report.phases.values():
                f.write(f"{report.epoch},{ps.name},{ps.avg_time_ms:.2f},{ps.total_time_ms:.2f},"
                        f"{ps.peak_vram_mb:.1f},{ps.vram_delta_mb:.1f},{report.loader_wait_ms:.1f},"
                        f"{report.nccl_ms:.1f},{report.gpu_utilization_pct:.1f}\n")

    # ── final report ───────────────────────────────────────────────────────

    def save_final_report(self) -> None:
        """Aggregate analysis over all epochs. Call after the training loop."""
        if self.rank != 0 or not self.enabled or not self._all_reports:
            return

        path = self.save_dir / "profiling_final_report.txt"
        lines = ["=" * 70, "FINAL PROFILING REPORT — DDP TRAINING", "=" * 70, ""]

        # Epoch 0 is excluded from the averages: it includes initialisation
        # overhead (cuDNN autotuning, worker spawn, caching-allocator warm-up).
        epoch0_reports = [r for r in self._all_reports if r.epoch == 0]
        steady_reports = [r for r in self._all_reports if r.epoch > 0] or self._all_reports
        # Epochs captured by torch.profiler synchronise around every phase and
        # therefore measure real GPU time; the others measure CPU launch time.
        synced_reports = [r for r in self._all_reports
                          if self.profile_epoch_start <= r.epoch < self.profile_epoch_end]

        if epoch0_reports:
            e0 = epoch0_reports[0]
            lines.append("EPOCH 0 (reference — includes initialisation overhead):")
            for ps in e0.phases.values():
                lines.append(f"  [{ps.name:12s}]  avg {ps.avg_time_ms:7.1f} ms/batch | "
                             f"peak VRAM {ps.peak_vram_mb:6.0f} MB | Δ {ps.vram_delta_mb:+6.0f} MB")
            lines.append(f"  Starvation: {e0.loader_wait_ms:.0f} ms | NCCL: {e0.nccl_ms:.0f} ms | "
                         f"GPU util: {e0.gpu_utilization_pct:.1f}%")
            lines.append("")

        phase_source = synced_reports if synced_reports else steady_reports
        phase_label = (f"EPOCHS {self.profile_epoch_start}–{self.profile_epoch_end - 1} "
                       f"— GPU-accurate (cudaDeviceSynchronize active)"
                       if synced_reports else
                       "STEADY-STATE MEANS (no sync — CPU launch time only)")
        phase_names = set(name for r in phase_source for name in r.phases.keys())
        lines.append(f"PHASE TIMES [{phase_label}]:")
        for pname in sorted(phase_names):
            times = [r.phases[pname].avg_time_ms for r in phase_source if pname in r.phases]
            vrams = [r.phases[pname].peak_vram_mb for r in phase_source if pname in r.phases]
            if times:
                lines.append(f"  {pname:12s} -> {sum(times) / len(times):.1f} ms/batch | "
                             f"avg VRAM {sum(vrams) / len(vrams):.0f} MB | peak VRAM {max(vrams):.0f} MB")

        # Epoch time, starvation and NCCL are wall-clock, hence valid on every epoch.
        n = len(steady_reports)
        avg_epoch_ms = sum(r.total_epoch_ms for r in steady_reports) / n
        avg_stv = sum(r.loader_wait_ms for r in steady_reports) / n
        avg_nccl = sum(r.nccl_ms for r in steady_reports) / n
        avg_util = (sum(r.gpu_utilization_pct for r in synced_reports) / len(synced_reports)
                    if synced_reports else sum(r.gpu_utilization_pct for r in steady_reports) / n)

        lines += [
            "",
            f"Mean epoch time (steady state, {n} epochs):  {avg_epoch_ms / 1000:.2f} s",
            f"DataLoader starvation (mean):   {avg_stv:.0f} ms/epoch",
            f"NCCL gradient comm (mean):      {avg_nccl:.0f} ms/epoch",
            f"GPU utilisation (compute):      {avg_util:.1f}%",
            "=" * 70,
        ]

        path.write_text("\n".join(lines))
        logging.info(f"[Profiler] Final report -> {path}")
