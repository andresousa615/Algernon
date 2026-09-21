#!/usr/bin/env python3
"""
Per-rank nsys launcher with rank-aware trace flags.

Usage: torchrun ... profiling/nsys_launcher.py train.py [args...]

Each rank replaces itself with its own nsys process (os.execvp). One nsys per
rank = one CUDA context per nsys = no "Wrong event order" from mixed contexts.

Flags per rank:
  LOCAL_RANK 0  -> -t cuda,nvtx  GPU kernels + NVTX (forward/backward/all-reduce/
                                 augmentation). The only rank touching CUPTI on
                                 this node, so no collection conflicts.
  LOCAL_RANK >0 -> -t nvtx       NVTX only (worker augmentation, loader wait,
                                 epoch/batch markers). No CUDA events, no
                                 ordering risk.

Environment:
  RANK, LOCAL_RANK  — injected by torchrun
  NSYS_PROFDIR      — output directory (required)
"""
import os
import sys

rank = os.environ.get("RANK", "0")
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
profdir = os.environ.get("NSYS_PROFDIR")

if profdir is None:
    print("[nsys_launcher] ERROR: NSYS_PROFDIR is not set.", flush=True)
    sys.exit(1)

trace_flags = "cuda,nvtx" if local_rank == 0 else "nvtx"

# --capture-range=cudaProfilerApi: CUPTI only active between cudaProfilerStart() and
# cudaProfilerStop() (called in the training script at batch NSYS_WARMUP_STEPS and
# NSYS_WARMUP_STEPS+NSYS_PROFILE_STEPS). This keeps CUPTI off during dist.init and
# the warmup batches — no CUPTI overhead on NCCL AllReduce → clean optimizer timings.
#
# "Wrong event order" (QdstrmImporter non-fatal warning): background threads in the
# main process (_pin_thread, dispatch thread) may have CUPTI-buffered events with
# pre-capture timestamps. These are misordered in the .qdstrm stream but QdstrmImporter
# handles them as non-fatal — it still generates a valid .nsys-rep. The pipeline uses
# [[ -s "$nsys_rep" ]] (file exists and non-empty) to decide success, NOT exit code,
# so these warnings never cause the file to be deleted.
#
# --trace-fork-before-exec=false: excludes mp.Process fork workers from trace.
# --wait=primary: nsys finalises when main process exits, not when workers exit.
extra_flags = [
    "--capture-range=cudaProfilerApi",
    "--capture-range-end=stop",
    "--trace-fork-before-exec=false",
    "--wait=primary",
]

nsys_cmd = [
    "nsys", "profile",
    "-t", trace_flags,
    "-s", "none",
    "--cuda-memory-usage=false",
    "-f", "true",
    f"--output={profdir}/rank{rank}",
] + extra_flags + [sys.executable] + sys.argv[1:]

os.execvp("nsys", nsys_cmd)
