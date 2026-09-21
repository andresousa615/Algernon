#!/usr/bin/env python3
"""
simulate_loader.py — analytical simulation of data-loader strategies.

Discrete-event simulation (no sleeps, no GPU) comparing:

  A) PyTorch DataLoader   — in-order delivery, head-of-line blocking
  B) MinatoSegLoader      — work stealing, out-of-order delivery
  C) MinatoSegLoader + warm-up tranche (affine samples first)

Default parameters come from nsys measurements on an A100 (MedNeXt, bs=1):
  - RandomElasticDeformation : ~575 ms
  - RandomAffine             : ~148 ms
  - GPU step (fwd+bwd+opt)   : ~200 ms

Usage:
  python profiling/simulate_loader.py
  python profiling/simulate_loader.py --workers 8 12 --samples 99 --epochs 10 --seed 42
  python profiling/simulate_loader.py --csv results.csv
"""

import argparse
import csv
import heapq
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# Augmentation / step costs (ms), measured with nsys
ELASTIC_MS = 575.0
AFFINE_MS = 148.0
GPU_STEP_MS = 200.0
ELASTIC_PROB = 0.25
JITTER = 0.10   # ±10 % variance on augmentation times


@dataclass
class EpochResult:
    epoch: int
    strategy: str
    n_workers: int
    total_ms: float
    starvation_ms: float
    batches: int
    n_elastic: int

    @property
    def starvation_pct(self) -> float:
        return 100.0 * self.starvation_ms / max(self.total_ms, 1.0)

    @property
    def throughput(self) -> float:
        return 1000.0 * self.batches / max(self.total_ms, 1.0)


def _aug_ms(is_elastic: bool, rng: random.Random) -> float:
    base = ELASTIC_MS if is_elastic else AFFINE_MS
    return base * rng.uniform(1.0 - JITTER, 1.0 + JITTER)


# ─────────────────────────────────────────────────────────────────────────────
# A) PyTorch DataLoader — in-order, head-of-line blocking
# ─────────────────────────────────────────────────────────────────────────────

def sim_inorder(n_samples: int, num_workers: int, epoch: int, rng: random.Random) -> EpochResult:
    """
    Round-robin assignment of indices to workers; the consumer reads in
    assignment order (item k -> worker k % W). If the next worker is busy with
    a slow (elastic) sample the GPU waits, even though other workers already
    have samples ready.
    """
    is_elastic = [rng.random() < ELASTIC_PROB for _ in range(n_samples)]

    worker_done: List[List[float]] = [[] for _ in range(num_workers)]
    worker_elastic: List[List[bool]] = [[] for _ in range(num_workers)]
    for k, elast in enumerate(is_elastic):
        w = k % num_workers
        prev = worker_done[w][-1] if worker_done[w] else 0.0
        worker_done[w].append(prev + _aug_ms(elast, rng))
        worker_elastic[w].append(elast)

    gpu_free_ms = 0.0
    starvation_ms = 0.0
    ptr = [0] * num_workers
    n_elastic_total = 0

    for k in range(n_samples):
        w = k % num_workers
        j = ptr[w]
        if j >= len(worker_done[w]):
            break
        item_ready = worker_done[w][j]
        elast = worker_elastic[w][j]
        ptr[w] += 1

        if item_ready > gpu_free_ms:          # head-of-line blocking
            starvation_ms += item_ready - gpu_free_ms
            gpu_free_ms = item_ready

        gpu_free_ms += GPU_STEP_MS
        if elast:
            n_elastic_total += 1

    return EpochResult(epoch=epoch, strategy="PyTorch in-order", n_workers=num_workers,
                       total_ms=gpu_free_ms, starvation_ms=starvation_ms,
                       batches=n_samples, n_elastic=n_elastic_total)


# ─────────────────────────────────────────────────────────────────────────────
# Shared core: analytical work stealing
# ─────────────────────────────────────────────────────────────────────────────

def _run_workstealing(items_elastic: List[bool], num_workers: int, epoch: int,
                      strategy_name: str, rng: random.Random) -> EpochResult:
    """
    Workers pull items from the shared queue as soon as they are free; the
    consumer reads whichever item completes first. Implemented with an event
    heap, so the result is immediate.
    """
    n = len(items_elastic)
    pending = list(items_elastic)

    worker_heap = [(0.0, w) for w in range(num_workers)]   # (t_free, worker_id)
    heapq.heapify(worker_heap)
    done_heap: List[Tuple[float, bool]] = []                # (t_done, is_elastic)

    while pending:
        t_free, wid = heapq.heappop(worker_heap)
        elast = pending.pop(0)
        t_done = t_free + _aug_ms(elast, rng)
        heapq.heappush(done_heap, (t_done, elast))
        heapq.heappush(worker_heap, (t_done, wid))

    gpu_free_ms = 0.0
    starvation_ms = 0.0
    n_elastic_total = 0

    while done_heap:
        t_done, elast = heapq.heappop(done_heap)
        if t_done > gpu_free_ms:
            starvation_ms += t_done - gpu_free_ms
            gpu_free_ms = t_done
        gpu_free_ms += GPU_STEP_MS
        if elast:
            n_elastic_total += 1

    return EpochResult(epoch=epoch, strategy=strategy_name, n_workers=num_workers,
                       total_ms=gpu_free_ms, starvation_ms=starvation_ms,
                       batches=n, n_elastic=n_elastic_total)


def sim_workstealing(n_samples: int, num_workers: int, epoch: int, rng: random.Random) -> EpochResult:
    """Work stealing with a random distribution of transforms."""
    items = [rng.random() < ELASTIC_PROB for _ in range(n_samples)]
    return _run_workstealing(items, num_workers, epoch, "MinatoSegLoader (work-stealing)", rng)


def sim_warmup_tranche(n_samples: int, num_workers: int, epoch: int, rng: random.Random) -> EpochResult:
    """
    Warm-up tranche: the work queue is pre-ordered as
      [affine x num_workers]  -> the GPU starts after ~148 ms
      [elastic x n_elastic]   -> processed while the GPU has a runway
      [affine x rest]         -> fast tail guaranteed
    """
    n_elastic = round(n_samples * ELASTIC_PROB)
    n_affine = n_samples - n_elastic
    n_warmup = min(num_workers, n_affine)
    items = [False] * n_warmup + [True] * n_elastic + [False] * (n_affine - n_warmup)
    assert len(items) == n_samples
    return _run_workstealing(items, num_workers, epoch, "MinatoSegLoader + warmup tranche", rng)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics and output
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


def summarise(results: List[EpochResult]) -> dict:
    return {
        "total_ms": _mean_std([r.total_ms for r in results]),
        "starvation_ms": _mean_std([r.starvation_ms for r in results]),
        "starvation_pct": _mean_std([r.starvation_pct for r in results]),
        "throughput": _mean_std([r.throughput for r in results]),
        "n_elastic": _mean_std([r.n_elastic for r in results]),
    }


STRATEGIES = [
    ("PyTorch in-order", sim_inorder),
    ("MinatoSegLoader (work-stealing)", sim_workstealing),
    ("MinatoSegLoader + warmup tranche", sim_warmup_tranche),
]


def print_table(all_results: dict, n_samples: int, n_epochs: int) -> None:
    col_w = [36, 10, 12, 18, 16, 14]
    header = ["Strategy", "Workers", "Epoch (s)", "Starvation (ms)", "Starvation %", "Throughput"]

    def row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    sep = "─" * (sum(col_w) + 2 * (len(col_w) - 1))
    print()
    print(f"  DATALOADER SIMULATION  |  {n_samples} samples/rank  |  {n_epochs} epochs")
    print(f"  Elastic: {ELASTIC_PROB * 100:.0f}%  |  elastic={ELASTIC_MS:.0f}ms  "
          f"affine={AFFINE_MS:.0f}ms  GPU={GPU_STEP_MS:.0f}ms/batch")
    print(sep)
    print(row(header))
    print(sep)

    prev_workers = None
    for (strat, nw), results in sorted(all_results.items(), key=lambda x: (x[0][1], x[0][0])):
        if nw != prev_workers and prev_workers is not None:
            print()
        prev_workers = nw
        s = summarise(results)
        print(row([strat, nw,
                   f"{s['total_ms'][0] / 1000:.1f}±{s['total_ms'][1] / 1000:.1f}s",
                   f"{s['starvation_ms'][0]:.0f}±{s['starvation_ms'][1]:.0f}",
                   f"{s['starvation_pct'][0]:.2f}±{s['starvation_pct'][1]:.2f}%",
                   f"{s['throughput'][0]:.2f} b/s"]))
    print(sep)


def print_detail(all_results: dict) -> None:
    print()
    print("  PER-EPOCH DETAIL (starvation ms):")
    for (strat, nw), results in sorted(all_results.items(), key=lambda x: (x[0][1], x[0][0])):
        vals = "  ".join(f"{r.starvation_ms:.0f}" for r in results[:10])
        print(f"  [{nw}w] {strat[:35]:35s}  {vals}")


def write_csv(path: str, all_results: dict) -> None:
    rows = []
    for (strat, nw), results in all_results.items():
        for r in results:
            rows.append({
                "strategy": strat, "n_workers": nw, "epoch": r.epoch,
                "total_ms": round(r.total_ms, 1), "starvation_ms": round(r.starvation_ms, 1),
                "starvation_pct": round(r.starvation_pct, 3), "throughput": round(r.throughput, 3),
                "n_elastic": r.n_elastic,
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical simulation of data-loader strategies",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 12], help="num_workers values to test")
    parser.add_argument("--samples", type=int, default=99, help="Samples per rank per epoch")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs to simulate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default=None, help="Save results to this CSV")
    parser.add_argument("--detail", action="store_true", help="Print per-epoch starvation")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_results: dict = {}

    print(f"\n  Simulating {len(STRATEGIES)} strategies x {len(args.workers)} worker configs "
          f"x {args.epochs} epochs...", flush=True)

    for nw in args.workers:
        for name, fn in STRATEGIES:
            key = (name, nw)
            all_results[key] = []
            for ep in range(args.epochs + 1):      # epoch 0 excluded (start-up)
                r = fn(args.samples, nw, ep, rng)
                if ep > 0:
                    all_results[key].append(r)
        print(f"  workers={nw} done", flush=True)

    print_table(all_results, args.samples, args.epochs)
    if args.detail:
        print_detail(all_results)

    print()
    for nw in args.workers:
        base = summarise(all_results[("PyTorch in-order", nw)])
        steal = summarise(all_results[("MinatoSegLoader (work-stealing)", nw)])
        tranche = summarise(all_results[("MinatoSegLoader + warmup tranche", nw)])
        base_starv, base_epoch = base["starvation_ms"][0], base["total_ms"][0]
        for label, s in (("work stealing", steal), ("warm-up tranche", tranche)):
            red = base_starv - s["starvation_ms"][0]
            print(f"  [workers={nw}]  starvation PyTorch -> {label}: {base_starv:.0f}ms -> "
                  f"{s['starvation_ms'][0]:.0f}ms  (reduction {red:.0f}ms = "
                  f"{100 * red / max(base_epoch, 1):.1f}% of the epoch)")

    if args.csv:
        write_csv(args.csv, all_results)

    print()
    print("  NOTE: this simulation does not include:")
    print("    - IPC overhead (pickle/unpickle) of the PyTorch DataLoader")
    print("    - Python GIL contention with many workers")
    print("    - queue_size back-pressure when the queue is full")
    print("    - I/O variance when reading .nii.gz files")
    print("  -> real PyTorch DataLoader starvation is typically HIGHER than simulated here.")
    print()


if __name__ == "__main__":
    main()
