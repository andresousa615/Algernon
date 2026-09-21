"""
minato_loader.py — Out-of-order, work-stealing data loader with a background
pin-memory thread.

Architecture
------------

  worker processes ──→ _out_queue (shared memory) ──→ _pin_thread (background)
                                                            │
                                                     stack + pin_memory
                                                            │
                                                     _pinned_queue (threading.Queue)
                                                            │
                                                     __next__() ≈ µs

The pin thread runs concurrently with the GPU forward/backward pass, so the
stack + pin_memory cost never sits on the training loop's critical path.

What it fixes compared to torch.utils.data.DataLoader
-----------------------------------------------------
1. Head-of-line blocking: samples are delivered as soon as they are ready,
   not in index order, so one slow augmentation does not stall the batch.
2. Tail imbalance: a shared work queue (work stealing) keeps every worker busy
   until the last index of the epoch is processed.
3. Loader wait ≈ µs: stacking and pinning happen in the background, like the
   DataLoader's internal _PinMemoryThread.

Epoch protocol
--------------
  _dispatch_epoch():
    1. Sends the epoch config to the pin thread via _epoch_config_queue.
    2. Pushes (index, augmentation) pairs plus num_workers None sentinels
       onto _work_queue.

  _pin_thread_fn():
    - Reads tensors from _out_queue (shared memory, zero-copy on the consumer).
    - Counts real items and sentinels separately. The epoch is complete when
      items_received >= n_items AND sentinels_seen >= n_workers, which is robust
      to a fast worker draining several sentinels early.
    - Puts _EPOCH_DONE on _pinned_queue when the epoch ends.

  __next__():
    - Pops a pre-pinned batch from _pinned_queue.

  _drain_previous_epoch():
    - Consumes _pinned_queue up to _EPOCH_DONE before dispatching a new epoch.

Augmentation schedule
---------------------
Each index is tagged "affine" or "elastic" by the loader (3:1 interleaving,
exactly round(n * elastic_fraction) elastic samples per epoch). The worker
exports the tag through the MINATO_FORCE_TRANSFORM environment variable, which
SegmentationDataset reads. This spreads the expensive elastic deformations
evenly over time instead of letting them cluster on one worker.
"""

import csv
import logging
import os
import random
import threading
import time
from queue import Empty
from queue import Queue as ThreadQueue

import torch
import torch.multiprocessing as mp

_SHUTDOWN = -1          # worker shutdown sentinel (never a valid index)
_EPOCH_DONE = object()  # marker on _pinned_queue: epoch finished

# Worker state codes (stored in the shared _worker_stats array)
_WSTATE_WAITING = 0   # blocked on work_queue.get()
_WSTATE_WORKING = 1   # running dataset[idx] (I/O + augmentation)

# Per-worker layout in the shared array: [state, items_done, last_output_ns]
_N_SLOTS = 3
_SLOT_STATE = 0
_SLOT_ITEMS = 1
_SLOT_NS = 2

_log = logging.getLogger("MinatoLoader")


def _safe_qsize(q) -> int:
    """mp.Queue.qsize() is not implemented on macOS; report -1 there."""
    try:
        return q.qsize()
    except NotImplementedError:
        return -1


# ---------------------------------------------------------------------------
# Persistent worker (module level — picklable for fork)
# ---------------------------------------------------------------------------

def _worker_fn(dataset, work_queue: mp.Queue, out_queue: mp.Queue, worker_id: int,
               worker_stats: mp.Array) -> None:
    """
    Producer process with work stealing.

    Pulls (idx, aug_type) from the shared work queue, runs dataset[idx] and
    puts the tensors on out_queue. torch.multiprocessing's ForkingPickler
    calls storage.share_memory_() automatically: only a handle travels through
    the pipe and the consumer mmaps the data without a copy.
    """
    torch.set_num_threads(1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[var] = "1"
    os.environ["MINATO_WORKER_ID"] = str(worker_id)

    rank = os.environ.get("RANK", "?")
    base = worker_id * _N_SLOTS

    while True:
        worker_stats[base + _SLOT_STATE] = _WSTATE_WAITING
        raw = work_queue.get()

        if raw == _SHUTDOWN:
            break

        if raw is None:              # end-of-epoch sentinel: forward it
            out_queue.put(None)
            continue

        # Sentinels are never tuples, so unpacking is unambiguous.
        idx, aug_type = raw
        os.environ["MINATO_FORCE_TRANSFORM"] = aug_type  # process-local
        worker_stats[base + _SLOT_STATE] = _WSTATE_WORKING
        try:
            item = dataset[idx]
            out_queue.put((item["image"].contiguous(), item["mask"].contiguous()))
            worker_stats[base + _SLOT_ITEMS] += 1
            worker_stats[base + _SLOT_NS] = time.perf_counter_ns()
        except Exception as exc:
            _log.error("[rank=%s worker=%d] idx=%s failed: %s", rank, worker_id, idx, exc)


# ---------------------------------------------------------------------------
# Pin thread (background, in-process)
# ---------------------------------------------------------------------------

def _pin_thread_fn(out_queue: mp.Queue, pinned_queue: ThreadQueue, epoch_config_queue: ThreadQueue,
                   batch_size: int, pin_memory: bool, pin_device: str,
                   stop_event: threading.Event) -> None:
    """
    Read tensors from shared memory, assemble batches, pin them.

    Counters and the partial-batch buffer are reset only when an epoch is
    declared complete, never when a config arrives: with a fast epoch
    turnaround the first items of the next epoch can reach this thread before
    its config does, and resetting on config would silently drop them.
    """
    n_items_expected = 0
    n_workers_expected = 0
    items_received = 0
    sentinels_seen = 0
    buffer: list = []
    epoch_active = False

    def _epoch_done() -> bool:
        return (epoch_active and sentinels_seen >= n_workers_expected
                and items_received >= n_items_expected)

    while not stop_event.is_set():
        # Non-blocking check for a new epoch config
        try:
            cfg = epoch_config_queue.get_nowait()
            n_items_expected = cfg["n_items"]
            n_workers_expected = cfg["n_workers"]
            epoch_active = True
        except Empty:
            pass

        if not _epoch_done():
            try:
                item = out_queue.get(timeout=0.05)
            except Empty:
                continue

            if item is None:
                sentinels_seen += 1
            else:
                items_received += 1
                buffer.append(item)

                if len(buffer) >= batch_size:
                    batch, buffer = buffer[:batch_size], buffer[batch_size:]
                    images = torch.stack([t[0] for t in batch])
                    masks = torch.stack([t[1] for t in batch])
                    if pin_memory:
                        images = images.pin_memory(pin_device)
                        masks = masks.pin_memory(pin_device)
                    pinned_queue.put({"image": images, "mask": masks})

        if _epoch_done():
            buffer.clear()            # drop_last: leftover items of an incomplete batch
            items_received = 0
            sentinels_seen = 0
            epoch_active = False
            pinned_queue.put(_EPOCH_DONE)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_fork_ctx = mp.get_context("fork")


class MinatoSegLoader:
    """
    Drop-in replacement for DataLoader over a SegmentationDataset under DDP.

    Args:
        dataset:           Dataset whose __getitem__ returns {"image": tensor, "mask": tensor}.
        sampler:           Typically a DistributedSampler. Exposed as .sampler.
        batch_size:        Samples per batch.
        num_workers:       Persistent producer processes.
        pin_memory:        Pin tensors before handing them to the training loop.
        pin_memory_device: CUDA device for pinning (e.g. "cuda:0"). Empty = current device.
        queue_size:        Shared-memory buffer between workers and the pin thread.
        drop_last:         Drop the last incomplete batch (default True).
        elastic_fraction:  Fraction of samples per epoch that receive RandomElasticDeformation.
        debug:             Verbose logging plus a per-batch queue-monitor CSV (requires log_dir).
        log_dir:           Directory for the queue-monitor CSV when debug=True.
        name:              Loader name, used in log messages and file names.
        batch_timeout:     Seconds __next__ waits for a batch before raising RuntimeError.
    """

    def __init__(self, dataset, sampler, batch_size: int = 1, num_workers: int = 8,
                 pin_memory: bool = True, pin_memory_device: str = "", queue_size: int = 32,
                 drop_last: bool = True, elastic_fraction: float = 0.25, debug: bool = False,
                 log_dir: str | None = None, name: str = "loader", batch_timeout: float = 120.0) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.queue_size = queue_size
        self.drop_last = drop_last
        self._elastic_fraction = elastic_fraction
        self.batch_timeout = batch_timeout
        self._debug = debug
        self._rank = int(os.environ.get("RANK", 0))
        self._pfx = f"[MinatoLoader {name} rank={self._rank}]"

        self._n_batches = 0
        self._batches_yielded = 0
        self._epoch_started = False
        self._current_epoch = -1

        # IPC: workers -> pin thread (tensors in shared memory)
        self._work_queue = _fork_ctx.Queue()
        self._out_queue = _fork_ctx.Queue(maxsize=queue_size)

        # Shared memory: per-worker state (state, items_done, last_output_ns)
        self._worker_stats = _fork_ctx.Array("q", num_workers * _N_SLOTS)

        # In-process: main -> pin thread (epoch config); pin thread -> main (batches)
        self._epoch_config_queue = ThreadQueue()
        self._pinned_queue = ThreadQueue(maxsize=32)

        # Optional queue-monitor CSV (one per loader per rank)
        self._queue_log_fh = None
        self._queue_log_writer = None
        if debug and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"queue_monitor_{name}_rank{self._rank}.csv")
            self._queue_log_fh = open(log_path, "w", newline="", buffering=1)
            header = ["epoch", "batch", "wait_ms", "pinned_q", "out_q", "work_q"]
            for wid in range(num_workers):
                header += [f"w{wid}_state", f"w{wid}_items", f"w{wid}_ms_since"]
            self._queue_log_writer = csv.writer(self._queue_log_fh)
            self._queue_log_writer.writerow(header)

        # Persistent workers
        self._workers: list = []
        for wid in range(num_workers):
            p = _fork_ctx.Process(
                target=_worker_fn,
                args=(dataset, self._work_queue, self._out_queue, wid, self._worker_stats),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Pin thread
        self._stop_event = threading.Event()
        self._pin_thread = threading.Thread(
            target=_pin_thread_fn,
            args=(self._out_queue, self._pinned_queue, self._epoch_config_queue,
                  batch_size, pin_memory, pin_memory_device, self._stop_event),
            daemon=True,
        )
        self._pin_thread.start()
        self._active = True

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def _drain_previous_epoch(self) -> None:
        """Consume _pinned_queue up to _EPOCH_DONE so the previous epoch is fully flushed."""
        if not self._epoch_started:
            return
        drained = 0
        while True:
            try:
                item = self._pinned_queue.get(timeout=30.0)
                if item is _EPOCH_DONE:
                    break
                drained += 1
            except Empty:
                _log.warning("%s drain timeout (30 s) — the pin thread may have crashed "
                             "(drained %d batches before timing out).", self._pfx, drained)
                break
        if drained and self._debug:
            _log.debug("%s drain: discarded %d unconsumed batch(es).", self._pfx, drained)
        self._epoch_started = False

    def _build_schedule(self, indices: list) -> list:
        """
        Tag each index with "affine" or "elastic", interleaved 3:1.

        Guarantees: (1) no two consecutive elastic samples, (2) no burst of
        elastic samples at the start of the epoch, (3) exactly
        round(n * elastic_fraction) elastic samples per epoch.
        """
        n_elastic = round(len(indices) * self._elastic_fraction)
        elastic_set = set(random.sample(indices, n_elastic))
        affine_list = [i for i in indices if i not in elastic_set]
        elastic_list = list(elastic_set)
        random.shuffle(affine_list)
        random.shuffle(elastic_list)

        schedule: list = []
        a_it, e_it = iter(affine_list), iter(elastic_list)
        a_count = 0
        exhausted_affine = exhausted_elastic = False
        while not (exhausted_affine and exhausted_elastic):
            if a_count < 3 and not exhausted_affine:
                try:
                    schedule.append((next(a_it), "affine"))
                    a_count += 1
                except StopIteration:
                    exhausted_affine = True
            elif not exhausted_elastic:
                try:
                    schedule.append((next(e_it), "elastic"))
                    a_count = 0
                except StopIteration:
                    exhausted_elastic = True
                    schedule.extend((remaining, "affine") for remaining in a_it)
                    exhausted_affine = True
            else:
                break
        schedule.extend((remaining, "elastic") for remaining in e_it)
        return schedule

    def _dispatch_epoch(self, indices: list) -> None:
        """Send the epoch config to the pin thread, then the work items to the workers."""
        now_ns = time.perf_counter_ns()
        for wid in range(self.num_workers):
            base = wid * _N_SLOTS
            self._worker_stats[base + _SLOT_NS] = now_ns
            self._worker_stats[base + _SLOT_ITEMS] = 0

        # Config goes first so the pin thread reads it before any result arrives.
        self._epoch_config_queue.put({"n_items": len(indices), "n_workers": self.num_workers})

        schedule = self._build_schedule(indices)
        for item in schedule:
            self._work_queue.put(item)
        for _ in range(self.num_workers):
            self._work_queue.put(None)

        if self._debug:
            n_ela = sum(1 for _, t in schedule if t == "elastic")
            _log.debug("%s dispatch: %d items (%d elastic, %d affine) + %d sentinels.",
                       self._pfx, len(schedule), n_ela, len(schedule) - n_ela, self.num_workers)

    def _shutdown(self) -> None:
        if not self._active:
            return
        self._active = False

        # Stop the pin thread: signal + drain queues to unblock pending puts
        self._stop_event.set()
        for q in (self._out_queue, self._pinned_queue):
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass
        self._pin_thread.join(timeout=5)

        # Stop workers
        try:
            while True:
                self._work_queue.get_nowait()
        except Exception:
            pass
        for _ in self._workers:
            self._work_queue.put(_SHUTDOWN)
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
        self._workers.clear()

        self._out_queue.close()
        try:
            self._out_queue.join_thread()
        except Exception:
            pass

        if self._queue_log_fh is not None:
            try:
                self._queue_log_fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------

    def __iter__(self) -> "MinatoSegLoader":
        self._current_epoch += 1
        self._drain_previous_epoch()

        indices = list(self.sampler)
        n = len(indices)
        if self.drop_last:
            n_batches = n // self.batch_size
            indices = indices[: n_batches * self.batch_size]
        else:
            n_batches = (n + self.batch_size - 1) // self.batch_size

        self._n_batches = n_batches
        self._batches_yielded = 0
        self._epoch_started = True

        self._dispatch_epoch(indices)
        return self

    def __next__(self) -> dict:
        if self._batches_yielded >= self._n_batches:
            if self._debug:
                _log.debug("%s epoch complete: %d/%d batches.", self._pfx,
                           self._batches_yielded, self._n_batches)
            raise StopIteration

        # Snapshot before blocking — reveals whether the queue was empty (starvation)
        pinned_q = self._pinned_queue.qsize()
        out_q = _safe_qsize(self._out_queue)
        work_q = _safe_qsize(self._work_queue)

        t0 = time.perf_counter()
        try:
            item = self._pinned_queue.get(timeout=self.batch_timeout)
        except Empty:
            raise RuntimeError(f"{self._pfx} timed out ({self.batch_timeout:.0f} s) waiting for a pinned batch. "
                               "Check the worker and pin-thread logs.")
        wait_ms = (time.perf_counter() - t0) * 1000

        if item is _EPOCH_DONE:
            _log.warning("%s _EPOCH_DONE received before all batches (%d/%d). "
                         "Possible data loss or protocol bug.", self._pfx,
                         self._batches_yielded, self._n_batches)
            raise StopIteration

        self._batches_yielded += 1

        if self._queue_log_writer is not None:
            now_ns = time.perf_counter_ns()
            row = [self._current_epoch, self._batches_yielded, round(wait_ms, 1), pinned_q, out_q, work_q]
            stats = self._worker_stats
            for wid in range(self.num_workers):
                base = wid * _N_SLOTS
                last_ns = stats[base + _SLOT_NS]
                ms_since = round((now_ns - last_ns) / 1e6) if last_ns > 0 else -1
                row += [stats[base + _SLOT_STATE], stats[base + _SLOT_ITEMS], ms_since]
            self._queue_log_writer.writerow(row)

        return item

    def __del__(self) -> None:
        self._shutdown()
