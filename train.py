# -*- coding: utf-8 -*-
# Portions derived from mede (https://pypi.org/project/mede/), Copyright 2025
# Lukas Heine & Moritz Rempe, Apache License 2.0 with Commons Clause — see
# LICENSE-mede and NOTICE.
"""
Distributed (DDP) training of the Algernon facial-structure segmentation network.

Launch with torchrun, e.g. on one node with 2 GPUs:

    torchrun --nproc_per_node=2 train.py --config configs/train_128.yaml --epochs 100

Features
--------
* FP16 autocast + GradScaler, torch.compile (max-autotune, no CUDA graphs).
* AdamW with 5-epoch linear warm-up followed by cosine annealing.
* Work-stealing out-of-order data loader (data/minato_loader.py).
* Snapshot per epoch (rank 0) for fault-tolerant resumption with --resume_id.
* Optional profiling (--profile): per-phase timing/VRAM, MFU, Chrome trace.
"""
import argparse
import logging
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
import torchvision
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from data.dataset import get_loaders
from models import build_network
from profiling.mfu import MFUTracker, compute_model_flops, verify_flop_convention
from profiling.profiling import TrainingProfiler
from utils import utilities
from utils.losses import DiceCELoss
from utils.validation import segmentation_validation

torchvision.disable_beta_transforms_warning()

# Nsight Systems capture window (in training steps). Read from the environment so
# the SLURM profiling script can set it. Without nsys attached the
# cudaProfilerStart/Stop calls are no-ops.
_NSYS_WARMUP_STEPS = int(os.environ.get("NSYS_WARMUP_STEPS", "100"))
_NSYS_PROFILE_STEPS = int(os.environ.get("NSYS_PROFILE_STEPS", "150"))

# A100-40GB peaks: 312 TFLOP/s FP16 tensor cores (optimistic under AMP),
# 156 TFLOP/s TF32 (conservative — part of the ops still run in TF32/FP32).
GPU_PEAK_TFLOPS = 312.0
GPU_PEAK_TFLOPS_SECONDARY = 156.0

parser = argparse.ArgumentParser(prog="Algernon training")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--earlystop", action="store_true", help="Enable early stopping on validation DSC")
parser.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
parser.add_argument("--resume_id", type=str, default=None,
                    help="Job ID whose snapshot should be resumed (snapshot_<model>_<id>.pt)")
parser.add_argument("--profile", action="store_true",
                    help="Enable the TrainingProfiler and MFU tracker (small overhead)")
parser.add_argument("--nsys-mode", action="store_true",
                    help="Disable torch.profiler so nsys can run without interference")
parser.add_argument("--profile-start-epoch", type=int,
                    default=int(os.environ.get("TORCH_PROF_START_EPOCH", "2")),
                    help="First epoch captured by torch.profiler (requires --profile)")
parser.add_argument("--profile-epochs", type=int,
                    default=int(os.environ.get("TORCH_PROF_EPOCHS", "0")),
                    help="Number of consecutive epochs captured by torch.profiler (0 = off)")


def ddp_setup() -> None:
    """Initialise the process group. torchrun injects the required env vars."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", timeout=timedelta(hours=1))
    logging.debug(f"DDP set up for global rank {os.environ['RANK']}, local rank {local_rank}")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.debug(f"Random seed set to {seed}")


class EarlyStopping:
    """Stop training when the monitored metric has not improved for `patience` epochs."""

    def __init__(self, patience=40, verbose=True, delta=0, monitor="val_loss", op_type="min"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type
        self.val_score_min = np.inf if op_type == "min" else 0

    def __call__(self, val_score):
        score = -val_score if self.op_type == "min" else val_score

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            if self.verbose:
                logging.info(f"{self.monitor} improved ({self.val_score_min:.6f} --> {val_score:.6f}).")
            self.val_score_min = val_score
            self.counter = 0
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class TrainNetwork:
    def __init__(self, args, config: dict) -> None:
        self.args = args
        self.config = config
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Run identifier, used to name the output directory and the snapshot.
        # ALGERNON_RUN_ID lets run_pipeline.sh pick the id so it can find the
        # results afterwards; under SLURM the job id is used automatically.
        self.current_job_id = (os.environ.get("ALGERNON_RUN_ID")
                               or os.environ.get("SLURM_JOB_ID", "standalone"))
        self.train_path: str = config["train_path"]
        self.val_path: str = config["val_path"]
        self.base_output = Path(config["base_output"])
        self.init_lr: float = config["lr"]
        self.out_channels: int = config["out_channels"]
        self.batch_size: int = int(config.get("batch_size", 1))   # per GPU
        self.epochs: int = args.epochs
        self.epochs_run = 0
        self.model_name = f"{config['model']}_{config['lr']}_{config['comment']}"

        self.save_folder = self.base_output / f"train_{self.model_name}_{self.current_job_id}"
        self.csv_metrics_path = self.save_folder / "training_metrics.csv"

        if self.global_rank == 0:
            self.save_folder.mkdir(parents=True, exist_ok=True)
            logging.info(f"Run directory: {self.save_folder}")
            if not self.csv_metrics_path.exists() or args.resume_id is None:
                with open(self.csv_metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,train_loss,val_loss,dsc,iou\n")
        dist.barrier()

        self.profiler = TrainingProfiler(
            save_dir=str(self.save_folder),
            global_rank=self.global_rank,
            enabled=args.profile,
            profile_torch=args.profile and not args.nsys_mode,
            profile_epoch_start=args.profile_start_epoch,
            profile_epoch_count=args.profile_epochs,
        )

        target_id = args.resume_id if args.resume_id else self.current_job_id
        self.snapshot_path = self.base_output / f"snapshot_{self.model_name}_{target_id}.pt"

        self.scaler = torch.amp.GradScaler("cuda")
        self.mfu_tracker = None
        self._mfu_failed = not args.profile   # MFU only when profiling is requested

        self.scheduler = None
        self.optimizer = None
        self.early_stopping = None
        self.loss = None
        self.train_loader = None
        self.val_loader = None
        self.metric = 0.0
        self.total_train_loss = None
        self.epoch = 0

        self._init_network()

    # ------------------------------------------------------------------

    def _init_network(self) -> None:
        model = build_network(self.config["model"], self.out_channels).to(self.device)
        logging.info(f"Selected model: {self.config['model']}")
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        self.model = DDP(
            model,
            device_ids=[self.local_rank],
            static_graph=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
        )

    def _init_mfu_tracker(self, data_sample: torch.Tensor) -> None:
        """
        Count forward FLOPs on the first batch and start the MFU tracker (rank 0).
        MFU is instrumentation only: any failure is logged and training continues.
        """
        try:
            raw = build_network(self.config["model"], self.out_channels).to(self.device)
            dummy = data_sample[:1].float().to(self.device)
            flops = compute_model_flops(raw, dummy)
            del raw
            torch.cuda.empty_cache()
            verify_flop_convention(device=self.device)
        except Exception as exc:
            logging.warning("[MFU] FLOPs not computed: %s", exc)
            flops = 0

        try:
            self.mfu_tracker = MFUTracker(
                forward_flops=flops,
                gpu_peak_tflops=GPU_PEAK_TFLOPS,
                warmup_epochs=4,
                world_size=self.world_size,
                gpu_peak_tflops_secondary=GPU_PEAK_TFLOPS_SECONDARY,
                samples_per_step=self.batch_size,
            )
        except Exception as exc:
            self._mfu_failed = True
            logging.warning("[MFU] Tracker not initialised (%s). Training continues without MFU.", exc)
            return

        logging.info("[MFU] Forward FLOPs=%.1f GFLOPs | GPU peak=%.0f TFLOP/s (FP16) | world_size=%d",
                     flops / 1e9, GPU_PEAK_TFLOPS, self.world_size)

    # ------------------------------------------------------------------

    def _load_snapshot(self) -> None:
        snapshot = torch.load(self.snapshot_path, map_location=f"cuda:{self.local_rank}")
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if "OPTIMIZER_STATE" in snapshot and self.optimizer is not None:
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        if "SCALER_STATE" in snapshot and self.scaler is not None:
            self.scaler.load_state_dict(snapshot["SCALER_STATE"])
        if "SCHEDULER_STATE" in snapshot and self.scheduler is not None:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        logging.info(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int) -> None:
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCALER_STATE": self.scaler.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(snapshot, self.snapshot_path)
        logging.info(f"Epoch {epoch} | snapshot saved at {self.snapshot_path}")

    # ------------------------------------------------------------------

    @utilities.timer
    def train_fn(self) -> None:
        loop = tqdm(self.train_loader, disable=(self.global_rank != 0))
        self.total_train_loss = 0
        self._nsys_step = getattr(self, "_nsys_step", 0)

        for batch_idx, data_dict in enumerate(self.profiler.iter_dataloader(loop)):
            if (self.mfu_tracker is None and not self._mfu_failed
                    and self.global_rank == 0 and batch_idx == 0):
                self._init_mfu_tracker(data_dict["image"])

            torch.cuda.nvtx.range_push(f"algernon/epoch_{self.epoch}_batch_{batch_idx}")

            ev = None
            if self.global_rank == 0 and self.mfu_tracker is not None:
                ev = self.mfu_tracker.new_step_events()
                ev.fwd_start.record()

            with self.profiler.phase("data_to_gpu"):
                data = data_dict["image"].to(device=self.device, non_blocking=True)
                targets = data_dict["mask"].to(device=self.device, non_blocking=True)

            with self.profiler.phase("forward"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = self.model(data.float())
                    loss = self.loss(predictions, targets)
            if ev is not None:
                ev.fwd_end.record()

            with self.profiler.phase("backward"):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
            if ev is not None:
                ev.bwd_end.record()

            with self.profiler.phase("optimizer"):
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if ev is not None:
                ev.step_end.record()

            torch.cuda.nvtx.range_pop()  # batch

            # Nsys capture control, called between batches (after the optimizer
            # step, once the backward all-reduce has completed) to avoid a
            # CUPTI + NCCL deadlock.
            if self._nsys_step == _NSYS_WARMUP_STEPS:
                torch.cuda.cudart().cudaProfilerStart()
            if self._nsys_step == _NSYS_WARMUP_STEPS + _NSYS_PROFILE_STEPS:
                torch.cuda.cudart().cudaProfilerStop()
            self._nsys_step += 1

            loss_detached = loss.detach()
            self.total_train_loss += loss_detached

            if torch.isnan(loss):
                logging.warning("Loss is NaN — aborting epoch")
                break

            loop.set_postfix(loss=loss_detached.item())

        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]
        self.total_train_loss = self.total_train_loss / len(self.train_loader)

    @utilities.timer
    def validation(self) -> None:
        self.model.eval()
        # Accumulate on the GPU: no .item() inside the loop, a single sync after the reduce.
        total_val_loss = torch.zeros(1, device=self.device)
        total_dsc = torch.zeros(1, device=self.device)
        total_iou = torch.zeros(1, device=self.device)

        torch.cuda.nvtx.range_push(f"algernon/epoch_{self.epoch}_validation")
        loop = tqdm(self.val_loader, disable=(self.global_rank != 0))

        with torch.no_grad():
            for data_dict in loop:
                with self.profiler.phase("val_forward"):
                    data = data_dict["image"].to(device=self.device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = self.model(data.float())

                targets = data_dict["mask"].to(device=self.device, non_blocking=True)
                predictions_fp32 = predictions.float()

                with self.profiler.phase("val_metrics"):
                    loss = self.loss(predictions_fp32, targets)
                    val_metrics = segmentation_validation(predictions_fp32, targets, out_channels=self.out_channels)

                total_val_loss += loss.detach()
                total_dsc += val_metrics["dsc"].detach()
                total_iou += val_metrics["iou"].detach()

        n = len(self.val_loader)
        metrics_tensor = torch.cat([total_val_loss / n, total_dsc / n, total_iou / n])
        dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM)

        if self.global_rank == 0:
            avg_val_loss = metrics_tensor[0].item() / self.world_size
            avg_dsc = metrics_tensor[1].item() / self.world_size
            avg_iou = metrics_tensor[2].item() / self.world_size

            with open(self.csv_metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{self.epoch},{self.total_train_loss:.6f},{avg_val_loss:.6f},{avg_dsc:.6f},{avg_iou:.6f}\n")

            logging.info(f"Val loss: {avg_val_loss:.3f}")
            logging.info(f"DSC: {avg_dsc:.3f} | IoU: {avg_iou:.3f}")

            self.early_stopping(avg_dsc)
            if avg_dsc > self.metric:
                self.metric = avg_dsc
                save_path = self.save_folder / f"best_{self.model_name}.pt"
                torch.save(self.model.module.state_dict(), save_path)
                logging.info(f"New best model saved with DSC {self.metric:.4f}")

        torch.cuda.nvtx.range_pop()  # validation
        # Rank 0 may have just written a checkpoint (slow on NFS); without this
        # barrier the other ranks enter the next epoch and block on the first
        # all-reduce while rank 0 is still writing.
        dist.barrier()
        self.model.train()

    @utilities.timer
    def main(self) -> None:
        self.profiler.register_ddp_model(self.model)
        self.profiler.log_memory_checkpoint("Model loaded (pre-training)")

        train_paths = pd.read_csv(self.train_path)
        val_paths = pd.read_csv(self.val_path)

        if self.global_rank == 0:
            logging.info(f"Device: {self.device}")
            table, _ = utilities.count_parameters(self.model.module)
            logging.info(f"\n{table}")
            shutil.copyfile(self.args.config, self.save_folder / Path(self.args.config).name)

        self.train_loader, self.val_loader = get_loaders(
            train_paths,
            val_paths,
            batch_size=self.batch_size,
            debug=self.args.profile,
            log_dir=str(self.save_folder / "loader_logs") if self.args.profile else None,
        )

        self.loss = DiceCELoss(num_classes=self.out_channels).to(self.device)
        self.early_stopping = EarlyStopping(patience=20, verbose=True, monitor="dsc", op_type="max")

        self.lr = self.init_lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # Linear warm-up (5 epochs, 10 % -> 100 % of the LR) followed by a single
        # cosine decay over the remaining epochs down to 1e-6.
        warmup_epochs = 5
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, self.epochs - warmup_epochs), eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                      milestones=[warmup_epochs])

        if self.snapshot_path.exists() and (self.args.resume_id or self.current_job_id in str(self.snapshot_path)):
            if self.global_rank == 0:
                logging.info(f"Loading snapshot from {self.snapshot_path}...")
            self._load_snapshot()

        dist.barrier()

        for self.epoch in range(self.epochs_run, self.epochs):
            torch.cuda.nvtx.range_push(f"algernon/epoch_{self.epoch}")
            with self.profiler.profile_epoch(self.epoch):
                if self.global_rank == 0:
                    logging.info(f"Training epoch {self.epoch}")

                self.train_loader.sampler.set_epoch(self.epoch)
                self.train_fn()

                if self.global_rank == 0:
                    logging.info(f"Train loss: {self.total_train_loss:.3f}")

                self.validation()

                if self.global_rank == 0 and self.mfu_tracker is not None:
                    self.mfu_tracker.log_epoch(self.mfu_tracker.finalize_epoch(self.epoch))

                if self.global_rank == 0:
                    self._save_snapshot(self.epoch + 1)

                if self.args.earlystop:
                    stop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
                    if self.global_rank == 0 and self.early_stopping.early_stop:
                        stop_flag[0] = 1
                    dist.broadcast(stop_flag, src=0)
                    if stop_flag.item() == 1:
                        if self.global_rank == 0:
                            logging.info("Early stopping triggered — stopping all ranks.")
                        torch.cuda.nvtx.range_pop()  # epoch
                        break

            torch.cuda.nvtx.range_pop()  # epoch

        if self.global_rank == 0 and self.mfu_tracker is not None:
            self.mfu_tracker.log_summary()
            self.mfu_tracker.save_csv(str(self.save_folder / "mfu_metrics.csv"))

        self.profiler.save_final_report()


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(encoding="utf-8", level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(levelname)s - %(message)s")

    torch.backends.cudnn.benchmark = True      # fixed input shapes
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_num_threads(2)

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    try:
        ddp_setup()
        global_rank = int(os.environ["RANK"])
        set_seed(42 + global_rank)

        training = TrainNetwork(args=args, config=config)

        if global_rank == 0:
            logging.info("CPU thread limits: " + ", ".join(
                f"{v}={os.environ.get(v, 'unset')}" for v in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")))
            logging.info(f"PyTorch threads: {torch.get_num_threads()}")

        training.main()

        dist.barrier()
        if global_rank == 0:
            logging.info("Final synchronisation done. Shutting down DDP.")
    except Exception as e:
        logging.exception(e)
    finally:
        destroy_process_group()
