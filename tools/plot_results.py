# -*- coding: utf-8 -*-
"""
Plot training curves (loss, DSC, IoU) from training_metrics.csv.

Usage:
    python tools/plot_results.py --csv_file <run>/training_metrics.csv --output_img <run>/training_plot.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SMOOTH_ALPHA = 0.2   # EMA weight of the newest value


def plot_training_results(csv_path: str, output_img: str) -> None:
    if not os.path.exists(csv_path):
        print(f"[error] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        print("[warning] fewer than 2 epochs — nothing to plot.")
        return

    for col in ("train_loss", "val_loss", "dsc", "iou"):
        df[f"{col}_ema"] = df[col].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(df["epoch"], df["train_loss"], alpha=0.25, color="blue", label="Train loss (raw)")
    ax1.plot(df["epoch"], df["val_loss"], alpha=0.25, color="red", label="Val loss (raw)")
    ax1.plot(df["epoch"], df["train_loss_ema"], linewidth=2.5, color="blue", label="Train loss (EMA)")
    ax1.plot(df["epoch"], df["val_loss_ema"], linewidth=2.5, color="red", label="Val loss (EMA)")
    ax1.set_ylabel("Loss (Dice + CE)")
    ax1.set_title("Loss per epoch")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.plot(df["epoch"], df["dsc"], alpha=0.25, color="green", label="DSC (raw)")
    ax2.plot(df["epoch"], df["iou"], alpha=0.25, color="orange", label="IoU (raw)")
    ax2.plot(df["epoch"], df["dsc_ema"], linewidth=2.5, color="green", label="DSC (EMA)")
    ax2.plot(df["epoch"], df["iou_ema"], linewidth=2.5, color="orange", label="IoU (EMA)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation metrics")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--csv_file", required=True, help="training_metrics.csv")
    parser.add_argument("--output_img", required=True, help="Output PNG")
    args = parser.parse_args()
    plot_training_results(args.csv_file, args.output_img)
