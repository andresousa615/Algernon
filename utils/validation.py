# -*- coding: utf-8 -*-
# Portions derived from mede (https://pypi.org/project/mede/), Copyright 2025
# Lukas Heine & Moritz Rempe, Apache License 2.0 with Commons Clause — see
# LICENSE-mede and NOTICE.
"""
Segmentation metrics computed directly in torch (no torchmetrics dependency).

Both metrics are macro-averaged over the facial classes (1..C-1); the
background (class 0) is ignored as a class but its voxels still count as
false positives of the class predicted on them. A class that is absent from
both the ground truth and the prediction is skipped, not rewarded.

The Dice definition is identical to torchmetrics 0.9 `dice(average="macro",
ignore_index=0)`, which produced the numbers reported for this project.
"""
import torch


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(C, C) confusion matrix with rows = target class, columns = predicted class."""
    pred = logits.argmax(dim=1).flatten()
    tgt = targets.flatten().long()
    return torch.bincount(tgt * num_classes + pred, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def macro_dice_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> tuple:
    """Return (dice, iou) as 0-dim tensors, macro over classes 1..C-1."""
    cm = confusion_matrix(logits, targets, num_classes).float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    tp, fp, fn = tp[1:], fp[1:], fn[1:]          # ignore the background class
    present = (tp + fp + fn) > 0                  # skip classes absent from GT and prediction
    if not present.any():
        zero = torch.zeros((), device=logits.device)
        return zero, zero
    tp, fp, fn = tp[present], fp[present], fn[present]

    dice = (2 * tp / (2 * tp + fp + fn)).mean()
    iou = (tp / (tp + fp + fn)).mean()
    return dice, iou


def segmentation_validation(predictions: torch.Tensor, targets: torch.Tensor, out_channels: int) -> dict:
    """
    Dice and IoU for a batch of multi-class predictions.

    Args:
        predictions: Logits of shape (B, C, D, H, W).
        targets:     Integer label maps of shape (B, 1, D, H, W).
        out_channels: Number of classes C.

    Returns:
        dict with keys "dsc" and "iou" (0-dim tensors on the same device).
    """
    dice, iou = macro_dice_iou(predictions, targets.squeeze(1), out_channels)
    return {"dsc": dice, "iou": iou}
