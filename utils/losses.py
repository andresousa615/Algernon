# -*- coding: utf-8 -*-
# Portions derived from mede (https://pypi.org/project/mede/), Copyright 2025
# Lukas Heine & Moritz Rempe, Apache License 2.0 with Commons Clause — see
# LICENSE-mede and NOTICE.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Multi-class soft Dice loss combined with categorical cross-entropy:

        loss = (1 - mean_c Dice_c) + 0.1 * CE

    The Dice term is averaged over all classes (background included), so the
    small facial structures weigh as much as the background. The CE term adds
    a per-voxel gradient that stabilises the early epochs.

    Inputs are raw logits of shape (B, C, D, H, W); targets are integer label
    maps of shape (B, 1, D, H, W).
    """

    def __init__(self, num_classes: int = 5, ce_weight: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        targets = targets.squeeze(1).long()
        ce = self.criterion(inputs, targets)

        # Softmax in FP32 for numerical stability under AMP.
        probs = torch.softmax(inputs, dim=1).float()
        onehot = F.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)  # sum over batch and space, keep the class axis
        intersection = (probs * onehot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + onehot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + smooth) / (denominator + smooth)

        return (1.0 - dice_per_class.mean()) + self.ce_weight * ce


# Backwards-compatible alias (the training script historically imported DiceLoss).
DiceLoss = DiceCELoss
