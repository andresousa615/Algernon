# -*- coding: utf-8 -*-
# Portions derived from mede (https://pypi.org/project/mede/), Copyright 2025
# Lukas Heine & Moritz Rempe, Apache License 2.0 with Commons Clause — see
# LICENSE-mede and NOTICE.
"""
Segmentation network used by Algernon: a MedNeXt-S (kernel 3, expansion 2)
with five output classes (background + ears, mouth, nose, eyes).
"""
import torch
import torch.nn as nn

from models.mednext.MedNextV1 import MedNeXt


class MedNeXtSeg(nn.Module):
    """
    Thin wrapper around MedNeXt with the hyper-parameters used in Algernon.

    Args:
        in_channels:  Input channels (1 for grayscale MRI).
        out_channels: Number of segmentation classes (5: background + 4 facial regions).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 5) -> None:
        super().__init__()

        self.model = MedNeXt(
            in_channels=in_channels,
            n_channels=32,
            n_classes=out_channels,
            exp_r=2,
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            checkpoint_style=None,
        )

        # MedNeXt registers a dummy parameter for gradient checkpointing. With
        # checkpointing disabled it never receives a gradient, which makes DDP
        # wait for it forever — so it is frozen here.
        self.model.dummy_tensor.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Registry used by train.py / inference.py to instantiate the network from the config.
NETWORKS = {"mednext": MedNeXtSeg}


def build_network(name: str, out_channels: int) -> nn.Module:
    try:
        cls = NETWORKS[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(NETWORKS)}")
    return cls(out_channels=out_channels)
