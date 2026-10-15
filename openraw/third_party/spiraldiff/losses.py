# Adapted from SpiralDiff, models/gaussian_diffusion.py (logl1_loss),
# https://github.com/Chuancy-TJU/SpiralDiff at commit 30819932c998b4571d82e4e86fdd474cf23c83a7.
# Copyright (c) 2026 Chuancy-TJU. MIT License (see LICENSE in this folder).
# Paper: Yue, Xie, Cao, Wu, Zhang, Zhao, Yang, "SpiralDiff: Spiral Diffusion with LoRA for
# RGB-to-RAW Conversion Across Cameras", CVPR 2026, arXiv:2603.14885.
# SpiralDiff's eps follows HDR-CNN (Eilertsen, Kronander, Denes, Mantiuk, Unger, "HDR image
# reconstruction from a single exposure using deep CNNs", ACM TOG 36(6), 2017;
# https://github.com/gabrieleilertsen/hdrcnn).
#
# Changes made for OpenRAW: inputs are already in [0, 1] (SpiralDiff's are in [-1, 1] and are
# mapped to [0, 1] first); the mean is over the whole batch (SpiralDiff's mean_flat keeps one
# value per image); the inputs are clamped at 0 so a slightly negative prediction can't reach
# log of a negative number.
"""Log-domain L1 loss on raw values, as SpiralDiff trains with (train.py --logl1_loss).

An L1 loss on linear raw values barely notices errors in the shadows, where
the values are small; SpiralDiff's observation is that reconstruction
difficulty depends on intensity. Comparing log(x + eps) weights every
brightness level by its relative error instead.
"""

from __future__ import annotations

import torch

EPS = 1.0 / 255.0  # SpiralDiff's default, after HDR-CNN


def logl1_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """mean |log(pred + eps) - log(gt + eps)| for linear values in [0, 1]."""
    return torch.mean(torch.abs(torch.log(pred.clamp(min=0) + eps) - torch.log(gt.clamp(min=0) + eps)))
