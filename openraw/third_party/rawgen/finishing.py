# Adapted from RawGen, training/generate_variants_and_anchors.py
# (ISPAugmenter._generate_random_isp_params and ISPAugmenter._apply_real_isp_pipeline),
# https://github.com/DY112/RawGen at commit 39f959b4a4b0e8ba256465d53ce4b8a397d997ba.
# Copyright (c) 2026 the RawGen authors. Licensed under CC BY-NC-SA 4.0 (see LICENSE in
# this folder). Paper: Kim, Lee, Punnappurath, Afifi, Han, Levinshtein, Brown,
# "RawGen: Learning Camera Raw Image Generation", ECCV 2026, arXiv:2604.00093.
#
# Changes made for OpenRAW (as CC BY-NC-SA 4.0 asks us to indicate):
#   - ported from NumPy/colour-science to PyTorch, batched, one parameter set per image;
#   - applied to an already rendered sRGB image (OpenRAW's training JPEG) instead of running
#     RawGen's mini ISP from the DNG: OpenRAW's training pairs keep the sensor mosaic but
#     not the DNG's colour metadata, so the ISP stages are replaced by the existing render;
#   - the white-balance multipliers, which RawGen applies to AsShotNeutral before
#     demosaicing, are applied in linear sRGB here, and are off unless asked for;
#   - parameter ranges and the formulas themselves are RawGen's, unchanged.
"""RawGen's random photo-finishing, for train.py --finish_aug --finish_style rawgen.

RawGen trains on several differently finished sRGB versions of each raw, all
mapped to one target. Each version gets, after a standard ISP:

  white balance   R and B multipliers ~ U(0.7, 1.3)          (optional here)
  tone mapping    (1 + beta) E^gamma / (beta + E^gamma) in linear light,
                  beta ~ N(0.6, 0.1^2) clipped to [0.1, 2], gamma ~ N(0.9, 0.1^2) clipped to [0.5, 1.5]
  contrast        (x - 0.5) * c + 0.5 in sRGB,   c ~ U(0.8, 1.2)
  brightness      x + b in sRGB                  (off by default, as in RawGen)
"""

from __future__ import annotations

import torch

# RawGen's defaults (ISPAugmenter.__init__)
COLOR_GRADING_RANGE = (0.7, 1.3, 0.7, 1.3)  # r_min, r_max, b_min, b_max
CONTRAST_RANGE = (0.8, 1.2)
TONE_BETA = (0.6, 0.1)    # mean, std
TONE_GAMMA = (0.9, 0.1)


def _srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = x.clamp(min=0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def sample_params(n: int, device, white_balance: bool = False, brightness_range=None, generator=None) -> dict:
    """RawGen's _generate_random_isp_params, for a batch of n images."""
    def ru(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=device, generator=generator)

    def rn(mean, std):
        return mean + std * torch.randn(n, device=device, generator=generator)

    r_min, r_max, b_min, b_max = COLOR_GRADING_RANGE
    p = {"beta": rn(*TONE_BETA).clamp(0.1, 2.0),
         "gamma": rn(*TONE_GAMMA).clamp(0.5, 1.5),
         "contrast": ru(*CONTRAST_RANGE)}
    if white_balance:
        p["wb"] = torch.stack([ru(r_min, r_max), torch.ones(n, device=device), ru(b_min, b_max)], 1)
    if brightness_range is not None:
        p["brightness"] = ru(*brightness_range)
    return p


def apply(srgb: torch.Tensor, p: dict) -> torch.Tensor:
    """RawGen's photo-finishing (after its ISP) on display sRGB (N,3,H,W) in [0, 1]."""
    v = lambda t: t.view(-1, 1, 1, 1)  # noqa: E731
    lin = _srgb_to_linear(srgb.float().clamp(0, 1))
    if "wb" in p:
        lin = lin * p["wb"].view(-1, 3, 1, 1)
    e = lin.clamp(0, 1) ** v(p["gamma"])
    tone = (1 + v(p["beta"])) * e / (v(p["beta"]) + e).clamp(min=1e-8)
    out = _linear_to_srgb(tone).clamp(0, 1)
    out = (out - 0.5) * v(p["contrast"]) + 0.5
    if "brightness" in p:
        out = out + v(p["brightness"])
    return out.clamp(0, 1)
