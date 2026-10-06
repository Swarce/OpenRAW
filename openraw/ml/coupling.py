"""
coupling.py — affine coupling block, the learnable (eventually) half of
the invertible network. Architecture per RealNVP / Invertible-Image-
Rescaling / InvISP (see package docstring for full credit chain).

The trick that makes this exactly invertible regardless of what the
internal subnets compute: split the channels into two halves, leave one
half untouched, and transform the other half using only values computed
*from the untouched half*. That one-directional dependency is what makes
inversion possible in closed form -- you already have everything needed
(the untouched half) to undo the transform on the other half.

    forward:  y1 = x1
              y2 = x2 * exp(s(x1)) + t(x1)

    inverse:  x1 = y1
              x2 = (y2 - t(y1)) * exp(-s(y1))

s() and t() ("scale" and "translate") can be arbitrarily complex neural
nets -- invertibility holds no matter what they compute, which is exactly
why this architecture doesn't need to be trained to be verified correct
(see tests/test_invnet_invertibility.py).

IMPORTANT, stated plainly: the subnet here (`_ChannelMLP`) is a small
fixed/randomly-initialized channel-wise (1x1-conv-equivalent) MLP. It has
no spatial context and is NOT trained. It exists ONLY to give forward/
inverse something nontrivial to compute while we verify the plumbing.
This is the next real milestone: replace `_ChannelMLP` with a trained
spatial CNN (PyTorch, matching InvISP's own subnet design) once there's
paired RAW/JPEG training data and a GPU to train on.
"""

from __future__ import annotations

import numpy as np


class _ChannelMLP:
    """
    Fixed-weight, 2-layer channel-wise MLP: operates independently per
    pixel (no spatial mixing yet -- that comes from the Haar transform's
    local-block structure plus, eventually, real conv subnets). Outputs
    are split into (scale, translate); scale is passed through a bounded
    tanh*clamp before being exponentiated, which keeps the coupling
    layer numerically well-conditioned even with random, untrained
    weights -- an unbounded scale would make inversion numerically
    unstable (exp of a large number), which is a real failure mode
    worth guarding against even in a scaffold meant to be replaced.
    """

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 16, seed: int = 0, scale_clamp: float = 1.5):
        rng = np.random.default_rng(seed)
        # Small init, like a real network would use, so early-stage
        # (or in this case permanently-untrained) behavior stays mild.
        self.w1 = rng.normal(0, 0.15, size=(in_ch, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 0.15, size=(hidden, out_ch * 2))
        self.b2 = np.zeros(out_ch * 2)
        self.out_ch = out_ch
        self.scale_clamp = scale_clamp

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x: (H, W, in_ch) -> (scale, translate), each (H, W, out_ch)."""
        h1 = np.tanh(x @ self.w1 + self.b1)
        out = h1 @ self.w2 + self.b2
        raw_scale, translate = out[..., : self.out_ch], out[..., self.out_ch :]
        scale = np.tanh(raw_scale) * self.scale_clamp
        return scale, translate


class AffineCouplingBlock:
    def __init__(self, channels: int, seed: int = 0, swap: bool = False):
        if channels % 2 != 0:
            raise ValueError("AffineCouplingBlock needs an even channel count")
        self.c_half = channels // 2
        self.swap = swap  # alternate which half is transformed, across blocks
        self.subnet = _ChannelMLP(self.c_half, self.c_half, seed=seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[..., : self.c_half], x[..., self.c_half :]
        if self.swap:
            x1, x2 = x2, x1

        scale, translate = self.subnet(x1)
        y2 = x2 * np.exp(scale) + translate
        y1 = x1

        if self.swap:
            y1, y2 = y2, y1
        return np.concatenate([y1, y2], axis=-1)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y1, y2 = y[..., : self.c_half], y[..., self.c_half :]
        if self.swap:
            y1, y2 = y2, y1

        scale, translate = self.subnet(y1)
        x2 = (y2 - translate) * np.exp(-scale)
        x1 = y1

        if self.swap:
            x1, x2 = x2, x1
        return np.concatenate([x1, x2], axis=-1)
