"""
invnet.py — stacks haar.py + coupling.py into the full invertible
network shape (architecture credit: see pseudoraw/ml/__init__.py).

InvISPLiteNet.forward(jpeg_rgb) and .inverse(z) are EXACT inverses of
each other by construction -- true regardless of whether the internal
subnets are trained, random, or garbage, because every piece
(forward_haar/inverse_haar, each AffineCouplingBlock) is individually
invertible in closed form. That's the property this scaffold exists to
prove (see tests/test_invnet_invertibility.py); it is not yet a property
that forward() produces a *useful* pseudo-RAW, which requires training.

Shape: an (H, W, 3) image enters, forward_haar expands it to
(H/2, W/2, 12), n_blocks alternating-swap AffineCouplingBlocks run on
that, and the result is both the "latent" representation and, after
inverse_haar, invertible back to the exact original (H, W, 3) image.

This network is intentionally NOT wired into pipeline.py yet -- it is
groundwork for replacing deblock.py/chroma.py's classical heuristics,
not a drop-in replacement today. See README for the training plan.
"""

from __future__ import annotations

import numpy as np

from .haar import forward_haar, inverse_haar
from .coupling import AffineCouplingBlock


class InvISPLiteNet:
    def __init__(self, in_channels: int = 3, n_blocks: int = 4, seed: int = 0):
        self.in_channels = in_channels
        self.haar_channels = in_channels * 4
        self.blocks = [
            AffineCouplingBlock(self.haar_channels, seed=seed + i, swap=(i % 2 == 1))
            for i in range(n_blocks)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (H, W, in_channels) -> z: (H/2, W/2, in_channels*4)"""
        z = forward_haar(x.astype(np.float64))
        for block in self.blocks:
            z = block.forward(z)
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Exact inverse of forward(). z: (H/2, W/2, in_channels*4) -> (H, W, in_channels)"""
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return inverse_haar(x, channels=self.in_channels)
