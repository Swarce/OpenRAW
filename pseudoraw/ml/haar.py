"""
haar.py — invertible 2x spatial downsampling via the Haar wavelet basis.

This is the standard invertible-downsampling building block used by
Invertible-Image-Rescaling and, through it, InvISP (see package docstring
for full credit): it takes an HxWxC image and returns an (H/2)x(W/2)x4C
image with ZERO information loss -- the transform is an orthogonal,
involutory (self-inverse) linear map, not a learned or lossy operation.
That matters here specifically because every other downsampling choice
(strided conv, pooling, naive subsampling) throws information away, which
would be exactly the wrong building block for a tool whose entire premise
is "don't lose more data than JPEG already lost."

The four output channels-per-input-channel are the classic LL/HL/LH/HH
subbands: LL is the local average (behaves like a blurred half-res
image), the other three capture horizontal/vertical/diagonal local
differences. Because the 4x4 transform matrix below is both orthogonal
and symmetric, it is its own inverse -- forward_haar and inverse_haar
apply literally the same matrix.
"""

from __future__ import annotations

import numpy as np

# Orthonormal, symmetric (hence self-inverse) Haar transform matrix.
# Rows: LL, HL, LH, HH. Verified orthonormal + symmetric in
# tests/test_invnet_invertibility.py.
_H = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float64,
) / 2.0


def forward_haar(x: np.ndarray) -> np.ndarray:
    """
    x: float64/float32 array, shape (H, W, C), H and W even.
    returns: shape (H/2, W/2, 4*C) -- channels ordered as
        [LL_c0..LL_cN, HL_c0..HL_cN, LH_c0..LH_cN, HH_c0..HH_cN]
    """
    h, w, c = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"forward_haar needs even H, W; got {h}x{w}")

    x1 = x[0::2, 0::2, :]
    x2 = x[1::2, 0::2, :]
    x3 = x[0::2, 1::2, :]
    x4 = x[1::2, 1::2, :]
    stacked = np.stack([x1, x2, x3, x4], axis=-1)  # (H/2, W/2, C, 4)

    # Apply the 4x4 Haar matrix along the last axis, per input channel.
    transformed = stacked @ _H.T  # (H/2, W/2, C, 4) -> subbands on last axis

    # Reorder to (H/2, W/2, 4*C) grouped by subband (LL block, HL block, ...)
    out = np.transpose(transformed, (0, 1, 3, 2)).reshape(h // 2, w // 2, 4 * c)
    return out


def inverse_haar(y: np.ndarray, channels: int) -> np.ndarray:
    """
    Exact inverse of forward_haar.
    y: shape (H/2, W/2, 4*channels)
    channels: the original C (number of channels before forward_haar)
    returns: shape (H, W, channels)
    """
    hh, ww, c4 = y.shape
    if c4 != 4 * channels:
        raise ValueError(f"expected last dim {4*channels}, got {c4}")

    subbands = y.reshape(hh, ww, 4, channels)
    subbands = np.transpose(subbands, (0, 1, 3, 2))  # (H/2, W/2, C, 4)

    # _H is its own inverse (orthonormal + symmetric), so apply it again.
    original = subbands @ _H.T  # (H/2, W/2, C, 4) -> x1,x2,x3,x4 on last axis

    h, w = hh * 2, ww * 2
    out = np.empty((h, w, channels), dtype=y.dtype)
    out[0::2, 0::2, :] = original[..., 0]
    out[1::2, 0::2, :] = original[..., 1]
    out[0::2, 1::2, :] = original[..., 2]
    out[1::2, 1::2, :] = original[..., 3]
    return out
