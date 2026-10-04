"""
bitdepth.py — expand an 8-bit-sourced linear image into a clean 16-bit
container without just re-displaying the original posterization at a
bigger bit depth.

The problem this solves: converting an 8-bit sRGB value to linear light
stretches shadow values across a much wider numeric range (the sRGB EOTF
has a steep slope near black), so the ~256 original quantization steps
become *visible, large steps* in linear space — exactly where you'd want
headroom to lift shadows. Simply storing that in 16 bits does not add
information back; it just gives banding more room to be ugly.

Two standard, honest mitigations, both of which only redistribute
existing error rather than inventing detail:

1. Triangular dithering, amplitude-matched per input level to the local
   step size of the sRGB->linear curve (via a precomputed 256-entry LUT
   of local slopes), so banding becomes noise instead of hard edges —
   classic dithering, nothing ML about it.
2. Gradient-gated debanding smoothing: a mild edge-aware blur applied
   only in genuinely flat, low-gradient regions (skies, skin, shadows),
   masked off everywhere there's real detail, so we're not just
   blurring the whole image to hide banding.
"""

from __future__ import annotations

import numpy as np
import cv2

from .tonecurve import srgb_to_linear


def _local_slope_lut() -> np.ndarray:
    """256-entry LUT: d(linear)/d(8bit value) at each of the 256 sRGB levels."""
    levels = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    linear = srgb_to_linear(levels.astype(np.float32)).astype(np.float64)
    slope = np.diff(linear)  # length 256, slope[i] = step size leaving level i
    return slope.astype(np.float32)


_SLOPE_LUT = _local_slope_lut()


def _dither_amplitude(srgb_u8: np.ndarray) -> np.ndarray:
    """
    srgb_u8: uint8 HxW (or HxWx1), the ORIGINAL 8-bit sRGB values (before
    linearization) that this pipeline started from — used only to look up
    local step size, not reused as output.
    """
    idx = np.clip(srgb_u8, 0, 255).astype(np.int64)
    return _SLOPE_LUT[idx]


def expand_to_16bit(
    linear_rgb: np.ndarray,
    source_srgb_u8: np.ndarray,
    dither: bool = True,
    deband: bool = True,
    seed: int | None = 0,
) -> np.ndarray:
    """
    linear_rgb: float32 HxWx3, scene-linear, range [0, 1] (post tonecurve).
    source_srgb_u8: uint8 HxWx3, the original 8-bit sRGB pixels this was
        derived from — used to size the dither/deband amplitude per pixel,
        not written to the output.
    Returns: uint16 HxWx3, range [0, 65535].
    """
    out = linear_rgb.copy()
    rng = np.random.default_rng(seed)

    if dither:
        amp = _dither_amplitude(source_srgb_u8)  # HxWx3, local linear step size
        # Triangular dither: sum of two uniforms, centered at 0, spread
        # +/-1 step — standard shape for breaking quantization banding
        # without adding a DC bias.
        noise = (rng.random(out.shape, dtype=np.float32) - rng.random(out.shape, dtype=np.float32))
        out = out + noise * amp

    if deband:
        gray = cv2.cvtColor((np.clip(linear_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        grad = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        grad_mag = np.abs(grad)
        flat_mask = np.clip(1.0 - grad_mag / 12.0, 0.0, 1.0)  # 1 = flat, 0 = edge/detail
        flat_mask = cv2.GaussianBlur(flat_mask, (9, 9), 0)[..., None]

        smoothed = np.empty_like(out)
        for c in range(3):
            smoothed[..., c] = cv2.bilateralFilter(
                out[..., c].astype(np.float32), d=9, sigmaColor=0.02, sigmaSpace=9
            )
        out = out * (1.0 - flat_mask * 0.85) + smoothed * (flat_mask * 0.85)

    out = np.clip(out, 0.0, 1.0)
    return (out * 65535.0 + 0.5).astype(np.uint16)
