"""
tonecurve.py — undo the display-referred tone response and get back to
scene-linear light, which is what a RAW converter expects to receive and
re-grade from scratch.

Two things happen here, and they are NOT equally trustworthy:

1. sRGB EOTF inversion (srgb_to_linear). This is exact, well-defined math
   — if the JPEG is tagged/assumed sRGB, this step is a correct inverse,
   not a guess. The only loss already happened upstream, in JPEG's 8-bit
   quantization of the gamma-encoded values; this step doesn't add error,
   it just stops *hiding* the error that's already there (which is why
   bitdepth.py's dithering step matters immediately after this one).

2. Generic contrast-curve removal (remove_generic_s_curve). Most cameras
   apply a further S-shaped "picture style" tone curve on top of sRGB
   gamma (lifted shadows, rolled-off highlights, added contrast) before
   JPEG encoding, and that curve is camera/profile-specific and NOT
   recoverable from the JPEG alone. This function applies a generic,
   conservative inverse of a *typical* S-curve. It is a guess, it is
   disabled by default, and it should be treated as a visual nicety, not
   a correctness step. The honest long-term fix is a learned, per-scene
   or per-camera-profile curve estimator — not this.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Exact inverse sRGB EOTF. rgb: float32 in [0, 1]."""
    a = 0.055
    low = rgb <= 0.04045
    linear = np.where(
        low,
        rgb / 12.92,
        ((rgb + a) / (1 + a)) ** 2.4,
    )
    return linear.astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Forward sRGB EOTF, used for generating quick preview JPEGs/PNGs."""
    a = 0.055
    low = linear <= 0.0031308
    srgb = np.where(
        low,
        linear * 12.92,
        (1 + a) * np.power(np.clip(linear, 0, None), 1 / 2.4) - a,
    )
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def remove_generic_s_curve(linear: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Conservative, generic inverse of a typical camera contrast S-curve.
    strength: 0 disables this entirely (default — use explicit opt-in).
              Reasonable range if enabled: 0.1 - 0.4.

    This is deliberately a mild, smooth, monotonic correction — not an
    attempt to match any particular camera's actual curve, which we have
    no way of knowing from a JPEG.
    """
    if strength <= 0.0:
        return linear

    x = np.clip(linear, 0.0, 1.0)
    # Mild inverse-sigmoid-ish lift of shadows / pull-down of highlights,
    # symmetric around 0.5, magnitude controlled by `strength`.
    k = strength * 2.0
    centered = x - 0.5
    corrected = centered + k * centered * (0.25 - centered**2)
    return np.clip(corrected + 0.5, 0.0, 1.0).astype(np.float32)
