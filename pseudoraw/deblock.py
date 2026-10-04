"""
deblock.py — reduce 8x8 blocking and ringing artifacts before anything
downstream (chroma reconstruction, tone inversion) trusts the pixel values.

This is a classical placeholder, not the long-term answer. A proper
implementation needs the raw DCT coefficients (so it can dequantize and
deblock in frequency space, which is where the artifacts actually live);
Pillow only hands us already-IDCT'd pixels. For the POC we do the best
achievable thing in pixel space: an edge-aware guided filter whose
strength is driven by the real quant-table-derived quality estimate from
decode.py, plus a targeted pass that specifically softens the 8x8 grid
lines where blocking artifacts concentrate.

TODO (tracked as the first real improvement over this POC): swap this
module for a small learned deblocking net (ARCNN-style) trained on
quality-labeled JPEG/source pairs, or move to a JPEG decoder that exposes
DCT coefficients (e.g. a jpeglib binding) so deblocking can happen before
the IDCT instead of after it.
"""

from __future__ import annotations

import numpy as np
import cv2


def _grid_mask(h: int, w: int, period: int = 8) -> np.ndarray:
    """1.0 near predicted 8x8 block boundaries, 0.0 in block interiors."""
    mask = np.zeros((h, w), dtype=np.float32)
    mask[::period, :] = 1.0
    mask[:, ::period] = 1.0
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return np.clip(mask, 0.0, 1.0)


def deblock(rgb: np.ndarray, quality_estimate: float) -> np.ndarray:
    """
    rgb: float32 HxWx3 in [0, 1].
    quality_estimate: 1-100, from decode.estimate_quality. Lower quality
        (more aggressive original quantization) gets stronger smoothing.
    """
    h, w = rgb.shape[:2]

    # Strength curve: quality 95+ -> almost no-op; quality <40 -> strong.
    q = float(np.clip(quality_estimate, 1.0, 100.0))
    strength = np.clip((85.0 - q) / 70.0, 0.0, 1.0)  # 0..1
    if strength <= 0.01:
        return rgb.copy()

    radius = int(2 + round(6 * strength))
    eps = (0.01 + 0.09 * strength) ** 2

    guide = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    guide = guide.astype(np.float32) / 255.0

    smoothed = np.empty_like(rgb)
    for c in range(3):
        smoothed[..., c] = cv2.ximgproc.guidedFilter(
            guide=guide, src=rgb[..., c], radius=radius, eps=eps
        )

    # Blend globally by strength, then push extra smoothing specifically
    # onto the predicted block-grid lines where ringing/blocking live.
    blended = rgb * (1.0 - strength) + smoothed * strength

    grid = _grid_mask(h, w)[..., None]
    grid_strength = np.clip(strength * 1.5, 0.0, 1.0)
    result = blended * (1.0 - grid * grid_strength) + smoothed * (grid * grid_strength)

    return np.clip(result, 0.0, 1.0).astype(np.float32)
