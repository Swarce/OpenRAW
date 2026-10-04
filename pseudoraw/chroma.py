"""
chroma.py — sharpen the chroma planes back toward luma edges.

Important limitation, stated plainly: Pillow decodes straight to full-
resolution RGB, so by the time we see the image the original 4:2:0 (or
4:2:2) chroma subsampling has *already* been upsampled by libjpeg's
internal decoder, using its own fast upsampling filter. We never see the
true half-resolution Cb/Cr planes, so we can't do a textbook joint-
bilateral upsample from genuine low-res chroma — that data is gone by
the time it reaches us.

What we *can* do: treat the already-upsampled chroma as "too soft,
possibly bled across edges" and re-sharpen it using the luma channel
(which is full resolution and far more reliable) as a guide. This recovers
some of the chroma edge definition that naive upsampling throws away,
without pretending to reconstruct data we don't have.

TODO: decode via a path that exposes the true subsampled planes (e.g. a
jpeglib/libjpeg-turbo binding with raw component output) and do a real
joint-bilateral upsample from half-res chroma. That is a strictly better
version of this module with the same interface.
"""

from __future__ import annotations

import numpy as np
import cv2


def refine_chroma(rgb: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    rgb: float32 HxWx3 in [0, 1].
    strength: 0..1, how much guided re-sharpening to apply to chroma.

    Note: cv2's float32 color-conversion path expects [0, 1]-range input
    for RGB<->YCrCb (unlike its uint8 path, which is [0, 255]) -- mixing
    that up silently desaturates everything, so we stay in [0, 1] the
    whole way through here.
    """
    ycrcb = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    guide = y  # already [0, 1]

    def guided_sharpen(channel: np.ndarray) -> np.ndarray:
        # Edge-aware smoothing of the chroma plane guided by luma...
        smooth = cv2.ximgproc.guidedFilter(guide=guide, src=channel, radius=4, eps=1e-3)
        # ...then push back the high-frequency residual, i.e. unsharp
        # masking but only along edges the guide actually supports.
        detail = channel - smooth
        sharpened = channel + detail * strength
        return sharpened.astype(np.float32)

    cr_refined = guided_sharpen(cr)
    cb_refined = guided_sharpen(cb)

    merged = cv2.merge([y, cr_refined, cb_refined])
    rgb_out = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    return np.clip(rgb_out, 0.0, 1.0).astype(np.float32)
