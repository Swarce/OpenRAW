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

Two real signals this module now actually uses, both available from
decode.py but previously decoded and thrown away unused:

- is_chroma_subsampled: a 4:4:4 JPEG was never chroma-subsampled in the
  first place, so there is no upsampling softness to correct. Running
  this correction on 4:4:4 source would be pure risk (altering chroma
  that didn't need it) for zero possible benefit -- so it's skipped
  entirely rather than applied "just in case."
- chroma_quality_estimate: derived from the ACTUAL quant table the
  chroma channels were encoded with (not assumed to be table index 1 --
  decode.py reads which table the chroma components really reference),
  which can differ meaningfully from the luma quality estimate. Chroma
  is frequently quantized more aggressively than luma even at the same
  nominal JPEG quality setting, so scaling refinement strength off
  chroma's own quality figure is more grounded than a fixed constant.
"""

from __future__ import annotations

import numpy as np
import cv2


def refine_chroma(
    rgb: np.ndarray,
    strength: float = 0.6,
    is_chroma_subsampled: bool = True,
    chroma_quality_estimate: float | None = None,
) -> np.ndarray:
    """
    rgb: float32 HxWx3 in [0, 1].
    strength: 0..1 base strength for the guided re-sharpening.
    is_chroma_subsampled: from DecodedJpeg.is_chroma_subsampled. False
        (a 4:4:4 source) short-circuits to a no-op -- see module docstring.
    chroma_quality_estimate: from DecodedJpeg.chroma_quality_estimate.
        When given, scales `strength` so a heavily-quantized chroma
        channel (low quality figure) gets more correction and a lightly
        quantized one gets less, instead of always applying the same
        fixed amount regardless of how aggressive the original encoder
        actually was on this specific JPEG's chroma.

    Note: cv2's float32 color-conversion path expects [0, 1]-range input
    for RGB<->YCrCb (unlike its uint8 path, which is [0, 255]) -- mixing
    that up silently desaturates everything, so we stay in [0, 1] the
    whole way through here.
    """
    if not is_chroma_subsampled:
        return rgb.copy()

    if chroma_quality_estimate is not None:
        # quality ~95+ -> scale toward 0.4x; quality <40 -> scale toward 1.6x.
        # Multiplicative, not a replacement for `strength`, so a caller's
        # explicit strength choice still matters -- this adjusts it
        # rather than overriding it.
        q = float(np.clip(chroma_quality_estimate, 1.0, 100.0))
        scale = float(np.clip(1.6 - (q / 100.0) * 1.2, 0.4, 1.6))
        strength = float(np.clip(strength * scale, 0.0, 1.5))
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
