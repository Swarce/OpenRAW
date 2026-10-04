"""
colormatrix.py — decide what "camera native color space" means for data
that never came from a camera sensor, and tell the DNG reader the truth
about it.

Here's the honest version of what step 6 of the pipeline actually is:
there is no real sensor to invert back to, and any invented matrix meant
to simulate "what a sensor's native gamut might have looked like" is pure
fabrication with nothing grounding it. So instead of fabricating a fake
sensor profile, this module does the defensible thing: it declares the
"virtual sensor" native space to simply BE the linear-light sRGB space our
reconstructed pixels are already, honestly, in. We then hand DNG readers
the exact, correct XYZ<->sRGB matrix for that space (under D65, matching
CalibrationIlluminant1), so color rendering is accurate rather than
guessed, and no invented per-pixel color transform is required at all.

A separate, clearly-labeled EXPERIMENTAL option lets you apply a mild
gamut-widening matrix to pixel data, approximating the fact that many real
sensors have a native gamut somewhat wider than sRGB (more grading
headroom in saturated colors). It is off by default and should be treated
as a creative option, not a correctness improvement.
"""

from __future__ import annotations

import numpy as np

# Standard sRGB (D65) <-> CIE XYZ (D65) matrices. These are exact,
# standardized constants -- not estimates.
SRGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

XYZ_D65_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ_D65)

# A mild, hand-picked gamut-widening matrix (pulls saturated colors
# slightly outward). EXPERIMENTAL -- not derived from any real sensor.
_EXPERIMENTAL_WIDEN = np.array(
    [
        [1.08, -0.05, -0.03],
        [-0.04, 1.08, -0.04],
        [-0.03, -0.05, 1.08],
    ],
    dtype=np.float64,
)


def dng_color_matrix1() -> np.ndarray:
    """
    The matrix to write into the DNG ColorMatrix1 tag (XYZ -> camera
    native), paired with CalibrationIlluminant1 = D65. Since our "native"
    space is honestly just linear sRGB, this is the exact standard
    XYZ->sRGB matrix, not an estimate.
    """
    return XYZ_D65_TO_SRGB.copy()


def apply_experimental_gamut_widen(linear_rgb: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    linear_rgb: float32 HxWx3, scene-linear sRGB-primaries data.
    strength: 0 disables (default). 0..1 blends toward the experimental
        widen matrix. Treat this as a look, not a reconstruction step --
        label it as such anywhere it's exposed in a UI/CLI.
    """
    if strength <= 0.0:
        return linear_rgb

    identity = np.eye(3)
    m = identity * (1.0 - strength) + _EXPERIMENTAL_WIDEN * strength
    h, w = linear_rgb.shape[:2]
    flat = linear_rgb.reshape(-1, 3).astype(np.float64)
    out = flat @ m.T
    return np.clip(out, 0.0, 1.0).reshape(h, w, 3).astype(np.float32)
