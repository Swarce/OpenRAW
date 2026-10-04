"""
dng_writer.py — package 16-bit linear RGB data as a minimal, valid
Linear DNG (DNG's PhotometricInterpretation=34892 mode, used for data
that is already demosaiced/not a Bayer mosaic — exactly our case).

This writes real DNG tags (DNGVersion, ColorMatrix1, CalibrationIlluminant1,
WhiteLevel, BlackLevel, AsShotNeutral, etc.) via tifffile's extratags
mechanism, not a renamed TIFF. Raw-capable editors (Lightroom, Capture
One, darktable, RawTherapee) read Linear DNG as a real raw file: you get
their white balance / tone curve / highlight recovery tools operating on
our reconstructed linear data, instead of on a baked JPEG.

We deliberately keep this file focused on "write correct, minimal,
honest tags" — no attempt to fake tags we have no grounds for (e.g. no
invented NoiseProfile, no fake CFA pattern, no fabricated lens/camera
EXIF beyond clearly labeling the file as pseudoraw-generated).
"""

from __future__ import annotations

import numpy as np
import tifffile

from .colormatrix import dng_color_matrix1

DNG_PHOTOMETRIC_LINEAR_RAW = 34892
CALIBRATION_ILLUMINANT_D65 = 21


def _float_matrix_to_srational(matrix: np.ndarray, denom: int = 1_000_000) -> list[int]:
    """Flat [num0, den0, num1, den1, ...] -- tifffile's extratags packer
    wants rational/srational values as a flat sequence of (count*2) ints,
    not a list of (num, den) tuples."""
    flat = matrix.flatten()
    out: list[int] = []
    for v in flat:
        out.extend([int(round(v * denom)), denom])
    return out


def _rational_flat(value: float, denom: int = 1_000_000) -> list[int]:
    return [int(round(value * denom)), denom]


def write_linear_dng(
    path: str,
    rgb16: np.ndarray,
    source_jpeg_path: str = "",
    pipeline_version: str = "0.0.1-poc",
    compress: bool = True,
) -> None:
    """
    rgb16: uint16 HxWx3, range [0, 65535], scene-linear.
    compress: if True (default), write with lossless Adobe Deflate
        (DNG Compression tag = 8) + horizontal-differencing predictor.
        This is a real, lossless, widely-supported DNG compression mode
        -- Lightroom/ACR/darktable/RawTherapee/libraw all read it -- not
        a workaround. Without it, a single 16-bit HxWx3 linear image is
        written completely uncompressed: a 24MP photo is ~140MB on disk
        for no reason other than this flag being off. Set False only if
        you've hit a specific reader that chokes on compressed DNG (rare)
        and need to confirm compression is the cause.
    """
    if rgb16.dtype != np.uint16:
        raise ValueError("rgb16 must be uint16")
    if rgb16.ndim != 3 or rgb16.shape[2] != 3:
        raise ValueError("rgb16 must be HxWx3")

    color_matrix1 = dng_color_matrix1()

    description = (
        f"pseudoraw POC reconstruction from JPEG ({source_jpeg_path}). "
        f"NOT real sensor RAW data -- a linear reconstruction for extra "
        f"grading headroom. pipeline={pipeline_version}"
    )

    extratags = [
        # DNGVersion / DNGBackwardVersion: BYTE[4], 1.4.0.0
        (50706, "B", 4, (1, 4, 0, 0), False),
        (50707, "B", 4, (1, 4, 0, 0), False),
        # UniqueCameraModel / Make / Model: ASCII
        (50708, "s", 0, "pseudoraw virtual sensor", False),
        (271, "s", 0, "pseudoraw", False),
        (272, "s", 0, "pseudoraw-poc", False),
        # CalibrationIlluminant1: SHORT, D65
        (50778, "H", 1, CALIBRATION_ILLUMINANT_D65, False),
        # ColorMatrix1: SRATIONAL[9], XYZ(D65) -> camera-native(=linear sRGB)
        (50721, "2i", 9, _float_matrix_to_srational(color_matrix1), False),
        # AsShotNeutral: RATIONAL[3] -- our data is already neutral-balanced
        (50728, "2I", 3, _rational_flat(1.0) * 3, False),
        # WhiteLevel / BlackLevel: LONG[3]
        (50717, "I", 3, (65535, 65535, 65535), False),
        (50714, "I", 3, (0, 0, 0), False),
        # BaselineExposure: SRATIONAL, left neutral at 0
        (50730, "2i", 1, _rational_flat(0.0), False),
    ]

    write_kwargs = dict(
        photometric=DNG_PHOTOMETRIC_LINEAR_RAW,
        planarconfig="contig",
        description=description,
        software=f"pseudoraw {pipeline_version}",
        extratags=extratags,
        metadata=None,  # don't let tifffile add its own JSON/OME shape metadata
    )
    if compress:
        # Deflate = DNG Compression tag value 8, a standard lossless DNG
        # compression mode (not a hack -- it's in Adobe's own DNG spec).
        # predictor=True applies horizontal differencing before deflate,
        # which matters a lot here specifically: adjacent pixels in a
        # photo are usually close in value, so differencing turns that
        # correlation into many small/zero values deflate compresses
        # well -- without it, deflate alone barely helps on this kind of
        # data. Tiling (vs. one giant strip) lets readers decode regions
        # without pulling the whole plane into memory, which matters for
        # 20+ MP output.
        write_kwargs.update(
            compression="deflate",
            compressionargs={"level": 6},
            predictor=True,
            tile=(256, 256),
        )

    tifffile.imwrite(path, rgb16, **write_kwargs)
