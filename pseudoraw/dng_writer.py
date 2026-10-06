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
invented NoiseProfile, no fake CFA pattern). Real camera metadata
(Make/Model/lens/exposure/etc) IS carried through when the source JPEG
actually has it, via exif_transfer.py -- see that module for exactly
what's copied, what's deliberately not (MakerNote, GPS by default), and
why. Make/Model specifically: when the source JPEG has them, the real
camera's values are written (useful for lens-correction lookups, display,
etc in raw converters) and UniqueCameraModel is still always
"pseudoraw virtual sensor" regardless, so readers can't mistake this for
that camera's actual sensor data even while the real Make/Model is shown.
"""

from __future__ import annotations

import numpy as np
import cv2
import tifffile

from .colormatrix import dng_color_matrix1
from .exif_transfer import build_dng_extratags
from .tonecurve import linear_to_srgb

DNG_PHOTOMETRIC_LINEAR_RAW = 34892
CALIBRATION_ILLUMINANT_D65 = 21


def _make_preview(rgb16: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """
    rgb16: uint16 HxWx3, scene-linear (same data the main IFD gets).
    Returns: uint8 HxWx3, sRGB-gamma, downscaled so its longer side is
        max_dim (never upscaled -- a smaller source just gets used as-is).

    This exists because a DNG with only the full-res main image has no
    preview/thumbnail for quick-look viewers and raw-import screens to
    show without doing a full raw decode -- spec-legal, but a real
    source of broken-looking previews in practice (found investigating
    a user's "looks corrupt" report: that file had one IFD, no preview).
    """
    h, w = rgb16.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        small = cv2.resize(rgb16, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        small = rgb16

    linear = small.astype(np.float32) / 65535.0
    srgb = linear_to_srgb(linear)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


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
    compress: bool = False,
    compression_level: int = 9,
    exif_fields: dict | None = None,
    write_preview: bool = True,
    preview_max_dim: int = 1024,
    preview_quality: int = 90,
) -> None:
    """
    rgb16: uint16 HxWx3, range [0, 65535], scene-linear.
    compress: if True, write the MAIN image with Adobe Deflate (DNG
        Compression tag = 8) + horizontal-differencing predictor --
        smaller files, fully lossless (verified: pixel-identical
        round-trip). DEFAULT IS FALSE, deliberately, after real testing
        against actual libraw (via rawpy) turned up something the
        previous version of this docstring claimed wrongly: libraw does
        NOT support Deflate for this kind of DNG. Also tested and also
        failing: LZW, PackBits. The only compression libraw accepts here
        is JPEG (tag 7) -- see the note below on why that's not usable
        for the main data either. This matters a lot in practice: libraw
        is what darktable, RawTherapee, and a large swath of the open
        raw-processing ecosystem are built on, so a Deflate-compressed
        "optimized" file was silently unreadable by exactly the tools an
        open-source project's users are most likely to actually use.
        Confirmed via rawpy.imread() + a full raw.postprocess() call,
        not just a decode attempt. Set True only if you know your
        specific target reader supports it -- Adobe's own apps (which
        use Adobe's own DNG SDK, not libraw) likely do, but that's not
        verified here either; don't assume compatibility you haven't
        tested. Uncompressed means a 24MP photo is ~140MB -- a real cost,
        but a working file beats a smaller broken one.

        NOTE on JPEG (DNG Compression tag = 7) for the MAIN data: tested
        directly and it is NOT a lossless option here -- it's standard
        lossy DCT JPEG, and on our 16-bit samples it's not a subtle
        quality hit, it's catastrophic (~30000/65535 mean pixel error in
        testing). Never used for the main image, on purpose, even though
        it's the one compression mode libraw actually accepts for this
        photometric. (A genuinely lossless option DOES exist --
        imagecodecs' ljpeg_encode, true lossless-predictive JPEG, the
        scheme real cameras actually use for compressed RAW -- but it
        only handles single-component data; our 3-channel interleaved
        RGB would need per-channel encoding and manual TIFF tile
        construction bypassing tifffile's built-in compression entirely.
        Real potential future work, not done here -- see TRAINING.md-
        style honesty: untested complexity wasn't worth rushing in
        response to a bug report where "just turn it off" was already a
        proven-safe fix.) JPEG IS used for the preview below, where
        lossy is fine -- a thumbnail isn't the data.
    compression_level: 1 (fastest) - 9 (smallest), zlib/deflate scale,
        for the MAIN image. Default 9: this runs once per photo, not in
        a hot loop, so there is no real reason to leave size on the
        table for speed here.
    exif_fields: output of exif_transfer.extract_exif() on the source
        JPEG's PIL Exif object, or None. When given, real camera metadata
        (Make/Model/lens/exposure/ISO/focal length/orientation/etc -- see
        exif_transfer.py for the exact list) is carried into the DNG.
        Fields absent from the source are simply absent here too, never
        invented -- a point-and-shoot's auto-mode JPEG with no lens info
        produces a DNG with no lens info, not a guessed one. No GPS --
        dropped entirely, see exif_transfer.py.
    write_preview: if True (default), write a small JPEG-compressed
        sRGB preview as IFD0, with the real full-resolution LinearRaw
        data in a SubIFD (DNG tag 330) off of it -- the structure DNG's
        own spec recommends specifically so quick-look viewers and
        raw-import screens can show something without a full raw decode.
        A DNG with only the main image (no preview) is spec-legal but a
        real source of broken-looking previews in practice -- found
        investigating a user's bug report. Lossy JPEG is fine here: this
        is a thumbnail, not the data.
    preview_max_dim: longer-side pixel size for the preview (default
        1024). Never upscales a smaller source.
    preview_quality: JPEG quality 0-100 for the preview (default 90).
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

    # Keyed by tag id so a real EXIF Make/Model (added below) cleanly
    # overrides the placeholder instead of tifffile receiving the same
    # tag code twice.
    tags: dict[int, tuple] = {
        # DNGVersion / DNGBackwardVersion: BYTE[4], 1.4.0.0
        50706: (50706, "B", 4, (1, 4, 0, 0), False),
        50707: (50707, "B", 4, (1, 4, 0, 0), False),
        # UniqueCameraModel: ASCII -- always this, regardless of real
        # camera info below, so a reader can never mistake this for that
        # camera's actual sensor data (see module docstring).
        50708: (50708, "s", 0, "pseudoraw virtual sensor", False),
        # Make / Model placeholders -- overridden below if the source
        # JPEG actually had real camera EXIF.
        271: (271, "s", 0, "pseudoraw", False),
        272: (272, "s", 0, "pseudoraw-poc", False),
        # CalibrationIlluminant1: SHORT, D65
        50778: (50778, "H", 1, CALIBRATION_ILLUMINANT_D65, False),
        # ColorMatrix1: SRATIONAL[9], XYZ(D65) -> camera-native(=linear sRGB)
        50721: (50721, "2i", 9, _float_matrix_to_srational(color_matrix1), False),
        # AsShotNeutral: RATIONAL[3] -- our data is already neutral-balanced
        50728: (50728, "2I", 3, _rational_flat(1.0) * 3, False),
        # WhiteLevel / BlackLevel: LONG[3]
        50717: (50717, "I", 3, (65535, 65535, 65535), False),
        50714: (50714, "I", 3, (0, 0, 0), False),
        # BaselineExposure: SRATIONAL, left neutral at 0
        50730: (50730, "2i", 1, _rational_flat(0.0), False),
    }

    if exif_fields:
        for tag in build_dng_extratags(exif_fields):
            tags[tag[0]] = tag  # real value overrides any placeholder above

    extratags = list(tags.values())

    main_kwargs = dict(
        photometric=DNG_PHOTOMETRIC_LINEAR_RAW,
        planarconfig="contig",
        description=description,
        software=f"pseudoraw {pipeline_version}",
        extratags=extratags,
        metadata=None,  # don't let tifffile add its own JSON/OME shape metadata
        subfiletype=0,  # full-resolution image (vs. the preview's subfiletype=1 below)
        # Explicit, deliberately not left to tifffile's inference: DNG's
        # LinearRaw (34892) isn't in tifffile's own recognized PHOTOMETRIC
        # enum, so tifffile has to guess how many of our 3 samples its
        # photometric value "already accounts for" versus how many are
        # "extra". Different tifffile versions guess differently for an
        # unrecognized photometric -- confirmed by a real bug report: a
        # user's tifffile install guessed 1, writing a spurious
        # ExtraSamples(UNSPECIFIED, UNSPECIFIED) tag claiming our 2 real
        # color channels were "extra" samples needing interpretation,
        # which a strict DNG reader could reasonably choke on even
        # though lenient ones (including tifffile's own reader) tolerate
        # it fine. This sandbox's tifffile version happened to guess
        # correctly (3) by default, which is exactly why this didn't
        # reproduce here until explicitly tested against the reported
        # file -- a version-dependent default was never something to
        # rely on. extrasamples=() makes the real answer (zero extra
        # samples; all 3 are the photometric's own color channels)
        # explicit and version-independent.
        extrasamples=(),
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
        main_kwargs.update(
            compression="deflate",
            compressionargs={"level": int(np.clip(compression_level, 1, 9))},
            predictor=True,
            tile=(256, 256),
        )

    if not write_preview:
        tifffile.imwrite(path, rgb16, **main_kwargs)
        return

    preview = _make_preview(rgb16, max_dim=preview_max_dim)

    # Standard DNG structure: IFD0 = small preview (what a basic viewer
    # or quick-look reads first), main full-res data in a SubIFD (tag
    # 330) off of it -- verified structurally with tifffile's own reader
    # and exiftool; see write_preview's docstring for why this exists.
    with tifffile.TiffWriter(path) as tf:
        tf.write(
            preview,
            photometric="rgb",
            compression="jpeg",
            compressionargs={"level": int(np.clip(preview_quality, 1, 100))},
            subfiletype=1,  # reduced-resolution image
            subifds=1,  # reserve one SubIFD slot for the main image below
            description=f"pseudoraw preview ({preview.shape[1]}x{preview.shape[0]})",
        )
        tf.write(rgb16, **main_kwargs)
