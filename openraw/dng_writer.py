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
"OpenRAW virtual sensor" regardless, so readers can't mistake this for
that camera's actual sensor data even while the real Make/Model is shown.
"""

from __future__ import annotations

import numpy as np
import cv2
import tifffile

from .colormatrix import dng_color_matrix1
from .exif_transfer import build_dng_extratags
from .tonecurve import linear_to_srgb, srgb_to_linear
from ._version import __version__

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


def _preview_write_kwargs(quality: int) -> tuple[dict, str | None]:
    """
    Decide the preview's compression BEFORE opening the output file at
    all, by actually testing JPEG encoding on a throwaway array rather
    than assuming it'll work. Returns (kwargs_for_tf.write, warning_or_None).

    Why this exists: a real bug shipped where the preview always tried
    JPEG compression (needs the imagecodecs package, which is NOT part
    of tifffile's own built-in codecs -- unlike Deflate, which IS built
    in), with no fallback. On any machine without imagecodecs already
    installed for some unrelated reason, this crashed the ENTIRE
    pipeline run -- not a degraded preview, a total failure, just to
    produce a thumbnail. imagecodecs is now a declared (pyproject.toml)
    dependency, but a missing optional-feeling C-extension package
    souldn't be able to take down the whole tool: this checks up front
    and falls back to an uncompressed preview (still small -- it's
    already downscaled -- just bigger than a JPEG one) with a clear,
    one-line warning instead.
    """
    try:
        import imagecodecs  # noqa: F401

        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        imagecodecs.jpeg8_encode(probe, level=90)
        return (
            dict(
                photometric="rgb",
                compression="jpeg",
                compressionargs={"level": int(np.clip(quality, 1, 100))},
            ),
            None,
        )
    except Exception as e:
        return (
            dict(photometric="rgb", compression=None),
            f"preview: JPEG compression unavailable ({type(e).__name__}: {e}) "
            f"-- install/upgrade 'imagecodecs' for a smaller "
            f"preview. Falling back to an uncompressed preview for now; the "
            f"main image is unaffected.",
        )


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



LJPEG_TILE = 256
_VALID_BIT_DEPTHS = (16, 14, 12, 10)


def _linearization_table(bits: int) -> np.ndarray:
    """DNG LinearizationTable (tag 50712): code value -> linear uint16.
    Codes are sRGB-gamma encoded, so precision is spent perceptually evenly
    (the same idea as the tone curves inside Nikon/Leica raw files)."""
    n = 1 << bits
    g = np.arange(n, dtype=np.float32) / (n - 1)
    return np.round(srgb_to_linear(g) * 65535.0).astype(np.uint16)


def _encode_codes(rgb16: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Exact nearest-table-entry code for every linear value."""
    n = len(table)
    t = table.astype(np.int64)
    x = rgb16.astype(np.int64)
    code = np.clip(np.round(linear_to_srgb(rgb16.astype(np.float32) / 65535.0) * (n - 1)), 0, n - 1).astype(np.int64)
    for d in (-1, 1):  # gamma rounding can land one entry off; fix it
        alt = np.clip(code + d, 0, n - 1)
        code = np.where(np.abs(t[alt] - x) < np.abs(t[code] - x), alt, code)
    return code.astype(np.uint16)


def _ljpeg_tiles(data: np.ndarray, bits: int, tile: int = LJPEG_TILE, threads: int | None = None):
    """Yield each tile as a 3-component lossless JPEG (ITU T.81 process 14,
    predictor 1, no color transform) -- the layout Adobe's own DNG Converter
    uses for linear DNGs. Edge tiles are zero-padded to full tile size.

    Tiles are independent, and imagecodecs releases the GIL while encoding,
    so a thread pool gives real parallelism here. executor.map preserves
    order, which matters: tifffile writes tiles in the order yielded, so the
    output is byte-identical to single-threaded (pinned by a test).
    threads: None/0 = os.cpu_count(); 1 = no pool at all.
    """
    import imagecodecs
    import os
    from concurrent.futures import ThreadPoolExecutor

    h, w, c = data.shape
    coords = [(ty, tx) for ty in range(0, h, tile) for tx in range(0, w, tile)]

    def encode(yx):
        ty, tx = yx
        t = np.zeros((tile, tile, c), dtype=np.uint16)
        blk = data[ty:ty + tile, tx:tx + tile]
        t[:blk.shape[0], :blk.shape[1]] = blk
        return imagecodecs.jpeg8_encode(
            t, lossless=True, predictor=1, bitspersample=bits,
            colorspace="RGB", outcolorspace="RGB",
        )

    n = threads or os.cpu_count() or 1
    if n <= 1:
        yield from map(encode, coords)
        return
    with ThreadPoolExecutor(max_workers=n) as pool:
        # map() submits everything up front; at ~50KB/tile even an 18MP image
        # is ~1000 tiles / ~50MB in flight, same order as the file itself.
        yield from pool.map(encode, coords)


def _main_image_payload(rgb16: np.ndarray, compression: str, bit_depth: int, threads: int | None = None):
    """Return (data, extra_write_kwargs, extra_raw_ifd_tags) for the main image."""
    if compression == "none":
        # Uncompressed is always bit-exact 16-bit linear (bit_depth ignored).
        return rgb16, {}, []
    if compression != "ljpeg":
        raise ValueError(f"compression must be 'ljpeg' or 'none', got {compression!r}")
    if bit_depth not in _VALID_BIT_DEPTHS:
        raise ValueError(f"bit_depth must be one of {_VALID_BIT_DEPTHS}, got {bit_depth}")
    try:  # probe BEFORE writing anything: needs imagecodecs w/ libjpeg-turbo >= 3 lossless
        import imagecodecs
        imagecodecs.jpeg8_encode(np.zeros((8, 8, 3), np.uint16), lossless=True, predictor=1,
                                 bitspersample=bit_depth, colorspace="RGB", outcolorspace="RGB")
    except Exception as e:
        print(f"[openraw] lossless JPEG unavailable ({type(e).__name__}: {e}) -- install/upgrade "
              f"'imagecodecs' (>=2023.9.18). Writing UNCOMPRESSED (larger, still valid) instead.")
        return rgb16, {}, []
    raw_tags = []
    if bit_depth == 16:
        data = rgb16  # bit-exact: no table, values stored as-is
    else:
        table = _linearization_table(bit_depth)
        data = _encode_codes(rgb16, table)
        raw_tags.append((50712, "H", len(table), tuple(int(v) for v in table), False))
    kwargs = dict(
        shape=rgb16.shape, dtype=np.uint16,
        compression=7,  # JPEG family; payload is LOSSLESS JPEG, not DCT
        bitspersample=bit_depth,  # explicit: tifffile otherwise assumes 12 for JPEG+uint16
        tile=(LJPEG_TILE, LJPEG_TILE),
    )
    return _ljpeg_tiles(data, bit_depth, threads=threads), kwargs, raw_tags


def _finalize_main_ifd(path: str, main_in_subifd: bool) -> None:
    """
    Post-write fix-up of the main raw IFD, in place. Why this exists:

    DNG's LinearRaw PhotometricInterpretation (34892) is unknown to tifffile,
    so tifffile has to GUESS how the 3 samples relate to it -- and different
    tifffile versions guess differently. That bit this project twice: one
    version wrote a spurious ExtraSamples tag (a real user bug), and a newer
    one (2026.9.20, found in a clean-venv install test) miscounts the tile
    layout and raises StopIteration. Patching around each version's guess is
    a losing game, so the main image is written as plain RGB -- which every
    tifffile version understands exactly -- and fixed up here:

    - PhotometricInterpretation is set to 34892 (2-byte in-place edit).
    - YCbCrSubSampling (530) / ReferenceBlackWhite (532) are removed:
      tifffile adds them to ANY JPEG-compressed RGB IFD, but they're
      meaningless for LinearRaw lossless-JPEG data. Removal = shift later
      12-byte IFD entries up and decrement the entry count; the leftover
      bytes are simply unreferenced, which is valid TIFF.

    Result: byte-identical output across tifffile versions (verified on
    2026.3.3 and 2026.9.20). Classic (non-Big) TIFF only -- tifffile writes
    classic for files under 4GB, i.e. any realistic photo.
    """
    import struct

    with tifffile.TiffFile(path) as tf:
        if tf.is_bigtiff:
            raise RuntimeError("unexpected BigTIFF output; _finalize_main_ifd supports classic TIFF only")
        bo = tf.byteorder
        page = tf.pages[0].pages[0] if main_in_subifd else tf.pages[0]
        ifd_off = page.offset
    with open(path, "r+b") as f:
        f.seek(ifd_off)
        (n,) = struct.unpack(bo + "H", f.read(2))
        entries = [f.read(12) for _ in range(n)]
        (next_ifd,) = struct.unpack(bo + "I", f.read(4))
        kept = []
        for e in entries:
            code, typ, count = struct.unpack(bo + "HHI", e[:8])
            if code in (530, 532):
                continue
            if code == 262:
                e = e[:8] + struct.pack(bo + "H", DNG_PHOTOMETRIC_LINEAR_RAW) + e[10:]
            kept.append(e)
        f.seek(ifd_off)
        f.write(struct.pack(bo + "H", len(kept)) + b"".join(kept) + struct.pack(bo + "I", next_ifd))
        f.write(b"\0" * (12 * (n - len(kept))))  # zero the now-unused tail

def write_linear_dng(
    path: str,
    rgb16: np.ndarray,
    source_jpeg_path: str = "",
    pipeline_version: str = __version__,
    compression: str = "ljpeg",
    bit_depth: int = 12,
    threads: int | None = None,
    exif_fields: dict | None = None,
    write_preview: bool = True,
    preview_max_dim: int = 1024,
    preview_quality: int = 90,
) -> None:
    """
    rgb16: uint16 HxWx3, range [0, 65535], scene-linear.
    compression: "ljpeg" (default) or "none".
        "ljpeg": the main image is stored as 256x256 tiles, each a
        3-component LOSSLESS JPEG (ITU T.81 process 14 -- the predictive
        lossless codec real cameras use for compressed raw, NOT the lossy
        DCT JPEG used for photos). Verified on a real 18MP photo: passes
        Adobe's dng_validate with zero errors/warnings, and both Adobe's
        DNG SDK and libraw decode it bit-identically to the uncompressed
        file. History, so nobody repeats it: Deflate/LZW/PackBits were
        tried earlier and libraw rejects all three for this DNG structure;
        lossy DCT JPEG wrecks 16-bit data (~30000/65535 mean error); and a
        single-component "W*3 wide" LJPEG layout passes Adobe's SDK but
        libraw scrambles the pixels inside each tile -- only genuine
        3-component lossless JPEG works in both.
        "none": one uncompressed bit-exact 16-bit linear image (largest).
    bit_depth: 16, 14, 12 (default) or 10 -- ljpeg only.
        16 stores our linear values exactly. Below 16, values are stored as
        sRGB-gamma code values plus a DNG LinearizationTable (tag 50712)
        that readers use to expand them back to linear -- standard DNG,
        same idea as the curves inside Nikon/Leica raws. Gamma spends
        precision perceptually evenly, so fewer bits suffice. Measured on
        a real 4896x3672 photo, error in units of the SOURCE JPEG's own
        8-bit steps: 14-bit <= 0.015 (~62MB), 12-bit <= 0.05 (~49MB),
        10-bit <= 0.144 (~36MB), vs 16-bit ~76MB and uncompressed 108MB.
        12-bit still keeps 16x finer tonal steps than the source, so the
        deband/dither headroom survives intact. For scale: the source's
        own 8-bit pixels with zero headroom need ~24MB losslessly -- the
        6MB JPEG is only that small because it discards information.
    threads: worker threads for lossless-JPEG tile encoding. None (default) =
        all CPU cores; 1 = single-threaded. Output is byte-identical either way.
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

    # Filename only, never the full path: a path like C:\\Users\\<name>\\... would
    # leak the user's account name and folder layout into every shared DNG.
    source_name = source_jpeg_path.replace("\\", "/").rsplit("/", 1)[-1]
    description = (
        f"OpenRAW reconstruction from JPEG ({source_name}). "
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
        50708: (50708, "s", 0, "OpenRAW virtual sensor", False),
        # Make / Model placeholders -- overridden below if the source
        # JPEG actually had real camera EXIF.
        271: (271, "s", 0, "OpenRAW", False),
        272: (272, "s", 0, "OpenRAW", False),
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

    # Orientation: TIFF/DNG structural default 1 ("stored as-is") when the
    # source has none -- not invented camera metadata, just the spec's
    # default made explicit (Adobe's dng_validate warns when it's absent).
    # A real source Orientation overrides it below.
    tags[274] = (274, "H", 1, 1, False)

    if exif_fields:
        for tag in build_dng_extratags(exif_fields):
            tags[tag[0]] = tag  # real value overrides any placeholder above

    # Split per the DNG spec: raw-data tags belong to the raw image's own
    # IFD; EVERYTHING else (DNGVersion, UniqueCameraModel, ColorMatrix1,
    # AsShotNeutral, Make/Model/Orientation, EXIF...) belongs in IFD0,
    # whatever IFD0 happens to be. Readers built on Adobe's DNG SDK
    # (Android's Skia, Luminar, and most commercial tools) identify a
    # file as DNG by finding DNGVersion IN IFD0 -- a real bug shipped
    # where adding the preview IFD moved all of these into the SubIFD
    # with the main image, so IFD0 (the preview) had no DNGVersion.
    # libraw scans every IFD and still opened it, which hid the bug;
    # Adobe's own dng_validate reported "Missing DNGVersion".
    RAW_IFD_TAGS = {50714, 50717}  # BlackLevel, WhiteLevel
    raw_extratags = [t for c, t in tags.items() if c in RAW_IFD_TAGS]
    ifd0_extratags = [t for c, t in tags.items() if c not in RAW_IFD_TAGS]
    extratags = ifd0_extratags + raw_extratags  # single-IFD layout: all together

    main_kwargs = dict(
        # Written as plain RGB, then patched to LinearRaw (34892) by
        # _finalize_main_ifd -- see its docstring for why.
        photometric="rgb",
        planarconfig="contig",
        description=description,
        software=f"OpenRAW {pipeline_version}",
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
    main_data, main_extra, main_raw_tags = _main_image_payload(rgb16, compression, bit_depth, threads)
    main_kwargs.update(main_extra)
    raw_extratags = raw_extratags + main_raw_tags  # LinearizationTable is a raw-IFD tag
    main_kwargs["extratags"] = main_kwargs["extratags"] + main_raw_tags

    if not write_preview:
        tifffile.imwrite(path, main_data, **main_kwargs)
        _finalize_main_ifd(path, main_in_subifd=False)
        return

    # Preview layout: IFD0 (preview) carries the DNG/EXIF tags, the raw
    # SubIFD carries only its raw-data tags -- see the split above.
    main_kwargs["extratags"] = raw_extratags

    preview = _make_preview(rgb16, max_dim=preview_max_dim)
    preview_kwargs, preview_warning = _preview_write_kwargs(preview_quality)
    if preview_warning:
        print(f"[openraw] {preview_warning}")

    # Standard DNG structure: IFD0 = small preview (what a basic viewer
    # or quick-look reads first), main full-res data in a SubIFD (tag
    # 330) off of it -- verified structurally with tifffile's own reader
    # and exiftool; see write_preview's docstring for why this exists.
    # preview_kwargs' compression was decided up front (see
    # _preview_write_kwargs) rather than discovered mid-write, so this
    # TiffWriter session can't fail partway through from a missing codec
    # and leave a half-written file behind.
    with tifffile.TiffWriter(path) as tf:
        tf.write(
            preview,
            subfiletype=1,  # reduced-resolution image
            subifds=1,  # reserve one SubIFD slot for the main image below
            description=f"OpenRAW preview ({preview.shape[1]}x{preview.shape[0]})",
            extratags=ifd0_extratags,  # DNGVersion etc MUST be in IFD0 -- see above
            metadata=None,  # same as the main write -- without this, tifffile
            # tries to parse our custom description as its own auto-generated
            # shape-JSON metadata and warns "invalid shaped series metadata or
            # corrupted file" on read. Cosmetic (pixel data was never affected,
            # confirmed both with and without this fix) but alarming to see in
            # a terminal, worth silencing properly rather than leaving it.
            **preview_kwargs,
        )
        tf.write(main_data, **main_kwargs)
    _finalize_main_ifd(path, main_in_subifd=True)
