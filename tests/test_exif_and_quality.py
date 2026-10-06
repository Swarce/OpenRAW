"""
Tests for the EXIF-preservation and core-quality improvements:
- exif_transfer.py: real metadata carried through, nothing fabricated
  for absent fields. No GPS (dropped entirely, not just off-by-default).
- decode.py: chroma subsampling detection, chroma-specific quality est.
- chroma.py: 4:4:4 short-circuit
- bitdepth.py: dither amplitude reduced in detailed regions
- dng_writer.py: no spurious ExtraSamples tag, preview+SubIFD structure,
  and critically: real libraw (via rawpy) actually opens the output --
  not just our own writer round-tripping with itself.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile
from PIL import Image
from PIL.ExifTags import Base, IFD

from openraw.decode import load_jpeg
from openraw.exif_transfer import extract_exif, build_dng_extratags
from openraw.chroma import refine_chroma
from openraw.dng_writer import write_linear_dng
from openraw import OpenRawPipeline, PipelineConfig

from .helpers import main_page, main_array, ifd0_tags, decoded_linear

try:
    import rawpy
    _HAVE_RAWPY = True
except ImportError:
    _HAVE_RAWPY = False

needs_rawpy = pytest.mark.skipif(not _HAVE_RAWPY, reason="needs rawpy (= libraw)")


def _make_jpeg_with_exif(path, make="FUJIFILM", model="X-T4", subsampling=2):
    img = Image.new("RGB", (64, 48), (120, 80, 40))
    exif = Image.Exif()
    exif[Base.Make] = make
    exif[Base.Model] = model
    exif[Base.Orientation] = 6
    exif[Base.DateTime] = "2026:05:01 12:30:00"

    sub_ifd = {
        Base.ExposureTime: (1, 250),
        Base.FNumber: (28, 10),
        Base.ISOSpeedRatings: 400,
        Base.FocalLength: (56, 1),
        Base.LensModel: "XF56mmF1.2 R",
        Base.DateTimeOriginal: "2026:05:01 12:30:00",
    }
    exif.get_ifd(IFD.Exif).update(sub_ifd)

    img.save(path, format="JPEG", quality=90, exif=exif, subsampling=subsampling)


def _make_jpeg_no_exif(path, subsampling=2):
    img = Image.new("RGB", (64, 48), (90, 140, 60))
    img.save(path, format="JPEG", quality=90, subsampling=subsampling)


# ---- exif_transfer.py ----

def test_extract_exif_pulls_real_camera_fields(tmp_path):
    p = str(tmp_path / "with_exif.jpg")
    _make_jpeg_with_exif(p)

    img = Image.open(p)
    img.load()
    fields = extract_exif(img.getexif())

    assert fields["ifd0"][Base.Make] == "FUJIFILM"
    assert fields["ifd0"][Base.Model] == "X-T4"
    assert fields["ifd0"][Base.Orientation] == 6
    assert fields["exif_sub"][Base.ISOSpeedRatings] == 400
    assert fields["exif_sub"][Base.LensModel] == "XF56mmF1.2 R"
    assert "gps" not in fields  # dropped entirely -- see exif_transfer.py


def test_extract_exif_on_auto_mode_jpeg_has_no_fabricated_fields(tmp_path):
    """The user's explicit point-and-shoot-in-auto case: no EXIF present
    should mean no fields present, not plausible-looking defaults."""
    p = str(tmp_path / "no_exif.jpg")
    _make_jpeg_no_exif(p)

    img = Image.open(p)
    img.load()
    fields = extract_exif(img.getexif())

    assert fields["ifd0"] == {}
    assert fields["exif_sub"] == {}


def test_dng_write_carries_real_make_model_overriding_placeholder(tmp_path):
    rgb16 = (np.random.default_rng(0).random((16, 16, 3)) * 65535).astype(np.uint16)
    p = str(tmp_path / "with_exif.jpg")
    _make_jpeg_with_exif(p, make="FUJIFILM", model="X-T4")
    img = Image.open(p)
    img.load()
    fields = extract_exif(img.getexif())

    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, exif_fields=fields)

    with tifffile.TiffFile(out) as tf:
        tags = ifd0_tags(tf)  # identity/EXIF tags live in IFD0 per DNG spec
        assert tags[271].value == "FUJIFILM"
        assert tags[272].value == "X-T4"
        # UniqueCameraModel always stays the synthetic-sensor label
        # regardless of real Make/Model being present.
        assert tags[50708].value == "OpenRAW virtual sensor"
        fnum_num, fnum_den = tags[33437].value  # RATIONAL tags read back as a raw (num, den) tuple
        assert fnum_num / fnum_den == pytest.approx(2.8, abs=0.01)


def test_dng_write_has_no_extrasamples_tag(tmp_path):
    """
    Regression test for a real bug a user hit: an ExtraSamples tag
    claiming 2 of our 3 real color channels were "extra" (unspecified)
    samples, which a strict DNG reader could reject even though lenient
    readers (tifffile's own included) tolerate it. Root cause: DNG's
    PhotometricInterpretation value (34892, LinearRaw) isn't in
    tifffile's own recognized enum, so different tifffile versions
    guess differently how many of our samples it "explains" versus
    leaves as extra. Pin it explicit and version-independent.

    Tested at the user's actual reported dimensions (4896x3672) as well
    as a small size, since the bug didn't reproduce at small sizes in
    the environment this was originally debugged in -- which turned out
    to be irrelevant (it was a tifffile-version issue, not a size
    issue), but there's no reason not to cover both now that it's cheap.
    """
    for h, w in [(16, 16), (3672, 4896)]:
        rgb16 = (np.random.default_rng(0).random((h, w, 3)) * 65535).astype(np.uint16)
        out = str(tmp_path / f"out_{h}x{w}.dng")
        write_linear_dng(out, rgb16)

        with tifffile.TiffFile(out) as tf:
            page = main_page(tf)
            assert 338 not in page.tags, f"spurious ExtraSamples tag at {h}x{w}"
            assert page.tags[277].value == 3  # SamplesPerPixel
            assert page.photometric == 34892

        exact = str(tmp_path / f"exact_{h}x{w}.dng")
        write_linear_dng(exact, rgb16, compression="none")
        assert np.array_equal(main_array(exact), rgb16)


def test_dng_write_without_exif_fields_keeps_placeholder(tmp_path):
    rgb16 = (np.random.default_rng(0).random((16, 16, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, exif_fields=None)

    with tifffile.TiffFile(out) as tf:
        tags = ifd0_tags(tf)
        assert tags[271].value == "OpenRAW"
        assert tags[272].value == "OpenRAW"
        assert 33437 not in tags  # no fabricated FNumber


def test_dng_has_preview_ifd_by_default(tmp_path):
    """A DNG with only the full-res main image (no preview) is spec-legal
    but a known real-world source of broken-looking previews -- found
    investigating a user's bug report. Pin that IFD0 is a usably-small
    preview and the main data lives in a SubIFD off of it."""
    rgb16 = (np.random.default_rng(0).random((512, 512, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, preview_max_dim=128)

    with tifffile.TiffFile(out) as tf:
        preview_page = tf.pages[0]
        assert max(preview_page.shape[:2]) <= 128
        assert preview_page.photometric == 6  # YCbCr, standard for embedded JPEG
        main = main_page(tf)
        assert main.shape == (512, 512, 3)
        assert main.photometric == 34892


def test_dng_preview_falls_back_gracefully_without_imagecodecs(tmp_path, monkeypatch):
    """
    Regression test for a real bug a user hit on Windows/Python 3.14:
    the preview always tried JPEG compression, which needs the
    imagecodecs package (NOT bundled with tifffile the way Deflate is).
    With it missing, the ENTIRE pipeline run crashed just to produce a
    thumbnail. Must now fall back to an uncompressed preview with a
    warning instead -- main image unaffected either way.

    Mocks the import failure (clean, repeatable) rather than actually
    uninstalling imagecodecs, which would affect other tests in this
    process.
    """
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "imagecodecs":
            raise ModuleNotFoundError("No module named 'imagecodecs'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rgb16 = (np.random.default_rng(0).random((64, 64, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16)  # must NOT raise

    with tifffile.TiffFile(out) as tf:
        assert tf.pages[0].compression == 1  # fell back to uncompressed preview
        main = main_page(tf)
        assert main.photometric == 34892

    arr = main_array(out)
    assert np.array_equal(arr, rgb16)  # main image completely unaffected


def test_dng_no_preview_option(tmp_path):
    rgb16 = (np.random.default_rng(0).random((32, 32, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, write_preview=False)

    with tifffile.TiffFile(out) as tf:
        assert len(tf.pages) == 1
        assert tf.pages[0].photometric == 34892


@needs_rawpy
def test_dng_default_settings_actually_open_in_real_libraw(tmp_path):
    """
    THE regression test that matters most here. Previous tests in this
    file only checked that tifffile (which wrote the file) can read it
    back -- that's necessary but not sufficient, and it's exactly how a
    real bug got through: default settings (Deflate compression) wrote
    files that tifffile round-tripped perfectly but real libraw rejected
    outright with "Unsupported file format or not RAW file". Found by
    testing against actual rawpy/libraw, not by inspecting tags harder.

    This test is the one that would have caught it: write with pipeline
    defaults, open with real libraw, run an actual raw.postprocess().
    """
    rgb16 = (np.random.default_rng(0).random((256, 256, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16)  # defaults: compress=False, write_preview=True

    with rawpy.imread(out) as raw:
        assert raw.sizes.raw_width == 256
        assert raw.sizes.raw_height == 256
        processed = raw.postprocess()
        assert processed.shape[0] > 0 and processed.shape[1] > 0


@needs_rawpy
@pytest.mark.parametrize("bit_depth,max_err_steps", [(16, 0.0), (14, 0.05), (12, 0.1), (10, 0.25)])
def test_ljpeg_decodes_correctly_in_real_libraw(tmp_path, bit_depth, max_err_steps):
    """
    Lossless-JPEG tiles (+ LinearizationTable below 16 bits) must decode in
    REAL libraw to the same image as the uncompressed file: bit-exact at 16,
    and within the measured precision bound (in units of one source 8-bit
    sRGB step) below that. History: Deflate/LZW/PackBits were rejected by
    libraw outright, and a single-component W*3 LJPEG layout passed Adobe's
    SDK but libraw scrambled pixels inside every tile -- this test pins the
    layout that works in BOTH.
    """
    from openraw.tonecurve import linear_to_srgb, srgb_to_linear
    rng = np.random.default_rng(0)
    # photo-like: smooth gradients + texture, 3 channels, not pure noise
    yy, xx = np.mgrid[0:300, 0:400]
    base = np.stack([xx / 400, yy / 300, 0.5 + 0.3 * np.sin(xx / 30)], -1)
    src8 = (np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1) * 255).astype(np.uint8)
    rgb16 = (srgb_to_linear(src8.astype(np.float32) / 255) * 65535).astype(np.uint16)

    def lr(p):
        with rawpy.imread(p) as r:
            return r.postprocess(output_bps=16, no_auto_bright=True, gamma=(1, 1),
                                 use_camera_wb=True).astype(np.float32) / 65535

    ref, out = str(tmp_path / "ref.dng"), str(tmp_path / "out.dng")
    write_linear_dng(ref, rgb16, compression="none")
    write_linear_dng(out, rgb16, compression="ljpeg", bit_depth=bit_depth)
    err = np.abs(linear_to_srgb(lr(out)) - linear_to_srgb(lr(ref))) * 255
    assert err.max() <= max_err_steps
    assert os.path.getsize(out) < os.path.getsize(ref)


def test_ljpeg_falls_back_to_uncompressed_without_imagecodecs(tmp_path, monkeypatch):
    """Missing imagecodecs must degrade to a valid uncompressed file with a
    warning -- never crash the run (that crash happened once, for the preview)."""
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "imagecodecs":
            raise ModuleNotFoundError("No module named 'imagecodecs'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    rgb16 = (np.random.default_rng(0).random((64, 64, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16)  # default ljpeg must not raise
    with tifffile.TiffFile(out) as tf:
        assert main_page(tf).compression == 1
    assert np.array_equal(main_array(out), rgb16)


# ---- decode.py: chroma subsampling + per-channel quality ----

def test_decode_detects_subsampled_vs_444(tmp_path):
    p_sub = str(tmp_path / "sub.jpg")
    p_444 = str(tmp_path / "full.jpg")
    _make_jpeg_no_exif(p_sub, subsampling=2)  # 4:2:0
    _make_jpeg_no_exif(p_444, subsampling=0)  # 4:4:4

    d_sub = load_jpeg(p_sub)
    d_444 = load_jpeg(p_444)

    assert d_sub.is_chroma_subsampled is True
    assert d_444.is_chroma_subsampled is False


def test_full_pipeline_end_to_end_with_exif_and_444_jpeg(tmp_path):
    """Integration check: a 4:4:4 JPEG with real EXIF goes through the
    whole classical pipeline and produces a valid DNG carrying the EXIF,
    without chroma refinement altering anything (since is_chroma_subsampled
    is False) and without crashing on any of the new wiring."""
    p = str(tmp_path / "in.jpg")
    _make_jpeg_with_exif(p, subsampling=0)
    out = str(tmp_path / "out.dng")

    pipeline = OpenRawPipeline()
    result = pipeline.run_to_dng(p, out)

    assert result.decoded.is_chroma_subsampled is False
    with tifffile.TiffFile(out) as tf:
        assert ifd0_tags(tf)[271].value == "FUJIFILM"


# ---- chroma.py: 4:4:4 short-circuit ----

def test_refine_chroma_is_noop_when_not_subsampled():
    rgb = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    out = refine_chroma(rgb, strength=0.9, is_chroma_subsampled=False)
    assert np.array_equal(rgb, out)


def test_refine_chroma_still_acts_when_subsampled():
    rgb = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    out = refine_chroma(rgb, strength=0.9, is_chroma_subsampled=True)
    assert not np.array_equal(rgb, out)


# ---- bitdepth.py: flatness-gated dithering ----

def test_dither_amplitude_lower_in_detailed_regions():
    """A flat half and a noisy/detailed half of the same image should
    receive visibly different dither amounts -- the detailed half closer
    to its un-dithered value than the flat half is to its own."""
    from openraw.bitdepth import expand_to_16bit

    h, w = 64, 64
    linear = np.zeros((h, w, 3), dtype=np.float32)
    linear[:, :32] = 0.5  # flat half
    rng = np.random.default_rng(0)
    linear[:, 32:] = 0.5 + rng.normal(0, 0.15, size=(h, 32, 3)).astype(np.float32)
    linear = np.clip(linear, 0, 1)
    srgb_u8 = (linear * 255).astype(np.uint8)

    out_a = expand_to_16bit(linear, srgb_u8, dither=True, deband=False, seed=1)
    out_b = expand_to_16bit(linear, srgb_u8, dither=True, deband=False, seed=2)

    # Different seeds -> different dither noise. Compare how much the
    # flat half moved between seeds vs. the detailed half -- flat should
    # move more (full-strength dither) than detailed (reduced-strength).
    flat_diff = np.abs(out_a[:, :32].astype(np.int32) - out_b[:, :32].astype(np.int32)).mean()
    detailed_diff = np.abs(out_a[:, 32:].astype(np.int32) - out_b[:, 32:].astype(np.int32)).mean()
    assert flat_diff > detailed_diff


def test_deband_never_changes_pixels_by_more_than_one_quantization_step():
    """
    Regression test for a real user report ("overall softness on the entire
    image"): deband used to (a) judge flatness on the linear image, where
    shadows/midtones look artificially flat, and (b) apply an unbounded
    bilateral correction -- together they removed ~40% of measured detail
    on a real photo. Now its correction is clamped to one local 8-bit
    quantization step, so on detailed content, deband-only output must
    round-trip back to within +/-1 sRGB level of the source, everywhere.
    """
    from openraw.bitdepth import expand_to_16bit
    from openraw.tonecurve import srgb_to_linear, linear_to_srgb

    rng = np.random.default_rng(0)
    # Low-contrast fine texture across the tonal range -- the kind of
    # content the old flatness mask misclassified as "flat".
    base = np.tile(np.linspace(0.05, 0.9, 256), (256, 1))
    tex = base + rng.normal(0, 0.012, (256, 256))
    src8 = (np.clip(np.stack([tex] * 3, -1), 0, 1) * 255 + 0.5).astype(np.uint8)
    lin = srgb_to_linear(src8.astype(np.float32) / 255.0)

    out = expand_to_16bit(lin, src8, dither=False, deband=True)
    back8 = (linear_to_srgb(out.astype(np.float32) / 65535.0) * 255 + 0.5).astype(np.uint8)
    assert np.abs(back8.astype(int) - src8.astype(int)).max() <= 1


@pytest.mark.parametrize("write_preview", [True, False])
def test_dng_identity_tags_live_in_ifd0(tmp_path, write_preview):
    """
    Regression test for a real bug: Android's Skia ("image format may not
    be supported") and Luminar refused output files. Adobe's dng_validate:
    "Missing DNGVersion". Adding the preview IFD had moved DNGVersion,
    UniqueCameraModel, ColorMatrix1 etc. into the raw SubIFD; readers built
    on Adobe's DNG SDK look for them in IFD0. libraw scans all IFDs, so it
    still opened the file -- which is why libraw-only testing missed this.
    """
    rgb16 = (np.random.default_rng(0).random((64, 64, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, write_preview=write_preview)

    with tifffile.TiffFile(out) as tf:
        ifd0 = tf.pages[0].tags
        for code in (50706, 50707, 50708, 50721, 50778, 50728, 271, 272, 274):
            assert code in ifd0, f"tag {code} missing from IFD0"
        raw = main_page(tf).tags
        assert 50714 in raw and 50717 in raw  # BlackLevel/WhiteLevel on the raw IFD


def _dng_validate_bin():
    import os, shutil
    p = os.environ.get("OPENRAW_DNG_VALIDATE") or shutil.which("dng_validate")
    return p if p and os.path.exists(p) else None


@pytest.mark.skipif(_dng_validate_bin() is None,
                    reason="Adobe dng_validate not available (build: tools/build_dng_validate.sh)")
@pytest.mark.parametrize("write_preview", [True, False])
@pytest.mark.parametrize("compression,bit_depth", [("none", 16), ("ljpeg", 16), ("ljpeg", 12), ("ljpeg", 10)])
def test_dng_passes_adobe_dng_validate(tmp_path, write_preview, compression, bit_depth):
    """Strictest check available: Adobe's own reference validator must
    report no errors AND no warnings, for both layouts."""
    import subprocess
    rgb16 = (np.random.default_rng(0).random((128, 128, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, write_preview=write_preview, compression=compression, bit_depth=bit_depth)
    r = subprocess.run([_dng_validate_bin(), out], capture_output=True, text=True)
    text = r.stdout + r.stderr
    assert "*** Error" not in text and "*** Warning" not in text, text
    assert "Validation complete" in text


@pytest.mark.parametrize("write_preview", [True, False])
@pytest.mark.parametrize("compression,bit_depth", [("none", 16), ("ljpeg", 16), ("ljpeg", 12)])
def test_main_ifd_has_exactly_the_tags_we_intend(tmp_path, write_preview, compression, bit_depth):
    """
    The main image is written as plain RGB and patched to LinearRaw afterwards,
    because tifffile has to guess what an unknown photometric (34892) means and
    different versions guess differently (one wrote spurious ExtraSamples --
    a real user bug; 2026.9.20 raised StopIteration). Pin the end result:
    LinearRaw, no ExtraSamples, and none of the YCbCr tags tifffile adds to
    any JPEG-compressed RGB IFD. Holds for every tifffile version.
    """
    rgb16 = (np.random.default_rng(0).random((300, 300, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, write_preview=write_preview, compression=compression, bit_depth=bit_depth)
    with tifffile.TiffFile(out) as tf:
        page = main_page(tf)
        codes = {t.code for t in page.tags}
        assert page.photometric == 34892
        assert not codes & {338, 530, 532}
        assert page.tags[277].value == 3
    from .helpers import decoded_linear
    if bit_depth == 16:
        assert np.array_equal(decoded_linear(out), rgb16)


@pytest.mark.parametrize("src", ["/home/alice/Pictures/trip/IMG_1.jpg", r"C:\Users\alice\Pictures\IMG_1.jpg"])
def test_dng_never_embeds_the_source_folder_path(tmp_path, src):
    """Privacy: ImageDescription used to embed the full source path, leaking
    the user's account name and folder layout into every shared DNG."""
    rgb16 = (np.random.default_rng(0).random((16, 16, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, source_jpeg_path=src)
    blob = open(out, "rb").read()
    assert b"alice" not in blob and b"Pictures" not in blob
    assert b"IMG_1.jpg" in blob  # the filename itself is fine and useful
