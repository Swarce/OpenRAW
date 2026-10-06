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

import numpy as np
import pytest
import tifffile
from PIL import Image
from PIL.ExifTags import Base, IFD

from pseudoraw.decode import load_jpeg
from pseudoraw.exif_transfer import extract_exif, build_dng_extratags
from pseudoraw.chroma import refine_chroma
from pseudoraw.dng_writer import write_linear_dng
from pseudoraw import PseudoRawPipeline, PipelineConfig

from .helpers import main_page, main_array

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
        tags = main_page(tf).tags
        assert tags[271].value == "FUJIFILM"
        assert tags[272].value == "X-T4"
        # UniqueCameraModel always stays the synthetic-sensor label
        # regardless of real Make/Model being present.
        assert tags[50708].value == "pseudoraw virtual sensor"
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

        arr = main_array(out)
        assert np.array_equal(arr, rgb16)


def test_dng_write_without_exif_fields_keeps_placeholder(tmp_path):
    rgb16 = (np.random.default_rng(0).random((16, 16, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, exif_fields=None)

    with tifffile.TiffFile(out) as tf:
        tags = main_page(tf).tags
        assert tags[271].value == "pseudoraw"
        assert tags[272].value == "pseudoraw-poc"
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
def test_dng_compress_true_is_known_incompatible_with_libraw(tmp_path):
    """
    Documents, with a real reproducible test (not just a comment), why
    compress=True is opt-in rather than the default: real libraw
    rejects it. If a future libraw/imagecodecs/tifffile version fixes
    this, this test will start failing its xfail and that's the signal
    to reconsider the default -- not a silent assumption either way.
    """
    rgb16 = (np.random.default_rng(0).random((64, 64, 3)) * 65535).astype(np.uint16)
    out = str(tmp_path / "out.dng")
    write_linear_dng(out, rgb16, compress=True)

    with pytest.raises(Exception):
        with rawpy.imread(out) as raw:
            raw.postprocess()


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

    pipeline = PseudoRawPipeline()
    result = pipeline.run_to_dng(p, out)

    assert result.decoded.is_chroma_subsampled is False
    with tifffile.TiffFile(out) as tf:
        assert main_page(tf).tags[271].value == "FUJIFILM"


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
    from pseudoraw.bitdepth import expand_to_16bit

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
