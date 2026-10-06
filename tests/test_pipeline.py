"""
Minimal sanity tests. These check that the pipeline runs end-to-end,
produces a readable DNG, preserves hue identity on saturated patches
(regression test for the RGB<->YCrCb scale bug caught during POC
development), and genuinely increases tonal resolution on a gradient
(regression test for "did dithering/debanding actually do anything").

Run with: pytest tests/
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import tifffile
from PIL import Image

from pseudoraw import PseudoRawPipeline, PipelineConfig
from pseudoraw.tonecurve import srgb_to_linear, linear_to_srgb
from .helpers import main_page, main_array


def _make_test_jpeg(path: str, quality: int = 60) -> None:
    h, w = 128, 192
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x / w, y / h, np.full((h, w), 0.5)], axis=-1)
    img[20:60, 20:60] = [0.9, 0.1, 0.1]
    img[20:60, 80:120] = [0.1, 0.9, 0.1]
    img[20:60, 140:180] = [0.1, 0.1, 0.9]
    arr8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr8, "RGB").save(path, quality=quality)


@pytest.fixture
def jpeg_path(tmp_path):
    p = str(tmp_path / "test.jpg")
    _make_test_jpeg(p)
    return p


def test_pipeline_runs_and_writes_valid_dng(jpeg_path, tmp_path):
    out = str(tmp_path / "out.dng")
    pipeline = PseudoRawPipeline()
    result = pipeline.run_to_dng(jpeg_path, out)

    assert os.path.exists(out)
    with tifffile.TiffFile(out) as tf:
        page = main_page(tf)
        assert page.photometric == 34892  # LinearRaw
        assert page.dtype == np.uint16
        assert page.shape == (128, 192, 3)
    assert result.rgb16.dtype == np.uint16


def test_saturated_patches_keep_their_hue(jpeg_path, tmp_path):
    """Regression test: a prior bug in chroma.py fed 0-255-scaled float
    data into cv2's YCrCb conversion, which expects 0-1 for float32 and
    silently desaturated everything. This checks hue identity survives."""
    out = str(tmp_path / "out.dng")
    pipeline = PseudoRawPipeline()
    pipeline.run_to_dng(jpeg_path, out)

    arr = main_array(out).astype(np.float32) / 65535.0
    preview = linear_to_srgb(arr)

    red_patch = preview[40, 40]
    green_patch = preview[40, 100]
    blue_patch = preview[40, 160]

    assert red_patch[0] > red_patch[1] and red_patch[0] > red_patch[2]
    assert green_patch[1] > green_patch[0] and green_patch[1] > green_patch[2]
    assert blue_patch[2] > blue_patch[0] and blue_patch[2] > blue_patch[1]


def test_bitdepth_expansion_increases_tonal_resolution(jpeg_path, tmp_path):
    """The whole point of dithering: more distinguishable levels across a
    gradient than the 8-bit JPEG source had, without fabricating new
    structure (this just confirms the mechanism engages, not image
    quality, which isn't something a unit test can judge)."""
    out = str(tmp_path / "out.dng")
    pipeline = PseudoRawPipeline()
    pipeline.run_to_dng(jpeg_path, out)

    from pseudoraw.decode import load_jpeg

    d = load_jpeg(jpeg_path)
    row_u8 = (d.rgb[100, :, 0] * 255).astype(np.uint8)
    levels_8bit = len(np.unique(row_u8))

    arr16 = main_array(out)
    row16 = arr16[100, :, 0]
    levels_16bit = len(np.unique(row16))

    assert levels_16bit > levels_8bit


def test_dng_default_is_uncompressed_and_compress_opt_in_is_smaller(jpeg_path, tmp_path):
    """
    Corrected regression test, superseding an earlier wrong version of
    this test that asserted compress=True was the (correct) default.
    It was NOT correct: real libraw testing (via rawpy) found Deflate-
    compressed output unreadable by libraw-based tools (darktable,
    RawTherapee, etc) -- see dng_writer.py's write_linear_dng docstring
    and test_exif_and_quality.py's test_dng_default_settings_actually_open_in_real_libraw,
    which is the test that actually catches that class of bug (this one
    only checks file size and tifffile-self-consistency, both necessary
    but not sufficient, which is exactly how the wrong default got
    shipped in the first place).

    What THIS test still correctly covers: compress=True, when a caller
    explicitly opts into it, really is smaller and really is still
    lossless -- it's just not the default anymore.
    """
    from pseudoraw.dng_writer import write_linear_dng
    from .helpers import main_page as _main_page, main_array as _main_array

    pipeline = PseudoRawPipeline()
    result = pipeline.run(jpeg_path)

    compressed_path = str(tmp_path / "compressed.dng")
    uncompressed_path = str(tmp_path / "uncompressed.dng")
    write_linear_dng(compressed_path, result.rgb16, compress=True)
    write_linear_dng(uncompressed_path, result.rgb16, compress=False)

    compressed_size = os.path.getsize(compressed_path)
    uncompressed_size = os.path.getsize(uncompressed_path)

    assert compressed_size < uncompressed_size

    with tifffile.TiffFile(compressed_path) as tf:
        page = _main_page(tf)
        assert page.photometric == 34892
    arr = _main_array(compressed_path)
    assert np.array_equal(arr, result.rgb16)  # still lossless when opted into

    # Default (no compress= passed) must match compress=False.
    default_path = str(tmp_path / "default.dng")
    write_linear_dng(default_path, result.rgb16)
    assert os.path.getsize(default_path) == uncompressed_size


def test_tonecurve_roundtrip_is_near_exact():
    x = np.linspace(0, 1, 1000).astype(np.float32)
    rt = linear_to_srgb(srgb_to_linear(x))
    assert np.allclose(x, rt, atol=1e-4)


def test_experimental_flags_do_not_crash(jpeg_path, tmp_path):
    out = str(tmp_path / "out.dng")
    config = PipelineConfig(generic_s_curve_strength=0.3, experimental_gamut_widen=0.4)
    pipeline = PseudoRawPipeline(config)
    pipeline.run_to_dng(jpeg_path, out)
    assert os.path.exists(out)


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _invisp_checkpoint_available():
    import os
    return os.path.isfile(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pretrained", "nikon.pth"
    ))


@pytest.mark.skipif(_torch_available(), reason="only meaningful when torch is NOT installed")
def test_invisp_path_fails_with_clear_actionable_error_without_torch(jpeg_path, tmp_path):
    """Missing torch should fail loudly and clearly, not crash confusingly
    deep in the call stack. Skipped in environments where torch IS
    installed (see the paired test below for that case)."""
    out = str(tmp_path / "out.dng")
    config = PipelineConfig(use_invisp=True, invisp_camera="NIKON_D700")
    pipeline = PseudoRawPipeline(config)
    with pytest.raises(ImportError, match="PyTorch"):
        pipeline.run_to_dng(jpeg_path, out)


@pytest.mark.skipif(not _torch_available(), reason="needs torch installed")
@pytest.mark.skipif(not _invisp_checkpoint_available(), reason="needs pretrained/nikon.pth")
def test_invisp_path_runs_real_network_and_writes_valid_dng(jpeg_path, tmp_path):
    """
    First real end-to-end run of the actual vendored InvISP network
    against the actual nikon.pth checkpoint (not a mock). Confirms:
    construction succeeds (catches the torch.qr compatibility patch
    regressing), inference produces finite, non-degenerate output (not
    NaN/all-zero/all-saturated, which untrained-looking or broken
    weight-loading would produce), and the result still writes through
    the same compressed-DNG path as the classical pipeline.
    """
    import tifffile

    out = str(tmp_path / "out.dng")
    config = PipelineConfig(use_invisp=True, invisp_camera="NIKON_D700")
    pipeline = PseudoRawPipeline(config)
    result = pipeline.run_to_dng(jpeg_path, out)

    assert os.path.exists(out)
    assert np.isfinite(result.linear_rgb).all()
    assert 0.0 < result.linear_rgb.mean() < 1.0  # not degenerate (all-0 / all-1)
    assert result.linear_rgb.std() > 1e-4  # not a flat/constant output

    with tifffile.TiffFile(out) as tf:
        page = main_page(tf)
        assert page.photometric == 34892
        assert page.dtype == np.uint16
