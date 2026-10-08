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

from openraw import OpenRawPipeline, PipelineConfig
from openraw.tonecurve import srgb_to_linear, linear_to_srgb
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
    pipeline = OpenRawPipeline()
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
    pipeline = OpenRawPipeline()
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
    pipeline = OpenRawPipeline()
    pipeline.run_to_dng(jpeg_path, out)

    from openraw.decode import load_jpeg

    d = load_jpeg(jpeg_path)
    row_u8 = (d.rgb[100, :, 0] * 255).astype(np.uint8)
    levels_8bit = len(np.unique(row_u8))

    arr16 = main_array(out)
    row16 = arr16[100, :, 0]
    levels_16bit = len(np.unique(row16))

    assert levels_16bit > levels_8bit


def test_dng_default_is_ljpeg_and_much_smaller_than_uncompressed(jpeg_path, tmp_path):
    """
    User report: ~108MB files made Luminar sluggish on mobile. Default is now
    lossless-JPEG tiles + 12-bit LinearizationTable (verified in Adobe's DNG
    SDK and libraw -- see test_exif_and_quality.py). Pins: default is
    meaningfully smaller than uncompressed, 16-bit LJPEG is bit-exact, and
    the 12-bit default stays within a small fraction of a source step.
    """
    from openraw.dng_writer import write_linear_dng
    from openraw.tonecurve import linear_to_srgb
    from .helpers import decoded_linear

    rgb16 = OpenRawPipeline().run(jpeg_path).rgb16
    none_p, l16_p, dflt_p = (str(tmp_path / n) for n in ("none.dng", "l16.dng", "default.dng"))
    write_linear_dng(none_p, rgb16, compression="none")
    write_linear_dng(l16_p, rgb16, compression="ljpeg", bit_depth=16)
    write_linear_dng(dflt_p, rgb16)

    assert np.array_equal(decoded_linear(l16_p), rgb16)  # truly lossless
    assert os.path.getsize(dflt_p) < 0.75 * os.path.getsize(none_p)
    to_steps = lambda a: linear_to_srgb(a.astype(np.float32) / 65535) * 255
    assert np.abs(to_steps(decoded_linear(dflt_p)) - to_steps(rgb16)).max() <= 0.1


def test_tonecurve_roundtrip_is_near_exact():
    x = np.linspace(0, 1, 1000).astype(np.float32)
    rt = linear_to_srgb(srgb_to_linear(x))
    assert np.allclose(x, rt, atol=1e-4)


def test_experimental_flags_do_not_crash(jpeg_path, tmp_path):
    out = str(tmp_path / "out.dng")
    config = PipelineConfig(generic_s_curve_strength=0.3, experimental_gamut_widen=0.4)
    pipeline = OpenRawPipeline(config)
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
    pipeline = OpenRawPipeline(config)
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
    pipeline = OpenRawPipeline(config)
    result = pipeline.run_to_dng(jpeg_path, out)

    assert os.path.exists(out)
    assert np.isfinite(result.linear_rgb).all()
    assert 0.0 < result.linear_rgb.mean() < 1.0  # not degenerate (all-0 / all-1)
    assert result.linear_rgb.std() > 1e-4  # not a flat/constant output

    with tifffile.TiffFile(out) as tf:
        page = main_page(tf)
        assert page.photometric == 34892
        assert page.dtype == np.uint16


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIKON = os.path.join(ROOT, "pretrained", "nikon.pth")


@pytest.mark.skipif(not os.path.exists(NIKON), reason="pretrained/nikon.pth not present")
def test_invisp_checkpoint_accepts_your_own_training_outputs(tmp_path):
    """--invisp-checkpoint: a weights-only file (best.pth / latest.pth) and
    train.py's full-state latest_state.pth must give exactly the same result
    as the same weights loaded via the upstream camera name."""
    torch = pytest.importorskip("torch")
    from openraw.invisp_bridge import reconstruct_pseudo_raw
    sd = torch.load(NIKON, map_location="cpu")
    weights, state = str(tmp_path / "best.pth"), str(tmp_path / "latest_state.pth")
    torch.save(sd, weights)
    torch.save({"net": sd, "optimizer": {"state": {}, "param_groups": []}, "scheduler": {},
                "epoch": 3, "step": 99, "best_raw_psnr": 39.0}, state)
    img = np.random.default_rng(0).random((32, 48, 3)).astype(np.float32)
    ref = reconstruct_pseudo_raw(img, camera="NIKON_D700", pretrained_dir=os.path.join(ROOT, "pretrained"))
    assert np.array_equal(reconstruct_pseudo_raw(img, checkpoint=weights), ref)
    assert np.array_equal(reconstruct_pseudo_raw(img, checkpoint=state), ref)


@pytest.mark.skipif(not os.path.exists(NIKON), reason="pretrained/nikon.pth not present")
def test_invisp_checkpoint_that_does_not_fit_is_a_clear_error(tmp_path):
    """Upstream loaded with strict=False, so a non-matching file silently loaded
    NOTHING (0 of 280 weights from a full-state dict), leaving a random network."""
    torch = pytest.importorskip("torch")
    from openraw.invisp_bridge import reconstruct_pseudo_raw
    bad = str(tmp_path / "bad.pth")
    torch.save({"model": torch.load(NIKON, map_location="cpu")}, bad)  # wrapped under an unknown key
    with pytest.raises(ValueError, match="weights missing"):
        reconstruct_pseudo_raw(np.zeros((8, 8, 3), np.float32), checkpoint=bad)
    with pytest.raises(FileNotFoundError):
        reconstruct_pseudo_raw(np.zeros((8, 8, 3), np.float32), checkpoint=str(tmp_path / "nope.pth"))


@pytest.mark.skipif(not os.path.exists(NIKON), reason="pretrained/nikon.pth not present")
def test_cli_invisp_checkpoint_flag(jpeg_path, tmp_path):
    pytest.importorskip("torch")
    import shutil
    from openraw.cli import main
    ck = str(tmp_path / "best.pth")
    shutil.copy(NIKON, ck)
    out = str(tmp_path / "out.dng")
    assert main([jpeg_path, "-o", out, "--invisp-checkpoint", ck, "-q"]) == 0  # implies --invisp
    assert os.path.getsize(out) > 0
