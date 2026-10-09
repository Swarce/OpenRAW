"""tools/raw_pair_eval.py: pairing, alignment and metrics (the full run needs
real RAW+JPEG pairs and is exercised by hand)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import raw_pair_eval as rpe  # noqa: E402


def test_pairs_match_jpegs_to_raws_case_insensitively(tmp_path):
    for n in ("DSC1.JPG", "DSC1.ARW", "dsc2.jpg", "dsc2.NEF", "lonely.jpg", "only.cr2"):
        (tmp_path / n).write_bytes(b"x")
    got = [(s, j.name, r.name) for s, j, r in rpe.find_pairs(tmp_path)]
    assert got == [("DSC1", "DSC1.JPG", "DSC1.ARW"), ("dsc2", "dsc2.jpg", "dsc2.NEF")]


def _scene(h, w, seed=0):
    rng = np.random.default_rng(seed)
    low = rng.random((h // 16 + 2, w // 16 + 2, 3)).astype(np.float32)
    img = np.kron(low, np.ones((16, 16, 1), np.float32))[:h, :w]
    return np.clip(img + 0.05 * rng.standard_normal((h, w, 3)).astype(np.float32), 0, 1) * 0.8


@pytest.mark.parametrize("dy,dx", [(12, 12), (0, 0), (-7, 30)])
def test_align_finds_the_crop_offset(dy, dx):
    ref = _scene(600, 800)
    y0, x0 = 100 + dy, 120 + dx
    img = ref[y0:y0 + 400, x0:x0 + 500] * 0.6  # a darker crop, like a different exposure
    assert rpe.align(ref, img, prior=(100, 120)) == (y0, x0)
    a, b = rpe.overlap(ref, img, y0, x0)
    assert np.allclose(a * 0.6, b)


def test_gain_matching_removes_exposure_and_white_balance_only():
    ref = _scene(300, 300)
    img = ref * np.array([0.5, 0.7, 0.9], np.float32)
    m = rpe.metrics(ref, img)
    assert m["psnr"] < 30 and m["psnr_gain"] > 90  # pure scaling: fully explained by the gains
    assert np.allclose([m["gain_r"], m["gain_g"], m["gain_b"]], [2.0, 1 / 0.7, 1 / 0.9], rtol=1e-3)
    curved = np.power(ref, 0.8)  # a tone difference is not explained by gains
    assert rpe.metrics(ref, curved)["psnr_gain"] < 40


def test_render_applies_the_cfa_dngs_default_crop(tmp_path):
    """A CFA DNG carries a 4 px border for demosaicing, removed by DefaultCrop.
    render() must apply it, or CFA renders are scored 4 px out of alignment."""
    pytest.importorskip("rawpy")
    from openraw.dng_writer import write_linear_dng
    yy, xx = np.mgrid[0:300, 0:400].astype(np.float32)  # (libraw mis-renders very small CFA images)
    img = np.repeat((np.sin(xx / 7) * np.cos(yy / 5))[..., None], 3, axis=2)
    img = ((img * 0.4 + 0.5) * 30000).astype(np.uint16)  # smooth: a CFA round trip keeps it
    lin, cfa = tmp_path / "l.dng", tmp_path / "c.dng"
    write_linear_dng(str(lin), img, write_preview=False)
    write_linear_dng(str(cfa), img, layout="cfa", write_preview=False)
    a, b = rpe.render(lin), rpe.render(cfa)
    assert a.shape == b.shape == (300, 400, 3)
    assert np.abs(a - b)[4:-4, 4:-4].mean() < 0.01  # aligned: same pixels
