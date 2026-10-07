"""CFA (Bayer mosaic) DNG output: structure, crop, bit-exact compression, and
round-trip quality through a real demosaic engine (libraw)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from openraw import OpenRawPipeline
from openraw.dng_writer import CFA_PAD, write_linear_dng

from .helpers import ifd0_tags, main_page

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def rgb16():  # a real (small) photo through the real pipeline
    return OpenRawPipeline().run(os.path.join(ROOT, "examples", "test_source.jpg")).rgb16


@pytest.mark.parametrize("write_preview", [True, False])
@pytest.mark.parametrize("compression,bit_depth", [("none", 16), ("ljpeg", 16), ("ljpeg", 12)])
def test_cfa_structure(tmp_path, rgb16, write_preview, compression, bit_depth):
    out = str(tmp_path / "x.dng")
    write_linear_dng(out, rgb16, layout="cfa", write_preview=write_preview, compression=compression, bit_depth=bit_depth)
    h, w = rgb16.shape[:2]
    with tifffile.TiffFile(out) as tf:
        m = main_page(tf)
        t = {x.code: x.value for x in m.tags}
        assert m.photometric == 32803 and t[277] == 1               # CFA, 1 sample
        assert m.shape == (h + 2 * CFA_PAD, w + 2 * CFA_PAD)         # padded mosaic...
        assert tuple(t[50719]) == (CFA_PAD, CFA_PAD)                 # ...cropped back by DefaultCrop
        assert tuple(t[50720]) == (w, h)
        assert tuple(t[33421]) == (2, 2) and bytes(t[33422]) == bytes((0, 1, 1, 2))  # RGGB
        assert 50706 in ifd0_tags(tf)                                # DNGVersion in IFD0
        assert not {338, 530, 532} & set(t)


def test_cfa_is_much_smaller_than_linear(tmp_path, rgb16):
    lin, cfa = str(tmp_path / "l.dng"), str(tmp_path / "c.dng")
    write_linear_dng(lin, rgb16)
    write_linear_dng(cfa, rgb16, layout="cfa")
    assert os.path.getsize(cfa) < 0.6 * os.path.getsize(lin)


rawpy = pytest.importorskip("rawpy")


def _libraw(path, alg=None):
    with rawpy.imread(path) as r:
        s = r.sizes
        kw = dict(output_bps=16, no_auto_bright=True, gamma=(1, 1), use_camera_wb=True)
        if alg:
            kw["demosaic_algorithm"] = alg
        o = r.postprocess(**kw)
        if s.crop_width:  # apply DefaultCrop, which libraw reports but doesn't apply
            o = o[s.crop_top_margin:s.crop_top_margin + s.crop_height, s.crop_left_margin:s.crop_left_margin + s.crop_width]
        return o


def test_libraw_reports_the_crop_and_ljpeg_is_bit_exact(tmp_path, rgb16):
    a, b = str(tmp_path / "a.dng"), str(tmp_path / "b.dng")
    write_linear_dng(a, rgb16, layout="cfa", compression="none")
    write_linear_dng(b, rgb16, layout="cfa", compression="ljpeg", bit_depth=16)
    ra = _libraw(a)
    assert ra.shape[:2] == rgb16.shape[:2]  # crop reported exactly
    assert np.array_equal(ra, _libraw(b))


def test_cfa_round_trip_quality_through_a_real_demosaic(tmp_path, rgb16):
    """Measured ~48 dB on a real 18 MP photo (generally invisible above ~40 dB).
    The test image's top rows hold razor-sharp edges between pure red/green/blue
    patches -- the worst case for ANY Bayer sensor (one colour sample per pixel
    can't fully describe such an edge; real cameras share this limit). Natural
    content is held to a high floor; the pathological part to a lower one."""
    lin, cfa = str(tmp_path / "l.dng"), str(tmp_path / "c.dng")
    write_linear_dng(lin, rgb16, compression="none")
    write_linear_dng(cfa, rgb16, layout="cfa")

    def to8(x):
        x = x.astype(np.float32) / 65535
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055) * 255
    ref, out = to8(_libraw(lin)), to8(_libraw(cfa, rawpy.DemosaicAlgorithm.AHD))
    psnr = lambda a, b: 10 * np.log10(255 ** 2 / ((a - b) ** 2).mean())
    natural = psnr(out[200:], ref[200:])  # gradients + noise (measured 45.9 dB)
    saturated = psnr(out[:200], ref[:200])  # hard primary-colour edges (measured 35.1 dB)
    assert natural > 44, natural
    assert saturated > 33, saturated
