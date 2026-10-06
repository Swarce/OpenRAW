"""dataset/mosaic_store.py: the compact training-raw format must be lossless,
orientation-correct, and give exactly what full-frame demosaicing would."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("colour_demosaicing")
from dataset import mosaic_store as ms  # noqa: E402

SIZES = [(64, 96), (63, 95)]  # even and odd


@pytest.mark.parametrize("pattern", ms.PATTERNS)
@pytest.mark.parametrize("h,w", SIZES)
def test_pack_unpack_is_bit_exact(pattern, h, w):
    m = np.random.default_rng(0).integers(0, 4000, (h, w)).astype(np.uint16)
    b = io.BytesIO(); np.savez(b, **ms.pack(m, pattern, 4000, [2, 1, 1.5, 1])); b.seek(0)
    z = np.load(b)
    m2, p2 = ms.unpack(z)
    assert ms.is_mosaic(z) and p2 == pattern and np.array_equal(m, m2)


@pytest.mark.parametrize("pattern", ms.PATTERNS)
@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("flip,k", [(3, 2), (5, 1), (6, 3)])
def test_rotated_mosaic_demosaics_to_the_rotated_image(pattern, h, w, flip, k):
    m = np.random.default_rng(1).integers(0, 4000, (h, w)).astype(np.uint16)
    mr, pr = ms.rotate_mosaic(m, pattern, flip)
    want = np.rot90(ms.demosaic(m, pattern), k)
    assert np.allclose(ms.demosaic(mr, pr)[2:-2, 2:-2], want[2:-2, 2:-2], atol=1e-3)


@pytest.mark.parametrize("pattern", ms.PATTERNS)
@pytest.mark.parametrize("h,w", SIZES)
def test_region_demosaic_equals_full_frame_crop_even_at_borders(pattern, h, w):
    m = np.random.default_rng(2).integers(0, 4000, (h, w)).astype(np.uint16)
    full = ms.demosaic(m, pattern)
    for y, x in ((0, 0), (10, 20), (h - h % 2 - 20, w - w % 2 - 30)):
        y, x = y - y % 2, x - x % 2
        assert np.allclose(ms.demosaic_region(m, pattern, y, x, 20, 30), full[y:y + 20, x:x + 30], atol=1e-3)


@pytest.mark.parametrize("pattern", ms.PATTERNS)
@pytest.mark.parametrize("h,w", SIZES + [(400, 600)])
def test_remosaic_recovers_the_original_exactly_including_edges(pattern, h, w):
    m = np.random.default_rng(3).integers(0, 4000, (h, w)).astype(np.uint16)
    rm, rp = ms.remosaic(ms.demosaic(m, pattern))
    assert rp == pattern and np.array_equal(rm, m)
