"""
Tests for pseudoraw/ml/ -- the InvISP-inspired invertible network
groundwork. These exist to prove the one claim this scaffold is actually
allowed to make: forward() and inverse() are exact inverses of each
other, independent of the (currently untrained/random) subnet weights.
They do NOT test output quality/usefulness, which requires training.
"""

from __future__ import annotations

import numpy as np
import pytest

from pseudoraw.ml.haar import forward_haar, inverse_haar, _H
from pseudoraw.ml.coupling import AffineCouplingBlock
from pseudoraw.ml.invnet import InvISPLiteNet


def test_haar_matrix_is_orthonormal_and_symmetric():
    # The whole "apply the same matrix for forward and inverse" trick
    # depends on these two properties -- pin them down explicitly.
    assert np.allclose(_H, _H.T)
    assert np.allclose(_H @ _H.T, np.eye(4))


def test_haar_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 24, 3))
    y = forward_haar(x)
    assert y.shape == (8, 12, 12)
    x_rt = inverse_haar(y, channels=3)
    assert np.allclose(x, x_rt, atol=1e-10)


def test_haar_rejects_odd_dimensions():
    with pytest.raises(ValueError):
        forward_haar(np.zeros((15, 16, 3)))


def test_coupling_block_roundtrip_is_exact():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 8, 12))
    for swap in (False, True):
        block = AffineCouplingBlock(channels=12, seed=5, swap=swap)
        y = block.forward(x)
        x_rt = block.inverse(y)
        assert np.allclose(x, x_rt, atol=1e-9)


def test_invispliten_full_roundtrip_is_exact():
    rng = np.random.default_rng(2)
    x = rng.random((32, 48, 3))  # plausible image-like range [0, 1)
    net = InvISPLiteNet(in_channels=3, n_blocks=6, seed=42)

    z = net.forward(x)
    assert z.shape == (16, 24, 12)

    x_rt = net.inverse(z)
    assert x_rt.shape == x.shape
    assert np.allclose(x, x_rt, atol=1e-8)


def test_invispliten_roundtrip_holds_for_varied_block_counts_and_seeds():
    rng = np.random.default_rng(3)
    x = rng.random((16, 16, 3))
    for n_blocks in (1, 2, 5, 8):
        for seed in (0, 7, 123):
            net = InvISPLiteNet(in_channels=3, n_blocks=n_blocks, seed=seed)
            x_rt = net.inverse(net.forward(x))
            assert np.allclose(x, x_rt, atol=1e-8), f"failed at n_blocks={n_blocks}, seed={seed}"
