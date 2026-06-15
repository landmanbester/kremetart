"""Unit tests for the HEALPix gridless DFT operator (CPU, xp=numpy)."""

import numpy as np
import pytest

from kremetart.utils.healpix_dft import dft_adjoint, dft_forward, dirty_map, make_pixel_grid

LIGHTSPEED = 299792458.0


def test_make_pixel_grid_shape_and_unit_norm():
    nside = 8
    pix = make_pixel_grid(nside, xp=np)
    assert pix.shape == (12 * nside**2, 3)
    np.testing.assert_allclose(np.linalg.norm(pix, axis=1), 1.0, atol=1e-12)


def test_make_pixel_grid_nested_matches_healpy():
    hp = pytest.importorskip("healpy")
    nside = 4
    pix = make_pixel_grid(nside, xp=np)  # nest=True default
    expected = np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), axis=1)
    np.testing.assert_allclose(pix, expected, atol=1e-12)


def test_forward_matches_explicit_fringe():
    """forward of a single-pixel source is the geometric fringe on every row/channel."""
    rng = np.random.default_rng(1)
    pix = make_pixel_grid(4, xp=np)
    npix = pix.shape[0]
    baselines = rng.standard_normal((5, 3))
    freqs = np.array([1.40e9, 1.575e9])
    image = np.zeros(npix)
    image[3] = 2.0
    vis = dft_forward(image, baselines, pix, freqs, xp=np)
    assert vis.shape == (5, 2)
    g = baselines @ pix[3]  # (nrow,)
    expected = 2.0 * np.exp(2j * np.pi * (freqs[None, :] / LIGHTSPEED) * g[:, None])
    np.testing.assert_allclose(vis, expected, rtol=1e-12, atol=1e-12)


def test_forward_adjoint_are_hermitian_transposes():
    """<forward(image), data> == <image, adjoint(data)>  (the adjointness dot-product test)."""
    rng = np.random.default_rng(0)
    pix = make_pixel_grid(8, xp=np)
    npix = pix.shape[0]
    nrow, nchan = 12, 3
    baselines = rng.standard_normal((nrow, 3))
    freqs = np.array([1.40e9, 1.50e9, 1.575e9])
    image = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    data = rng.standard_normal((nrow, nchan)) + 1j * rng.standard_normal((nrow, nchan))
    lhs = np.vdot(dft_forward(image, baselines, pix, freqs, xp=np), data)
    rhs = np.vdot(image, dft_adjoint(data, baselines, pix, freqs, xp=np))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_dirty_map_recovers_point_source():
    """dirty_map of a forward-modelled single-pixel source peaks (value 1) at that pixel."""
    rng = np.random.default_rng(2)
    pix = make_pixel_grid(16, xp=np)
    npix = pix.shape[0]
    nrow = 300
    baselines = rng.standard_normal((nrow, 3)) * 2.0
    freqs = np.array([1.575e9])
    src = 1234
    image = np.zeros(npix)
    image[src] = 1.0
    vis = dft_forward(image, baselines, pix, freqs, xp=np)
    weights = np.ones((nrow, 1))
    dmap = dirty_map(vis, weights, baselines, pix, freqs, xp=np)
    assert dmap.shape == (npix,)
    assert dmap.dtype == np.float64
    assert int(np.argmax(dmap)) == src
    np.testing.assert_allclose(dmap[src], 1.0, atol=1e-12)
