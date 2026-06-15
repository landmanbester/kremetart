"""Unit tests for the HEALPix gridless DFT operator (CPU, xp=numpy)."""

import numpy as np
import pytest

from kremetart.utils.healpix_dft import make_pixel_grid

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
