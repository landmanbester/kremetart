"""HEALPix geometry on astropy-healpix, checked for shape and against healpy (the reference) where installed."""

import numpy as np
import pytest

from kremetart.utils.healpix_geometry import lonlat_to_vec, pixel_centres, pixel_corners


@pytest.mark.parametrize("nest", [True, False], ids=["nested", "ring"])
def test_pixel_geometry_is_unit_vectors_of_the_right_shape(nest: bool) -> None:
    nside = 4
    centres, corners = pixel_centres(nside, nest=nest), pixel_corners(nside, nest=nest)
    assert centres.shape == (12 * nside**2, 3) and corners.shape == (12 * nside**2, 4, 3)
    np.testing.assert_allclose(np.linalg.norm(centres, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(corners, axis=-1), 1.0, atol=1e-12)


def test_lonlat_to_vec_places_the_poles_and_the_origin() -> None:
    np.testing.assert_allclose(lonlat_to_vec(0.0, 90.0), [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(lonlat_to_vec(0.0, 0.0), [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(lonlat_to_vec(90.0, 0.0), [0.0, 1.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("nest", [True, False], ids=["nested", "ring"])
@pytest.mark.parametrize("nside", [1, 8, 64])
def test_pixel_geometry_matches_healpy(nside: int, nest: bool) -> None:
    hp = pytest.importorskip("healpy")  # the reference implementation; no Windows wheels
    pix = np.arange(hp.nside2npix(nside))
    expected_centres = np.stack(hp.pix2vec(nside, pix, nest=nest), axis=-1)
    expected_corners = np.transpose(hp.boundaries(nside, pix, step=1, nest=nest), (0, 2, 1))
    np.testing.assert_allclose(pixel_centres(nside, nest=nest), expected_centres, rtol=0, atol=1e-14)
    np.testing.assert_allclose(pixel_corners(nside, nest=nest), expected_corners, rtol=0, atol=1e-14)


def test_lonlat_to_vec_matches_healpy() -> None:
    hp = pytest.importorskip("healpy")  # the reference implementation; no Windows wheels
    rng = np.random.default_rng(0)
    for lon, lat in zip(rng.uniform(0, 360, 20), rng.uniform(-90, 90, 20), strict=True):
        np.testing.assert_allclose(lonlat_to_vec(lon, lat), hp.ang2vec(lon, lat, lonlat=True), rtol=0, atol=1e-15)
