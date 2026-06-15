"""Unit tests for the accuracy-verification helpers."""

import numpy as np
import pytest

pyproj = pytest.importorskip("pyproj")

from tests.accuracy_helpers import (  # noqa: E402  (after importorskip)
    baselines_from_positions,
    enu_to_ecef_truth,
    simulate_visibilities,
    source_svec,
    sources_spanning_zenith,
)

SITE = dict(lat_deg=-20.2587508, lon_deg=57.7591989, alt_m=20.0)


def test_enu_origin_maps_to_site_ecef():
    """ENU (0,0,0) maps to the WGS84 site ECEF from the independent EPSG:4979->4978 path."""
    from pyproj import Transformer

    site = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(
        SITE["lon_deg"], SITE["lat_deg"], SITE["alt_m"]
    )
    got = enu_to_ecef_truth(np.zeros((1, 3)), **SITE)[0]
    np.testing.assert_allclose(got, site, atol=1e-3)


def test_enu_offset_preserves_distance():
    """A 3.4 m ENU offset produces a 3.4 m ECEF displacement (rotation is rigid)."""
    pts = enu_to_ecef_truth(np.array([[0.0, 0, 0], [3.4, 0, 0], [0, 3.4, 0]]), **SITE)
    np.testing.assert_allclose(np.linalg.norm(pts[1] - pts[0]), 3.4, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(pts[2] - pts[0]), 3.4, atol=1e-6)


def test_baselines_from_positions():
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 2, 0]])
    a1 = np.array([0, 0, 1])
    a2 = np.array([1, 2, 2])
    bl = baselines_from_positions(pos, a1, a2)
    np.testing.assert_allclose(bl, np.array([[-1, 0, 0], [0, -2, 0], [1, -2, 0]]))


def test_source_svec_unit_and_value():
    np.testing.assert_allclose(source_svec([0.0], [0.0])[0], [1.0, 0.0, 0.0], atol=1e-12)
    v = source_svec([0.3, 1.2], [-0.2, 0.4])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)


def test_sources_spanning_zenith_roundtrip():
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    times = np.array([1.6e9, 1.6e9 + 60, 1.6e9 + 120])
    els = np.array([20.0, 50.0, 80.0])
    ra, dec = sources_spanning_zenith(times, **SITE, els_deg=els)
    loc = EarthLocation(lat=SITE["lat_deg"] * u.deg, lon=SITE["lon_deg"] * u.deg, height=SITE["alt_m"] * u.m)
    tmid = Time(times[1], format="unix", scale="utc")
    back = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs").transform_to(AltAz(obstime=tmid, location=loc))
    np.testing.assert_allclose(np.sort(back.alt.deg), np.sort(els), atol=1e-3)


def test_simulate_visibilities_point_source_amplitude():
    """A single point source of flux f gives |V| == f on every baseline (|fringe| = 1)."""
    rng = np.random.default_rng(7)
    ecef_bl = rng.standard_normal((10, 3)) * 2.0
    vis = simulate_visibilities(
        np.array([4.0]), source_svec([0.5], [-0.3]), ecef_bl, np.array([1.6e9]), np.array([1.575e9])
    )
    assert vis.shape == (1, 10, 1)
    np.testing.assert_allclose(np.abs(vis), 4.0, atol=1e-9)
