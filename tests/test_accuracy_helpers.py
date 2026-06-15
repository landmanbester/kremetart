"""Unit tests for the accuracy-verification helpers."""

import numpy as np
import pytest

pyproj = pytest.importorskip("pyproj")

from tests.accuracy_helpers import baselines_from_positions, enu_to_ecef_truth  # noqa: E402  (after importorskip)

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
