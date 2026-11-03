"""HEALPix pixel geometry as unit vectors: pixel centres, pixel corners, and sky positions.

Built on astropy-healpix, which ships wheels for Linux, macOS and Windows (healpy has none for Windows).
Agrees with healpy's ``pix2vec``/``boundaries``/``ang2vec`` to rounding (about 1e-15). Each function
runs once per sky grid or viewer stream, never per snapshot.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy_healpix import boundaries_lonlat, healpix_to_lonlat


def _unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Unit vectors ``(..., 3)`` from longitude and latitude in radians."""
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1)


def _order(nest: bool) -> str:
    return "nested" if nest else "ring"


def pixel_centres(nside: int, *, nest: bool) -> np.ndarray:
    """``(npix, 3)`` unit vectors of every pixel centre, in pixel-index order."""
    lon, lat = healpix_to_lonlat(np.arange(12 * nside * nside), nside, order=_order(nest))
    return _unit_vectors(lon.to_value(u.rad), lat.to_value(u.rad))


def pixel_corners(nside: int, *, nest: bool) -> np.ndarray:
    """``(npix, 4, 3)`` unit vectors of every pixel's four corners (healpy's ``boundaries`` order)."""
    lon, lat = boundaries_lonlat(np.arange(12 * nside * nside), 1, nside, order=_order(nest))  # (npix, 4)
    return _unit_vectors(lon.to_value(u.rad), lat.to_value(u.rad))


def lonlat_to_vec(lon_deg: float, lat_deg: float) -> np.ndarray:
    """``(3,)`` unit vector of a sky position given in degrees (e.g. ICRS RA and Dec)."""
    return _unit_vectors(np.radians(lon_deg), np.radians(lat_deg))
