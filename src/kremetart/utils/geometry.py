"""Host-side sky geometry for imaging: ICRS<->ITRS rotations, zenith directions, rotated baselines.

Computed on the host with pyerfa via :mod:`tart_msv4.geometry` (``O(n_time)``); the per-snapshot
results (rotated baselines, boresight) are handed to the device-side maths in
:mod:`kremetart.imaging.healpix`, the same host/device split as :mod:`kremetart.utils.rephasing`.
Only numpy and pyerfa are needed: tart-msv4 >= 0.1.2 no longer imports its xarray backend on package import.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
from tart_msv4 import geometry as tart_geometry


def _icrs_to_itrs_matrices(times: np.ndarray) -> np.ndarray:
    """Per-timestamp ICRS->ITRS rotation matrices R(t), shape (n_time, 3, 3) (host, O(n_time)).

    Column ``i`` of ``R(t)`` is the ITRS image of the ``i``-th ICRS axis, so
    ``s_itrs(t) = R(t) @ s_icrs``. Computed with pyerfa via tart-msv4 (IAU 2006/2000A ``c2t06a``):
    frame bias, precession, nutation and Earth rotation, with UT1 = UTC and polar motion neglected.

    This is a *pure rotation* (the design's ``C(t)`` = latitude tilt then rotation about the polar
    axis by LST). It omits stellar aberration -- the non-rotational ICRS<->GCRS term (~20 arcsec), which
    no single matrix can capture -- negligible against the ~0.9 deg HEALPix pixel (<= 0.33 mm on
    TART's longest baseline versus astropy's full direction transform).
    """
    return tart_geometry.celestial_to_terrestrial(np.atleast_1d(np.asarray(times, dtype=np.float64)))


def zenith_icrs_vectors(times, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Instantaneous-zenith ICRS unit vectors at each timestamp (host, pyerfa; ``O(n_time)``).

    These are the antenna-boresight directions for the Airy primary beam: the site local zenith
    expressed as a unit vector in the same equatorial ICRS frame as :func:`make_pixel_grid`, so that
    ``pix_vec @ boresight`` is the cosine of each pixel's zenith angle. Precomputed per frame by the
    host prepare-step and applied on the GPU by the imager, mirroring the ``b_rot`` host/device split.
    Uses the same zenith (ra, dec) as tart-msv4's ``FIELD_PHASE_CENTER_DIRECTION`` (pyerfa ``atoc13``,
    UT1 = UTC), which agrees with astropy's ``AltAz -> ICRS`` to < 0.5 arcsec.

    Args:
        times: ``(n_time,)`` unix-second timestamps.
        lat_deg: site geodetic latitude in degrees.
        lon_deg: site longitude in degrees.
        alt_m: site altitude in metres.

    Returns:
        ``(n_time, 3)`` array of ICRS unit vectors of the site zenith.
    """
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    ra, dec = tart_geometry.zenith_radec(times, lat_deg, lon_deg, alt_m)
    return radec_to_unit_vectors(ra, dec)


def radec_to_unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Return ``(n, 3)`` ICRS unit vectors for (ra, dec) in radians."""
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def equatorial_baselines(itrs_baselines, times, *, backend: str = "astropy", xp: ModuleType = np):
    """Rotate fixed ITRS baselines into the equatorial frame for each timestamp.

    Args:
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors (e.g. from rephasing's ``itrs_baselines``).
        times: ``(n_time,)`` unix-second timestamps.
        backend: ``"astropy"`` (host-side rotation matrices; computed with pyerfa via tart-msv4 since the
            port, the name is kept for compatibility) or ``"native"`` (GPU polynomial; later phase).
        xp: Array module for the returned array.

    Returns:
        ``(n_time, nbl, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
    """
    if backend == "astropy":
        b = np.asarray(itrs_baselines)
        rot = _icrs_to_itrs_matrices(times)  # (n_time, 3, 3)
        b_rot = np.einsum("bi,tik->tbk", b, rot)  # b_itrs @ R(t)
        return xp.asarray(b_rot)
    if backend == "native":
        raise NotImplementedError("GPU-native C(t) backend is a later phase; use backend='astropy'.")
    raise ValueError(f"unknown backend {backend!r}")
