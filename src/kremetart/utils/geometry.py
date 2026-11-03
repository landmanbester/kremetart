"""Host-side sky geometry: ICRS<->ITRS rotations, zenith directions, rotated baselines, phase direction.

Everything here is per run or per timestamp (``O(n_time)``) and runs on the host: pyerfa via
:mod:`tart_msv4.geometry` for the rotations, astropy for the run's common phase direction. The prepare step
stores the per-snapshot results (rotated baselines, boresight) for the device-side maths in
:mod:`kremetart.imaging`, which never imports this module.
"""

from __future__ import annotations

from types import ModuleType

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from tart_msv4 import geometry as tart_geometry
from tart_msv4.backend import read_metadata


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


def equatorial_baselines(itrs_baselines, times, *, xp: ModuleType = np):
    """Rotate fixed ITRS baselines into the equatorial frame for each timestamp.

    Args:
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors ``pos(ant1) - pos(ant2)``.
        times: ``(n_time,)`` unix-second timestamps.
        xp: Array module for the returned array.

    Returns:
        ``(n_time, nbl, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
    """
    b = np.asarray(itrs_baselines)
    rot = _icrs_to_itrs_matrices(times)  # (n_time, 3, 3)
    b_rot = np.einsum("bi,tik->tbk", b, rot)  # b_itrs @ R(t)
    return xp.asarray(b_rot)


def common_phase_direction(hdf_paths) -> tuple[float, float]:
    """Single shared ICRS phase direction: the local zenith RA/Dec at the global mid-time.

    Reads the first and last timestamps across all files, takes the midpoint, and converts the local
    zenith (AltAz alt=90 deg) at that time to ICRS. Reusable as the common field center for
    multi-TART mosaicking: compute once, hand the same value to every TART.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.

    Returns:
        ``(ra_deg, dec_deg)`` of the local zenith at the global mid-time, in ICRS.

    Raises:
        ValueError: if ``hdf_paths`` is empty.
    """

    t_lo = t_hi = None
    config = None
    for path in hdf_paths:
        meta = read_metadata(path)  # timestamps and site only; no MSv4 tree needed
        config = config or meta["config"]  # site from the first file; all inputs share one site
        lo, hi = float(meta["time"].min()), float(meta["time"].max())
        t_lo = lo if t_lo is None else min(t_lo, lo)
        t_hi = hi if t_hi is None else max(t_hi, hi)
    if config is None:
        raise ValueError("no HDF files provided")

    t_mid = 0.5 * (t_lo + t_hi)
    loc = EarthLocation(lat=config["lat"] * u.deg, lon=config["lon"] * u.deg, height=config["alt"] * u.m)
    aa = AltAz(az=0.0 * u.deg, alt=90.0 * u.deg, obstime=Time(t_mid, format="unix", scale="utc"), location=loc)
    icrs = SkyCoord(aa).icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)
