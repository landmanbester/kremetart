"""Independent geodetic truth + simulation helpers for accuracy verification (test-only).

Truth antenna ECEF comes from pyproj/PROJ (WGS84) -- a code path independent of both the kremetart
reader (hand-rolled transform) and tart2ms (mean-Earth-radius offset_by). See
docs/superpowers/specs/2026-06-15-accuracy-verification-design.md.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

LIGHTSPEED = 299792458.0


def enu_to_ecef_truth(enu, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Independent ENU->ECEF via PROJ topocentric (WGS84).

    Args:
        enu: ``(n, 3)`` East/North/Up offsets (m) relative to the site.
        lat_deg, lon_deg, alt_m: site geodetic origin.

    Returns:
        ``(n, 3)`` geocentric ECEF positions (m).
    """
    from pyproj import Transformer

    enu = np.asarray(enu, dtype=np.float64)
    # Inverse PROJ topocentric: topocentric ENU -> geocentric ECEF (the forward direction maps
    # geocentric XYZ -> ENU). from_crs(topocentric, geocentric) trips a units mismatch, so build
    # the pipeline explicitly.
    pipe = f"+proj=pipeline +step +inv +proj=topocentric +ellps=WGS84 +lon_0={lon_deg} +lat_0={lat_deg} +h_0={alt_m}"
    tr = Transformer.from_pipeline(pipe)
    x, y, z = tr.transform(enu[:, 0], enu[:, 1], enu[:, 2])
    return np.stack([x, y, z], axis=1)


def baselines_from_positions(positions, ant1_idx, ant2_idx) -> np.ndarray:
    """Baseline vectors ``pos[ant1] - pos[ant2]`` -> ``(nbl, 3)``."""
    positions = np.asarray(positions)
    return positions[np.asarray(ant1_idx)] - positions[np.asarray(ant2_idx)]


def antenna_ecef(antenna_xds) -> np.ndarray:
    """ANTENNA_POSITION (ECEF, m) as ``(n_ant, 3)`` in antenna-index order."""
    return np.asarray(antenna_xds.ANTENNA_POSITION.values, dtype=np.float64)


def antenna_enu_and_site(partition):
    """Return (enu (n_ant,3), lat_deg, lon_deg, alt_m) from a kremetart partition node."""
    ant = partition["antenna_xds"].to_dataset(inherit=False)
    enu = np.asarray(ant.ANTENNA_POSITION_ENU.values, dtype=np.float64)
    info = partition.ds.attrs["observation_info"]
    return enu, info["site_latitude_deg"], info["site_longitude_deg"], info["site_altitude_m"]


def baseline_index_arrays(partition):
    """(ant1_idx, ant2_idx) mapping each baseline to antenna indices, in the partition's order."""
    ant = partition["antenna_xds"].to_dataset(inherit=False)
    names = list(ant.antenna_name.values)
    index = {name: i for i, name in enumerate(names)}
    a1 = np.array([index[n] for n in partition.ds.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in partition.ds.baseline_antenna2_name.values])
    return a1, a2


def source_svec(ra, dec) -> np.ndarray:
    """ICRS unit vectors (n, 3) for ra/dec arrays in radians."""
    ra = np.atleast_1d(np.asarray(ra, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec, dtype=np.float64))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def sources_spanning_zenith(times, lat_deg, lon_deg, alt_m, els_deg, az_deg=0.0):
    """ICRS (ra, dec) radians for sources at given elevations (deg) at the mid timestamp."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    times = np.asarray(times)
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    tmid = Time(times[times.size // 2], format="unix", scale="utc")
    els = np.atleast_1d(np.asarray(els_deg, dtype=np.float64))
    aa = AltAz(az=np.full(els.shape, az_deg) * u.deg, alt=els * u.deg, obstime=tmid, location=loc)
    icrs = SkyCoord(aa).icrs
    return np.atleast_1d(icrs.ra.rad), np.atleast_1d(icrs.dec.rad)


def simulate_visibilities(fluxes, svec, ecef_baselines, times, freqs, *, xp: ModuleType = np):
    """Truth visibilities V_pq(t) = sum_s f_s exp(2pi i (nu/c) b_pq(t).s_s), shape (n_time, nbl, nchan).

    Uses the shipped forward model with the shared C(t); ``ecef_baselines`` (nbl,3) are the ITRS
    baseline vectors whose accuracy is under test.
    """
    from kremetart.utils.healpix_dft import dft_forward, equatorial_baselines

    fluxes = xp.asarray(fluxes)
    svec = xp.asarray(svec)
    b_rot = equatorial_baselines(np.asarray(ecef_baselines), np.asarray(times), xp=xp)  # (nt, nbl, 3)
    nt, nbl = b_rot.shape[0], b_rot.shape[1]
    nchan = np.asarray(freqs).shape[0]
    vis = xp.zeros((nt, nbl, nchan), dtype=xp.complex128)
    for t in range(nt):
        vis[t] = dft_forward(fluxes, b_rot[t], svec, freqs, xp=xp)
    return vis


def angular_offset(a, b) -> float:
    """Angle (radians) between two unit vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def recovered_direction_and_flux(dmap, pix_vec, nside, *, nest: bool = True, near=None, search_radius_deg=None):
    """Flux-weighted centroid direction (unit vector) and peak flux (Jy) of a dirty map.

    Centroids the positive pixels within ~2.5 pixel-radii of the peak so a sub-pixel source is
    localised below the pixel scale.

    A coplanar array (TART has Up == 0) gives a dirty map that is mirror-symmetric about the array
    plane -- every source has an equal-amplitude reflection below the horizon. For a *known* injected
    source, pass ``near`` (a unit vector) and ``search_radius_deg`` to restrict the peak search to
    that hemisphere/neighbourhood, removing the mirror ambiguity.
    """
    import healpy as hp

    dmap = np.asarray(dmap)
    pix_vec = np.asarray(pix_vec)
    if near is not None and search_radius_deg is not None:
        disc0 = hp.query_disc(nside, np.asarray(near, dtype=np.float64), np.radians(search_radius_deg), nest=nest)
        peak = int(disc0[np.argmax(dmap[disc0])])
    else:
        peak = int(np.argmax(dmap))
    disc = hp.query_disc(nside, pix_vec[peak], 2.5 * hp.nside2resol(nside), nest=nest)
    w = np.clip(dmap[disc], 0.0, None)
    centroid = (w[:, None] * pix_vec[disc]).sum(axis=0)
    centroid /= np.linalg.norm(centroid)
    return centroid, float(dmap[peak])


def analytic_offset(b_rec, b_truth, svec) -> float:
    """Predicted peak offset (radians) when imaging truth data with b_rec instead of b_truth.

    Least-squares stationary-phase: solve for the tangent-plane shift delta minimising
    || (b_truth - b_rec).s - b_rec.delta ||. ``b_rec``/``b_truth`` are the (nbl,3) rotated baselines.
    """
    s = np.asarray(svec, dtype=np.float64)
    s = s / np.linalg.norm(s)
    e1 = np.cross(s, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(s, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    b_rec = np.asarray(b_rec, dtype=np.float64)
    extra = (np.asarray(b_truth, dtype=np.float64) - b_rec) @ s  # (nbl,)
    design = np.stack([b_rec @ e1, b_rec @ e2], axis=1)  # (nbl, 2)
    coef, *_ = np.linalg.lstsq(design, extra, rcond=None)
    return float(np.hypot(coef[0], coef[1]))
