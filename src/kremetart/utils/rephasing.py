"""Rephase TART visibilities onto a fixed celestial phase centre.

Raw TART visibilities are phased to the *instantaneous zenith* of each frame: the array is
zenith-pointing, does not fringe-stop, and is (very nearly) coplanar in the local horizontal
plane, so the natural phase centre drifts across the sky at the sidereal rate. For imaging and
detection on a sidereally fixed grid (see the design notes) we rephase every frame onto a single
fixed direction and recompute the UVW coordinates toward that direction.

This is the GPU-capable replacement for the casacore-based UVW synthesis in ``tart2ms``. The work
splits cleanly into two parts:

* **Per-frame celestial bookkeeping (host, ``O(n_time)``):** the source direction unit vector in
  the Earth-fixed ITRS frame, which folds in precession, nutation and Earth rotation. This is one
  small vector per timestamp, shared by every baseline, so it is computed with ``astropy`` and is
  never the bottleneck. (It could later be replaced by a GMST/precession polynomial for a
  dependency-free build; ``astropy`` is used here because it matches casacore to sub-mm.)
* **Per-(time, baseline, channel) arithmetic (device):** projecting the ITRS baselines onto the
  ``(u, v, w)`` axes and applying the fringe-rotation phasor. These are pure array operations
  performed through an injectable array module ``xp`` (``numpy`` or ``cupy``), so the same code
  runs on the GPU inside a Holoscan operator.

The visibility rephasor and the ``-1`` baseline phase-sign follow the NRAO/CASA convention so that
the output reproduces ``tart2ms --rephase`` (verified against a reference Measurement Set).
"""

from __future__ import annotations

from types import ModuleType

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from kremetart.utils import partition_datatree
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4


def midpoint_zenith(dt: xr.DataTree) -> tuple[float, float]:
    """Return the ICRS direction (ra, dec) of the zenith at the observation midpoint.

    This reproduces ``tart2ms``'s ``--rephase obs-midpoint`` phase centre and is the natural
    fixed direction for a short TART chunk.

    Args:
        dt: An MSv4 DataTree from :func:`kremetart.utils.read_tart_hdf.read_hdf_as_msv4`.

    Returns:
        ``(ra, dec)`` in radians (ICRS).
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    node = _partition(dt)
    times = node.ds.time.values
    loc = _site_location(node)
    tmid = Time(times[times.size // 2], format="unix", scale="utc")
    zen = SkyCoord(AltAz(az=0 * u.deg, alt=90 * u.deg, obstime=tmid, location=loc)).icrs
    return float(zen.ra.rad), float(zen.dec.rad)


def rephase_to_dir(
    dt: xr.DataTree,
    new_dir: tuple[float, float],
    *,
    xp: ModuleType = np,
    phasesign: int = -1,
) -> xr.DataTree:
    """Rephase a TART MSv4 DataTree onto a fixed celestial direction.

    Computes the current UVW (toward each frame's instantaneous zenith) and the new UVW (toward
    ``new_dir``), applies the fringe-rotation phasor to the visibilities, and returns an updated
    DataTree whose ``UVW``, ``VISIBILITY`` and field-and-source node reflect the new fixed phase
    centre.

    The heavy per-(time, baseline, channel) arithmetic is performed through ``xp``: pass
    ``xp=cupy`` to run on the GPU. Only the small per-frame source directions are computed on the
    host with ``astropy``.

    Note:
        The UVW coordinates are computed from the ITRF antenna positions stored in the input
        DataTree, which :func:`kremetart.utils.read_tart_hdf.read_hdf_as_msv4` derives from the
        local ENU positions via the exact tangent-plane ENU->ECEF transform at the site geodetic
        latitude. ``tart2ms`` instead places each antenna with a spherical ``astropy.offset_by``
        using a *mean* Earth radius (``R_earth`` ~ 6371 km) rather than the local WGS84 radius of
        curvature. That ~0.3% baseline-length convention difference makes our ``u``/``v`` differ
        from a ``tart2ms``-produced Measurement Set by ~1 cm at TART's longest (~3.4 m) baseline;
        ``w`` (toward the phase centre) is unaffected to ~1e-5 m. This is a position-convention
        difference, not a rephasing error -- our tangent-plane transform is the more accurate of
        the two -- and the visibility rephasor is unaffected because it scales as ``uvw`` times the
        tiny new-vs-old direction-cosine offset (verified to agree to ~1e-4).

    Args:
        dt: An MSv4 DataTree from :func:`kremetart.utils.read_tart_hdf.read_hdf_as_msv4`.
        new_dir: Target phase centre as ``(ra, dec)`` in radians (ICRS). Use
            :func:`midpoint_zenith` for the observation-midpoint zenith.
        xp: Array module for the device arithmetic (``numpy`` or ``cupy``). Defaults to ``numpy``.
        phasesign: Sign of the rephasor exponent; ``-1`` is the NRAO/CASA baseline convention
            (and what reproduces ``tart2ms``).

    Returns:
        A new ``xr.DataTree`` with rephased ``VISIBILITY``, recomputed ``UVW`` toward ``new_dir``,
        and a ``field_and_source`` node carrying ``FIELD_PHASE_CENTER_DIRECTION`` (ra, dec).
    """
    node = _partition(dt)
    main = node.ds
    times = main.time.values  # float64 unix seconds
    ra_new, dec_new = float(new_dir[0]), float(new_dir[1])

    # --- per-frame celestial bookkeeping (host) ----------------------------------------------
    # Source ITRS unit vectors for the NEW centre, and the OLD centre (instantaneous zenith) both
    # as ITRS unit vectors (for the current UVW) and as ICRS angles (for the l,m,n offset below).
    loc = _site_location(node)
    s_new = _itrs_unit_vectors(ra_new, dec_new, times)  # (n_time, 3)
    ra_old, dec_old, s_old = _instantaneous_zenith(times, loc)  # (n_time,), (n_time,), (n_time, 3)

    # ITRS baselines b = pos(ant1) - pos(ant2), in the same antenna order as the main node.
    baselines = itrs_baselines(node, xp)  # (n_bl, 3) on device

    # --- new and current UVW (device) --------------------------------------------------------
    uvw_new = _project_baselines(baselines, s_new, xp)  # (n_time, n_bl, 3)
    uvw_cur = _project_baselines(baselines, s_old, xp)  # current frame (~ ENU baseline, w ~ 0)

    # --- fringe-rotation phasor (device) -----------------------------------------------------
    # Direction cosines of the new centre measured in the OLD (instantaneous-zenith) frame.
    d_ra = ra_new - ra_old  # (n_time,)
    ll = np.cos(dec_new) * np.sin(d_ra)
    mm = np.sin(dec_new) * np.cos(dec_old) - np.cos(dec_new) * np.sin(dec_old) * np.cos(d_ra)
    nn = np.sin(dec_new) * np.sin(dec_old) + np.cos(dec_new) * np.cos(dec_old) * np.cos(d_ra) - 1.0

    freq = main.frequency.values  # (n_chan,) Hz
    inv_wl = xp.asarray(freq / 299792458.0)  # cycles per metre, (n_chan,)
    ll_x = xp.asarray(ll)[:, None, None]  # (n_time, 1, 1)
    mm_x = xp.asarray(mm)[:, None, None]
    nn_x = xp.asarray(nn)[:, None, None]

    # phase[t, bl, chan] = phasesign * 2pi * (u*ll + v*mm + w*nn) / wl
    geom = uvw_cur[..., 0:1] * ll_x + uvw_cur[..., 1:2] * mm_x + uvw_cur[..., 2:3] * nn_x  # (n_time, n_bl, 1) in metres
    phase = phasesign * 2.0j * np.pi * geom * inv_wl[None, None, :]  # (n_time, n_bl, n_chan)
    phasor = xp.exp(phase)

    vis = xp.asarray(main.VISIBILITY.values)  # (n_time, n_bl, n_chan, n_pol)
    vis_rephased = vis * phasor[..., None]

    # --- assemble the updated DataTree -------------------------------------------------------
    return _rebuild(dt, node, _to_numpy(vis_rephased, xp), _to_numpy(uvw_new, xp), ra_new, dec_new)


# ---------------------------------------------------------------------------------------------
# Host-side helpers (per-frame, O(n_time)): astropy celestial bookkeeping.
# ---------------------------------------------------------------------------------------------


def _itrs_unit_vectors(ra: float | np.ndarray, dec: float | np.ndarray, times: np.ndarray) -> np.ndarray:
    """ITRS unit vectors of an ICRS direction at each timestamp.

    Transforming ICRS -> ITRS at the epoch folds in precession, nutation and Earth rotation, which
    is what casacore does and what the ~0.3 deg J2000-to-date precession at this epoch requires.

    Args:
        ra: Right ascension in radians (scalar or per-time array).
        dec: Declination in radians (scalar or per-time array).
        times: Unix-second timestamps, shape ``(n_time,)``.

    Returns:
        ``(n_time, 3)`` array of ITRS unit vectors.
    """
    import astropy.units as u
    from astropy.coordinates import ITRS, SkyCoord
    from astropy.time import Time

    tt = Time(times, format="unix", scale="utc")
    itrs = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs").transform_to(ITRS(obstime=tt))
    return np.stack([itrs.x.value, itrs.y.value, itrs.z.value], axis=1)


def _instantaneous_zenith(times: np.ndarray, loc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ICRS angles and ITRS unit vectors of the site zenith at each timestamp.

    Args:
        times: Unix-second timestamps, shape ``(n_time,)``.
        loc: Site ``astropy.coordinates.EarthLocation``.

    Returns:
        ``(ra, dec, s_itrs)`` with ``ra``/``dec`` in radians (each ``(n_time,)``) and ``s_itrs``
        the ITRS unit vectors ``(n_time, 3)``.
    """
    import astropy.units as u
    from astropy.coordinates import ITRS, AltAz, SkyCoord
    from astropy.time import Time

    tt = Time(times, format="unix", scale="utc")
    zen = SkyCoord(AltAz(az=0 * u.deg, alt=90 * u.deg, obstime=tt, location=loc)).icrs
    itrs = zen.transform_to(ITRS(obstime=tt))
    s = np.stack([itrs.x.value, itrs.y.value, itrs.z.value], axis=1)
    return zen.ra.rad, zen.dec.rad, s


def _site_location(node: xr.DataTree):
    """Build the site EarthLocation from the partition's observation_info attrs."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    info = node.ds.attrs["observation_info"]
    return EarthLocation(
        lat=info["site_latitude_deg"] * u.deg,
        lon=info["site_longitude_deg"] * u.deg,
        height=info["site_altitude_m"] * u.m,
    )


# ---------------------------------------------------------------------------------------------
# Device-side helpers (per-baseline / per-channel): pure xp array ops, GPU-capable.
# ---------------------------------------------------------------------------------------------


def _project_baselines(baselines, s_host: np.ndarray, xp: ModuleType):
    """Project ITRS baselines onto the (u, v, w) axes of a per-time source direction.

    The ``(u, v, w)`` frame is the standard one: ``w`` toward the source, ``u`` east, ``v`` north.
    The orthonormal axes are derived on the host (one set per frame) and the projection itself runs
    under ``xp``.

    Args:
        baselines: ``(n_bl, 3)`` ITRS baseline vectors (an ``xp`` array).
        s_host: ``(n_time, 3)`` source ITRS unit vectors (host ``numpy`` array).
        xp: Array module.

    Returns:
        ``(n_time, n_bl, 3)`` UVW coordinates in metres (an ``xp`` array).
    """
    zhat = np.array([0.0, 0.0, 1.0])
    uhat = np.cross(zhat, s_host)
    uhat /= np.linalg.norm(uhat, axis=1, keepdims=True)  # east
    vhat = np.cross(s_host, uhat)  # north

    s_x = xp.asarray(s_host)[:, None, :]  # (n_time, 1, 3)
    u_x = xp.asarray(uhat)[:, None, :]
    v_x = xp.asarray(vhat)[:, None, :]
    b_x = baselines[None, :, :]  # (1, n_bl, 3)

    uu = (b_x * u_x).sum(-1)
    vv = (b_x * v_x).sum(-1)
    ww = (b_x * s_x).sum(-1)
    return xp.stack([uu, vv, ww], axis=-1)


def itrs_baselines(node: xr.DataTree, xp: ModuleType):
    """Form b = pos(ant1) - pos(ant2) in ITRS for every baseline, as an xp array."""
    antenna = node["antenna_xds"].to_dataset(inherit=False)
    pos = antenna.ANTENNA_POSITION.values  # (n_ant, 3) ITRS
    names = list(antenna.antenna_name.values)
    index = {name: i for i, name in enumerate(names)}
    a1 = np.array([index[n] for n in node.ds.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in node.ds.baseline_antenna2_name.values])
    return xp.asarray(pos[a1] - pos[a2])


def _to_numpy(arr, xp: ModuleType) -> np.ndarray:
    """Bring an xp array back to host numpy (no-op for numpy, .get() for cupy)."""
    if xp is np:
        return np.asarray(arr)
    return arr.get()


# ---------------------------------------------------------------------------------------------
# DataTree assembly.
# ---------------------------------------------------------------------------------------------


def _partition(dt: xr.DataTree) -> xr.DataTree:
    """Return the sole partition node beneath an MSv4 DataTree root."""
    children = list(dt.children)
    if len(children) != 1:
        raise ValueError(f"expected exactly one partition node, found {children}")
    return dt[children[0]]


def _rebuild(
    dt: xr.DataTree,
    node: xr.DataTree,
    vis: np.ndarray,
    uvw: np.ndarray,
    ra_new: float,
    dec_new: float,
) -> xr.DataTree:
    """Return a new DataTree with rephased visibilities, new UVW and a celestial phase centre."""
    partition_name = list(dt.children)[0]
    main = node.ds.copy()
    main["VISIBILITY"] = (main.VISIBILITY.dims, vis, dict(main.VISIBILITY.attrs))
    # UVW is now referenced to a fixed celestial direction rather than the drifting zenith.
    main["UVW"] = (main.UVW.dims, uvw, {"type": "uvw", "units": "m", "frame": "fk5"})
    # The phase centre is now a single fixed field; relabel the per-time field accordingly.
    main = main.assign_coords(field_name=("time", np.full(main.sizes["time"], "phasecenter")))

    # Replace the zenith el/az field node with the MSv4 ra/dec phase-centre representation.
    field = xr.Dataset(
        data_vars={
            "FIELD_PHASE_CENTER_DIRECTION": (
                ("field_name", "sky_dir_label"),
                np.array([[ra_new, dec_new]], dtype=np.float64),
                {"type": "sky_coord", "units": "rad", "frame": "fk5"},
            ),
        },
        coords={
            "field_name": ("field_name", np.array(["phasecenter"])),
            "source_name": ("field_name", np.array(["phasecenter"])),
            "sky_dir_label": ("sky_dir_label", np.array(["ra", "dec"])),
        },
        attrs={"type": "field_and_source"},
    )

    tree = {f"/{partition_name}": main}
    for child in node.children:
        if child == "field_and_source_base_xds":
            continue
        tree[f"/{partition_name}/{child}"] = node[child].to_dataset(inherit=False)
    tree[f"/{partition_name}/field_and_source_base_xds"] = field
    return xr.DataTree.from_dict(tree)


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
    info = None
    for path in hdf_paths:
        main = partition_datatree(read_hdf_as_msv4(path)).ds
        times = np.asarray(main.time.values)
        lo, hi = float(times.min()), float(times.max())
        t_lo = lo if t_lo is None else min(t_lo, lo)
        t_hi = hi if t_hi is None else max(t_hi, hi)
        # Site info comes from the first file; all inputs are assumed to share the same array site.
        if info is None:
            info = main.attrs["observation_info"]
    if info is None:
        raise ValueError("no HDF files provided")

    t_mid = 0.5 * (t_lo + t_hi)
    loc = EarthLocation(
        lat=info["site_latitude_deg"] * u.deg,
        lon=info["site_longitude_deg"] * u.deg,
        height=info["site_altitude_m"] * u.m,
    )
    aa = AltAz(az=0.0 * u.deg, alt=90.0 * u.deg, obstime=Time(t_mid, format="unix", scale="utc"), location=loc)
    icrs = SkyCoord(aa).icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)
