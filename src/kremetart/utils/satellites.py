"""Fetch TART catalogue satellite positions and project them into ICRS tracks.

The TART catalogue API returns, for a site ``(lon, lat)`` and a UTC datestr, the list of
sources above an elevation cutoff, each a dict with ``name``/``az``/``el``/``jy``/``r``.
:func:`satellite_tracks` queries it once per smoovie frame (one per sub-integration, in the
same order as the prepared imaging zarr; see :func:`kremetart.utils.read_tart_hdf.prepare_msv4_zarr`),
converts each ``(az, el)`` to
ICRS ``(ra, dec)`` at that timestamp, and groups the results by satellite name so the renderer
can draw a per-satellite track.

Network access lives here, isolated from the hermetic MSv4 reader. The ``fetch`` callable is
injectable so tests can supply a canned catalogue.
"""

from __future__ import annotations

import datetime

import numpy as np

from kremetart.utils import partition_datatree
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4


def _tart_api_fetch(lon, lat, datestr, elevation_deg):
    """Return the catalogue source list for a site at a UTC datestr (network).

    Args:
        lon: site longitude (deg).
        lat: site latitude (deg).
        datestr: ISO-8601 UTC timestamp string.
        elevation_deg: elevation cutoff (deg).

    Returns:
        A list of source dicts, each with ``name``/``az``/``el``/``jy``/``r``.

    Raises:
        RuntimeError: if the catalogue cannot be fetched after several retries.
    """
    from tart_tools import api_handler

    api = api_handler.APIhandler("")
    url = api.catalog_url(lon, lat, datestr=datestr) + f"&elevation={elevation_deg}"
    nretry = 5
    for retry in range(nretry):
        try:
            return api.get_url(url)
        except Exception as exc:  # noqa: BLE001 -- retry any transient API/network error
            print(f"Error fetching catalog (attempt {retry + 1}/{nretry}): {exc}")
    raise RuntimeError(f"Failed to fetch catalog after {nretry} attempts.")


def _frame_times_and_site(hdf_paths):
    """Per-frame unix timestamps and the shared site, in smoovie frame order.

    Returns:
        ``(times_unix, lat_deg, lon_deg, alt_m)`` where ``times_unix`` is a ``(n_frame,)`` array
        covering every sub-integration of every file, in order.

    Raises:
        ValueError: if ``hdf_paths`` is empty.
    """

    times_unix: list[float] = []
    info = None
    for path in hdf_paths:
        main = partition_datatree(read_hdf_as_msv4(path)).ds
        times_unix.extend(float(t) for t in np.asarray(main.time.values))
        if info is None:
            info = main.attrs["observation_info"]
    if info is None:
        raise ValueError("no HDF files provided")
    return (
        np.asarray(times_unix),
        info["site_latitude_deg"],
        info["site_longitude_deg"],
        info["site_altitude_m"],
    )


def _load_catalog_cache(path, lat, lon, elevation_deg):
    """Return ``{datestr -> source-list}`` from a cached catalogue zarr, or ``None`` on miss.

    A cache is reusable only if its site/elevation attrs match the request; otherwise ``None``
    (forces a refetch). Padding slots (``source_name == ""``) are dropped on read.
    """
    import os

    if not os.path.exists(path):
        return None
    import xarray as xr

    ds = xr.open_zarr(path)
    a = ds.attrs
    if not (
        np.isclose(a.get("site_latitude_deg", np.nan), lat)
        and np.isclose(a.get("site_longitude_deg", np.nan), lon)
        and np.isclose(a.get("elevation_deg", np.nan), elevation_deg)
    ):
        return None
    names = ds.source_name.values
    el = ds.source_elevation_deg.values
    az = ds.source_azimuth_deg.values
    jy = ds.source_flux_jy.values
    r = ds.source_height_m.values
    out: dict[str, list] = {}
    for ti, datestr in enumerate(ds.datestr.values):
        sources = []
        for si in range(names.shape[1]):
            name = str(names[ti, si])
            if name == "":
                continue  # padding slot
            sources.append(
                {
                    "name": name,
                    "el": float(el[ti, si]),
                    "az": float(az[ti, si]),
                    "jy": float(jy[ti, si]),
                    "r": float(r[ti, si]),
                }
            )
        out[str(datestr)] = sources
    return out


def _save_catalog_cache(path, datestrs, times_unix, per_frame, lat, lon, elevation_deg):
    """Write per-frame catalogue source lists to a ``(time, source)`` zarr ``Dataset``.

    ``source`` is padded to the max source count over all frames (``""`` / ``NaN`` for empty slots).
    ``source_name`` is stored as an object-dtype string array (verified to round-trip through zarr).
    """
    import os
    import shutil

    import xarray as xr

    nt = len(per_frame)
    nsrc = max((len(s) for s in per_frame), default=0)
    name = np.full((nt, nsrc), "", dtype=object)
    el = np.full((nt, nsrc), np.nan)
    az = np.full((nt, nsrc), np.nan)
    jy = np.full((nt, nsrc), np.nan)
    r = np.full((nt, nsrc), np.nan)
    for ti, sources in enumerate(per_frame):
        for si, s in enumerate(sources):
            name[ti, si] = str(s["name"])
            el[ti, si] = float(s["el"])
            az[ti, si] = float(s["az"])
            jy[ti, si] = float(s["jy"])
            r[ti, si] = float(s["r"])
    ds = xr.Dataset(
        data_vars={
            "source_name": (("time", "source"), name),
            "source_elevation_deg": (("time", "source"), el),
            "source_azimuth_deg": (("time", "source"), az),
            "source_flux_jy": (("time", "source"), jy),
            "source_height_m": (("time", "source"), r),
        },
        coords={
            "time": ("time", np.asarray(times_unix, dtype=np.float64)),
            "datestr": ("time", np.asarray(datestrs)),
        },
        attrs={
            "site_latitude_deg": float(lat),
            "site_longitude_deg": float(lon),
            "elevation_deg": float(elevation_deg),
        },
    )
    if os.path.exists(path):
        shutil.rmtree(path)  # zarr is a directory; overwrite cleanly
    ds.to_zarr(path, mode="w")


def satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch, cache_path=None, nframes=None):
    """Per-satellite ICRS tracks aligned 1:1 with the smoovie frame sequence.

    Iterates the same ordering as :func:`kremetart.core.smoovie.image_via_app`, so the global
    frame index produced here matches the dirty-map index exactly.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (same order as the imaged frames).
        elevation_deg: elevation cutoff (deg) for catalogue sources.
        fetch: ``callable(lon, lat, datestr, elevation_deg) -> list[dict]``; injectable so tests
            avoid the network. Defaults to :func:`_tart_api_fetch`.
        cache_path: optional zarr path; cached frames are reused and only misses are fetched, then
            the ``(time, source)`` dataset is rewritten. ``None`` disables caching.
        nframes: optional cap on the number of leading frames processed (profiling/preview aid).

    Returns:
        ``dict`` mapping satellite name -> list of ``(frame_index, ra_deg, dec_deg, flux_jy)``,
        sorted by ``frame_index``. Satellites absent from a frame simply have no point there.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    times_unix, lat, lon, alt = _frame_times_and_site(hdf_paths)
    if nframes is not None:
        times_unix = times_unix[:nframes]
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)

    datestrs = [datetime.datetime.fromtimestamp(float(t), tz=datetime.timezone.utc).isoformat() for t in times_unix]

    # Cache-aware per-frame source lists: reuse cached frames, fetch only the misses.
    cached = _load_catalog_cache(cache_path, lat, lon, elevation_deg) if cache_path else None
    per_frame, fetched_any = [], False
    for datestr in datestrs:
        if cached is not None and datestr in cached:
            per_frame.append(cached[datestr])
        else:
            per_frame.append(fetch(lon, lat, datestr, elevation_deg))
            fetched_any = True
    if cache_path and (cached is None or fetched_any):
        _save_catalog_cache(cache_path, datestrs, times_unix, per_frame, lat, lon, elevation_deg)

    # Convert az/el -> ICRS per frame and group into per-satellite tracks.
    tracks: dict[str, list] = {}
    for i, (t, sources) in enumerate(zip(times_unix, per_frame)):
        if not sources:
            continue
        az = np.array([float(s["az"]) for s in sources])
        el = np.array([float(s["el"]) for s in sources])
        obstime = Time(float(t), format="unix", scale="utc")
        icrs = SkyCoord(AltAz(az=az * u.deg, alt=el * u.deg, obstime=obstime, location=loc)).icrs
        for src, ra, dec in zip(sources, icrs.ra.deg, icrs.dec.deg):
            tracks.setdefault(src["name"], []).append((i, float(ra), float(dec), float(src["jy"])))
    return tracks


def frame_source_directions(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch, cache_path=None, nframes=None):
    """Per-frame ``(name, az_rad, el_rad)`` source lists aligned 1:1 with the imaged frame order.

    Reuses the cache-aware fetch loop of :func:`satellite_tracks` but returns ENU az/el (radians)
    for the calibration sky model rather than grouping into ICRS tracks. Catalogue az/el are stored
    in degrees and converted to radians here.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (same order as the imaged frames).
        elevation_deg: elevation cutoff (deg) for catalogue sources.
        fetch: ``callable(lon, lat, datestr, elevation_deg) -> list[dict]``; injectable for tests.
        cache_path: optional catalogue cache zarr path; ``None`` disables caching.
        nframes: optional cap on the number of leading frames processed.

    Returns:
        ``list`` (one entry per frame) of ``list[(name, az_rad, el_rad)]``.
    """
    times_unix, lat, lon, _alt = _frame_times_and_site(hdf_paths)
    if nframes is not None:
        times_unix = times_unix[:nframes]
    datestrs = [datetime.datetime.fromtimestamp(float(t), tz=datetime.timezone.utc).isoformat() for t in times_unix]

    cached = _load_catalog_cache(cache_path, lat, lon, elevation_deg) if cache_path else None
    out: list[list[tuple[str, float, float]]] = []
    for datestr in datestrs:
        if cached is not None and datestr in cached:
            sources = cached[datestr]
        else:
            sources = fetch(lon, lat, datestr, elevation_deg)
        out.append([(str(s["name"]), np.radians(float(s["az"])), np.radians(float(s["el"]))) for s in sources])
    return out
