"""Fetch TART catalogue satellite positions and project them into ICRS tracks.

The TART catalogue API returns, for a site ``(lon, lat)`` and a UTC datestr, the list of
sources above an elevation cutoff, each a dict with ``name``/``az``/``el``/``jy``/``r``.
:func:`satellite_tracks` queries it once per smoovie frame (one per sub-integration, in the
same order as :func:`kremetart.core.smoovie.frame_dirty_maps`), converts each ``(az, el)`` to
ICRS ``(ra, dec)`` at that timestamp, and groups the results by satellite name so the renderer
can draw a per-satellite track.

Network access lives here, isolated from the hermetic MSv4 reader. The ``fetch`` callable is
injectable so tests can supply a canned catalogue.
"""

from __future__ import annotations

import datetime

import numpy as np


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
    """Per-frame unix timestamps and the shared site, in frame_dirty_maps order.

    Returns:
        ``(times_unix, lat_deg, lon_deg, alt_m)`` where ``times_unix`` is a ``(n_frame,)`` array
        covering every sub-integration of every file, in order.

    Raises:
        ValueError: if ``hdf_paths`` is empty.
    """
    from kremetart.core.smoovie import _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    times_unix: list[float] = []
    info = None
    for path in hdf_paths:
        main = _partition(read_hdf_as_msv4(path)).ds
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


def satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch):
    """Per-satellite ICRS tracks aligned 1:1 with the smoovie frame sequence.

    Iterates the same ordering as :func:`kremetart.core.smoovie.frame_dirty_maps`, so the global
    frame index produced here matches the dirty-map index exactly.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (same order as ``frame_dirty_maps``).
        elevation_deg: elevation cutoff (deg) for catalogue sources.
        fetch: ``callable(lon, lat, datestr, elevation_deg) -> list[dict]``; injectable so tests
            avoid the network. Defaults to :func:`_tart_api_fetch`.

    Returns:
        ``dict`` mapping satellite name -> list of ``(frame_index, ra_deg, dec_deg, flux_jy)``,
        sorted by ``frame_index``. Satellites absent from a frame simply have no point there.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    times_unix, lat, lon, alt = _frame_times_and_site(hdf_paths)
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)

    tracks: dict[str, list] = {}
    for i, t in enumerate(times_unix):
        datestr = datetime.datetime.fromtimestamp(float(t), tz=datetime.timezone.utc).isoformat()
        sources = fetch(lon, lat, datestr, elevation_deg)
        if not sources:
            continue
        az = np.array([float(s["az"]) for s in sources])
        el = np.array([float(s["el"]) for s in sources])
        obstime = Time(float(t), format="unix", scale="utc")
        icrs = SkyCoord(AltAz(az=az * u.deg, alt=el * u.deg, obstime=obstime, location=loc)).icrs
        for src, ra, dec in zip(sources, icrs.ra.deg, icrs.dec.deg):
            tracks.setdefault(src["name"], []).append((i, float(ra), float(dec), float(src["jy"])))
    return tracks
