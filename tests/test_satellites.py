"""Tests for satellite-track assembly against the bundled catalogue (no network).

Uses the shared ``hdf_paths`` / ``catalog_cache`` / ``catalog_elevation`` fixtures (``conftest.py``)
so the pre-downloaded ``tests/data/catalog.zarr`` backs :func:`satellite_tracks` and the TART
catalogue API is never queried. Testing the API / cache-write machinery itself is out of scope; we
only check that cached ``az``/``el`` rows are assembled into frame-aligned ICRS tracks correctly. The
injected ``fetch`` raises, so any cache miss fails loudly instead of hitting the network.
"""

import numpy as np

from kremetart.utils import partition_datatree
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
from kremetart.utils.satellites import satellite_tracks


def _no_network(lon, lat, datestr, elevation_deg):
    """fetch stand-in: the bundled catalogue must cover every frame, so this is never called."""
    raise AssertionError(f"catalogue API must not be queried in tests (missing frame {datestr!r})")


def _tracks(paths, cache, elevation, *, nframes=None):
    """satellite_tracks against the bundled cache, with the network forbidden."""
    return satellite_tracks(paths, elevation, fetch=_no_network, cache_path=cache, nframes=nframes)


def test_satellite_tracks_align_with_frame_order(hdf_paths, catalog_cache, catalog_elevation):
    """Tracks are keyed by satellite and indexed by the global frame number, sorted and in range."""
    paths = hdf_paths[:2]
    expected_frames = sum(int(partition_datatree(read_hdf_as_msv4(p)).ds.time.size) for p in paths)

    tracks = _tracks(paths, catalog_cache, catalog_elevation)

    assert tracks  # the bundled catalogue has sources above the cutoff
    all_frames = [f for points in tracks.values() for (f, *_rest) in points]
    assert min(all_frames) == 0
    assert max(all_frames) == expected_frames - 1
    for points in tracks.values():
        frames = [f for (f, *_rest) in points]
        assert frames == sorted(frames)  # each track is ordered by frame
        assert all(0 <= f < expected_frames for f in frames)


def test_satellite_tracks_radec_matches_astropy(hdf_paths, catalog_cache, catalog_elevation):
    """The az/el -> ICRS conversion matches astropy for a real cached source at frame 0."""
    import astropy.units as u
    import xarray as xr
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    main = partition_datatree(read_hdf_as_msv4(hdf_paths[0])).ds
    info = main.attrs["observation_info"]
    t0 = float(np.asarray(main.time.values)[0])

    # A real source present in frame 0 of the bundled catalogue (slot 0 is never padding here).
    cat = xr.open_zarr(catalog_cache)
    name0 = str(cat.source_name.values[0, 0])
    az0 = float(cat.source_azimuth_deg.values[0, 0])
    el0 = float(cat.source_elevation_deg.values[0, 0])

    loc = EarthLocation(
        lat=info["site_latitude_deg"] * u.deg,
        lon=info["site_longitude_deg"] * u.deg,
        height=info["site_altitude_m"] * u.m,
    )
    ref = SkyCoord(
        AltAz(az=az0 * u.deg, alt=el0 * u.deg, obstime=Time(t0, format="unix", scale="utc"), location=loc)
    ).icrs

    tracks = _tracks(hdf_paths[:1], catalog_cache, catalog_elevation)
    frame, ra, dec, _jy = tracks[name0][0]
    assert frame == 0
    assert abs(ra - float(ref.ra.deg)) < 1e-6
    assert abs(dec - float(ref.dec.deg)) < 1e-6


def test_satellite_tracks_nframes_caps(hdf_paths, catalog_cache, catalog_elevation):
    """nframes truncates the frame sequence, so no track references a frame beyond the cap."""
    tracks = _tracks(hdf_paths[:1], catalog_cache, catalog_elevation, nframes=2)
    all_frames = [f for points in tracks.values() for (f, *_rest) in points]
    assert all_frames  # sources are present in the first two frames
    assert max(all_frames) <= 1  # only frames 0 and 1 were imaged
