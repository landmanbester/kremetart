"""Tests for satellite track assembly (catalogue fetch is injected; no network)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
pytest.importorskip("xarray_ms")  # read_hdf_as_msv4 path uses MSv4 machinery

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_satellite_tracks_align_with_frame_order():
    from kremetart.core.smoovie import _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:2]
    expected_frames = sum(int(_partition(read_hdf_as_msv4(p)).ds.time.size) for p in paths)

    calls = {"n": 0}

    def fake_fetch(lon, lat, datestr, elevation_deg):
        calls["n"] += 1
        return [{"name": "SAT-A", "az": 0.0, "el": 90.0, "jy": 1.0, "r": 7.0e6}]

    tracks = satellite_tracks(paths, 45.0, fetch=fake_fetch)

    assert calls["n"] == expected_frames  # one query per frame, in frame order
    assert set(tracks) == {"SAT-A"}
    points = tracks["SAT-A"]
    assert len(points) == expected_frames
    assert [p[0] for p in points] == list(range(expected_frames))  # frame indices 0..N-1


def test_satellite_tracks_radec_matches_astropy():
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    from kremetart.core.smoovie import _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:1]
    main = _partition(read_hdf_as_msv4(paths[0])).ds
    info = main.attrs["observation_info"]
    t0 = float(np.asarray(main.time.values)[0])

    def fake_fetch(lon, lat, datestr, elevation_deg):
        return [{"name": "SAT-A", "az": 30.0, "el": 60.0, "jy": 1.0, "r": 7.0e6}]

    tracks = satellite_tracks(paths, 45.0, fetch=fake_fetch)

    loc = EarthLocation(
        lat=info["site_latitude_deg"] * u.deg,
        lon=info["site_longitude_deg"] * u.deg,
        height=info["site_altitude_m"] * u.m,
    )
    ref = SkyCoord(
        AltAz(az=30.0 * u.deg, alt=60.0 * u.deg, obstime=Time(t0, format="unix", scale="utc"), location=loc)
    ).icrs

    frame, ra, dec, jy = tracks["SAT-A"][0]
    assert frame == 0
    assert abs(ra - float(ref.ra.deg)) < 1e-6
    assert abs(dec - float(ref.dec.deg)) < 1e-6
    assert jy == 1.0


def test_satellite_tracks_skips_empty_frames():
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:1]

    def fake_fetch(lon, lat, datestr, elevation_deg):
        return []  # nothing above the cutoff

    tracks = satellite_tracks(paths, 89.0, fetch=fake_fetch)
    assert tracks == {}
