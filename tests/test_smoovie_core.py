"""Tests for the smoovie core (HEALPix movie generation)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("xarray_ms")  # read_hdf_as_msv4 path uses MSv4 machinery

from kremetart.core.smoovie import frame_dirty_maps  # noqa: E402

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_frame_dirty_maps_one_frame_per_subintegration():
    from kremetart.core.smoovie import _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    paths = _hdfs()[:2]
    nside = 16
    expected = sum(int(_partition(read_hdf_as_msv4(p)).ds.time.size) for p in paths)
    maps, stamps, pix = frame_dirty_maps(paths, nside)
    npix = 12 * nside * nside
    assert len(maps) == len(stamps) == expected
    assert expected > len(paths)  # genuinely per-slice, not per-file
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
    assert pix.shape == (npix, 3)
    assert "UTC" in stamps[0]


def test_smoovie_produces_movie(tmp_path):
    pytest.importorskip("matplotlib")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from kremetart.core.smoovie import smoovie

    _hdfs()  # skip if reference data absent
    out = tmp_path / "movie.mp4"
    smoovie(hdf_dir=_DATA, movie=out, nside=32, fps=2)
    assert out.exists() and out.stat().st_size > 0


def test_common_phase_direction_dec_matches_latitude():
    pytest.importorskip("astropy")
    paths = _hdfs()
    from kremetart.core.smoovie import _partition, common_phase_direction
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    ra, dec = common_phase_direction(paths)
    info = _partition(read_hdf_as_msv4(paths[0])).ds.attrs["observation_info"]
    lat = info["site_latitude_deg"]
    # The declination of the local zenith equals the observer's geodetic latitude, up to the
    # geodetic-vs-geocentric difference (~0.2 deg). Independent physical check, not a re-derivation.
    assert abs(dec - lat) < 0.3
    assert 0.0 <= ra < 360.0
    # Deterministic.
    assert (ra, dec) == common_phase_direction(paths)


def test_common_phase_direction_empty_raises():
    from kremetart.core.smoovie import common_phase_direction

    with pytest.raises(ValueError, match="no HDF files"):
        common_phase_direction([])


def test_smoovie_requires_both_phase_components(tmp_path):
    from kremetart.core.smoovie import smoovie

    # Validation happens before any data access, so no HDFs are needed.
    with pytest.raises(ValueError, match="both or neither"):
        smoovie(hdf_dir=tmp_path, movie=tmp_path / "m.mp4", phase_ra_deg=10.0)


def test_smoovie_honors_explicit_phase_direction(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm

    _hdfs()  # need a non-empty glob; the heavy steps below are monkeypatched out
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True):
        captured["rot"] = rot
        return [Path("frame_0000.png")]

    def fail_cpd(paths):
        raise AssertionError("common_phase_direction must not be called when phase is supplied")

    monkeypatch.setattr(sm, "render_frames", fake_render)
    monkeypatch.setattr(sm, "common_phase_direction", fail_cpd)

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, phase_ra_deg=12.0, phase_dec_deg=-20.0)
    assert captured["rot"] == (12.0, -20.0)


def test_smoovie_auto_phase_direction_used(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm

    _hdfs()
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (123.0, 45.0))
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True):
        captured["rot"] = rot
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm, "render_frames", fake_render)

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1)
    assert captured["rot"] == (123.0, 45.0)
