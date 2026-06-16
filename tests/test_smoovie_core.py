"""Tests for the smoovie core (HEALPix movie generation)."""

import os
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


def test_correct_file_gains_real_data():
    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    node = _partition(read_hdf_as_msv4(_hdfs()[0]))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]

    vis_c, wgt_c = _correct_file_gains(node, vis, wgt)

    assert vis_c.shape == vis.shape
    assert np.all(np.isfinite(vis_c)) and np.all(np.isfinite(wgt_c))
    # The correction must actually change non-trivial gains.
    assert not np.allclose(vis_c, vis)
    # Dead antennas (gain 0) -> zero-weight, zero-vis baselines (no inf/nan).
    gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values
    if np.any(gains == 0):
        assert np.any(wgt_c == 0)


def test_frame_dirty_maps_correct_gains_finite():
    paths = _hdfs()[:1]
    nside = 16
    maps, stamps, pix = frame_dirty_maps(paths, nside, correct_gains=True)
    npix = 12 * nside * nside
    assert len(maps) == len(stamps) > 0
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))


def test_frame_dirty_maps_nframes_caps():
    paths = _hdfs()  # multiple files; nframes caps the total frames produced
    maps, stamps, pix = frame_dirty_maps(paths, 16, nframes=3)
    assert len(maps) == len(stamps) == 3


def test_render_frames_overlays_tracks(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import healpy as hp

    import kremetart.core.smoovie as sm

    calls = {"scatter": 0, "plot": 0, "text": 0}
    monkeypatch.setattr(hp, "projscatter", lambda *a, **k: calls.__setitem__("scatter", calls["scatter"] + 1))
    monkeypatch.setattr(hp, "projplot", lambda *a, **k: calls.__setitem__("plot", calls["plot"] + 1))
    monkeypatch.setattr(hp, "projtext", lambda *a, **k: calls.__setitem__("text", calls["text"] + 1))

    nside = 8
    npix = 12 * nside * nside
    maps = [np.arange(npix, dtype=float), np.arange(npix, dtype=float) + 1.0]
    stamps = ["t0 UTC", "t1 UTC"]
    # SAT-A is present in both frames; the trailing line only appears once there are >1 past points.
    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.0), (1, 12.0, -19.0, 1.0)]}

    pngs = sm.render_frames(maps, stamps, nside, "inferno", tmp_path, rot=(0.0, -30.0), tracks=tracks)

    assert len(pngs) == 2
    assert calls["scatter"] == 2  # one marker per frame
    assert calls["text"] == 2  # one label per frame
    assert calls["plot"] == 1  # trail drawn only on frame 1 (needs >1 past point)


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

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
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

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["rot"] = rot
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm, "render_frames", fake_render)

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1)
    assert captured["rot"] == (123.0, 45.0)


def test_smoovie_overlay_passes_tracks(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm
    import kremetart.utils.satellites as sat

    _hdfs()  # need a non-empty glob; heavy steps are monkeypatched out
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))
    monkeypatch.setattr(sat, "satellite_tracks", lambda paths, elev, **k: {"SAT": [(0, 1.0, 2.0, 1.0)]})

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm, "render_frames", fake_render)

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, overlay_catalog=True, catalog_elevation_deg=30.0)
    assert captured["tracks"] == {"SAT": [(0, 1.0, 2.0, 1.0)]}


def test_smoovie_no_overlay_passes_none_tracks(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm

    _hdfs()
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm, "render_frames", fake_render)

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1)
    assert captured["tracks"] is None


@pytest.mark.skipif(
    os.environ.get("KREMETART_MS_ORACLE") != "1",
    reason="opt-in: cross-checks tart2ms calibration convention (set KREMETART_MS_ORACLE=1)",
)
def test_weighted_corrected_vis_matches_calibrated_ms():
    """tart2ms writes the *weighted* corrected visibility into DATA: ``V_corr * |g_p g_q|**2``.

    ``_correct_file_gains`` returns ``(V_corr, W_corr)`` with ``W_corr = |g_p g_q|**2``, and the
    imaging step forms ``W_corr * V_corr`` -- which equals ``V_raw * conj(g_p) * g_q``, exactly the
    calibrated DATA tart2ms writes. (The MS WEIGHT column is a separate constant nominal value, not
    this gain weight.) The small residual tail is the known ~0.3% ITRF baseline-position convention
    difference -- the same source as the ~cm UVW tolerance in ``test_rephasing.py``.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_ms")  # registers the "xarray-ms:msv2" engine

    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    hdf = _DATA / "vis_2026-06-09_08_11_43.476804.hdf"
    ms = _DATA / "vis_2026-06-09_08_11_43.476804.ms"  # calibrated (NOT _nocal)
    if not (hdf.exists() and ms.exists()):
        pytest.skip("HDF or calibrated MS not present")

    node = _partition(read_hdf_as_msv4(hdf))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]
    vis_c, wgt_c = _correct_file_gains(node, vis, wgt)

    ref = _partition(xr.open_datatree(str(ms), engine="xarray-ms:msv2"))
    ref_vis = np.asarray(ref.ds.VISIBILITY.values)[..., 0]

    # Compare only weighted baselines (dead antennas are zeroed on our side); the MSv4 reader and
    # tart2ms share baseline ordering (see test_rephasing.py), so this is an element-wise compare.
    mask = wgt > 0
    resid = np.abs((vis_c * wgt_c)[mask] - ref_vis[mask])
    assert np.median(resid) < 0.015
    assert np.percentile(resid, 95) < 0.05


def test_print_profile_outputs(capsys):
    from kremetart.core.smoovie import _print_profile

    _print_profile([("imaging", 2.0), ("render", 1.0)], nframes=4)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out and "render" in out and "TOTAL" in out


def test_stage_timer_records():
    from kremetart.core.smoovie import _stage_timer

    timings = []
    with _stage_timer("stage_a", timings):
        pass
    assert len(timings) == 1 and timings[0][0] == "stage_a" and timings[0][1] >= 0.0


def test_smoovie_profile_prints(tmp_path, monkeypatch, capsys):
    import kremetart.core.smoovie as sm

    _hdfs()
    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "render_frames", lambda *a, **k: [Path("frame_0000.png")])
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, profile=True)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out and "render" in out


def test_smoovie_nframes_flows_to_imaging(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm

    _hdfs()
    captured = {}

    def fake_fdm(paths, nside, **k):
        captured.update(k)
        return ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))

    monkeypatch.setattr(sm, "frame_dirty_maps", fake_fdm)
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "render_frames", lambda *a, **k: [Path("frame_0000.png")])
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, nframes=7)
    assert captured.get("nframes") == 7


def test_smoovie_default_catalog_cache_path(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm
    import kremetart.utils.satellites as sat

    _hdfs()
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "render_frames", lambda *a, **k: [Path("frame_0000.png")])
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    def fake_tracks(paths, elevation_deg, **k):
        captured.update(k)
        return {}

    monkeypatch.setattr(sat, "satellite_tracks", fake_tracks)

    movie = tmp_path / "m.mp4"
    sm.smoovie(hdf_dir=_DATA, movie=movie, nside=1, overlay_catalog=True, nframes=3)
    assert captured["cache_path"] == str(movie) + ".catalog.zarr"
    assert captured["nframes"] == 3
