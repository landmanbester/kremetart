"""Tests for the smoovie core (HEALPix movie generation).

Shared test data comes from the ``hdf_paths`` / ``hdf_dir`` / ``catalog_cache`` / ``catalog_elevation``
fixtures in ``conftest.py``. The wiring tests use the ``sm_cpu`` fixture below, which disables the GPU
branch and stubs imaging/encode so they exercise ``smoovie``'s orchestration without touching a GPU,
the catalogue API, or the slow imaging path.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from kremetart.core.smoovie import frame_dirty_maps
from kremetart.utils import partition_datatree
from kremetart.utils.calibration import correct_file_gains


@pytest.fixture
def sm_cpu(monkeypatch):
    """The smoovie module with its heavy stages stubbed, for orchestration/wiring tests.

    Disables the GPU branch (so the patchable CPU ``frame_dirty_maps`` path is taken) and replaces
    imaging + encode with light stand-ins. Tests add their own render / tracks / phase patches via
    ``monkeypatch.setattr(sm_cpu, ...)``.
    """
    import kremetart.core.smoovie as sm

    monkeypatch.setattr(sm, "gpu_available", lambda: False)
    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))
    return sm


def test_frame_dirty_maps_one_frame_per_subintegration(hdf_paths):
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    paths = hdf_paths[:2]
    nside = 16
    expected = sum(int(partition_datatree(read_hdf_as_msv4(p)).ds.time.size) for p in paths)
    maps, stamps, pix = frame_dirty_maps(paths, nside)
    npix = 12 * nside * nside
    assert len(maps) == len(stamps) == expected
    assert expected > len(paths)  # genuinely per-slice, not per-file
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
    assert pix.shape == (npix, 3)
    assert "UTC" in stamps[0]


def test_correct_file_gains_real_data(hdf_paths):
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    node = partition_datatree(read_hdf_as_msv4(hdf_paths[0]))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]

    vis_c, wgt_c = correct_file_gains(node, vis, wgt)

    assert vis_c.shape == vis.shape
    assert np.all(np.isfinite(vis_c)) and np.all(np.isfinite(wgt_c))
    # The correction must actually change non-trivial gains.
    assert not np.allclose(vis_c, vis)
    # Dead antennas (gain 0) -> zero-weight, zero-vis baselines (no inf/nan).
    gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values
    if np.any(gains == 0):
        assert np.any(wgt_c == 0)


def test_frame_dirty_maps_correct_gains_finite(hdf_paths):
    nside = 16
    maps, stamps, pix = frame_dirty_maps(hdf_paths[:1], nside, correct_gains=True)
    npix = 12 * nside * nside
    assert len(maps) == len(stamps) > 0
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))


def test_frame_dirty_maps_nframes_caps(hdf_paths):
    maps, stamps, pix = frame_dirty_maps(hdf_paths, 16, nframes=3)  # nframes caps the total frames
    assert len(maps) == len(stamps) == 3


def test_overlay_tracks_marker_trail_label():
    from kremetart.core.smoovie import _overlay_tracks

    class FakeAx:
        def __init__(self):
            self.c = {"projscatter": 0, "projplot": 0}

        def projscatter(self, *a, **k):
            self.c["projscatter"] += 1

        def projplot(self, *a, **k):
            self.c["projplot"] += 1

    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.0), (1, 12.0, -19.0, 1.0)]}

    ax = FakeAx()
    _overlay_tracks(ax, tracks, 0)  # frame 0: marker + label, no trail yet
    assert ax.c == {"projscatter": 1, "projplot": 0}

    ax = FakeAx()
    _overlay_tracks(ax, tracks, 1)  # frame 1: marker + label + trail (>1 past point)
    assert ax.c == {"projscatter": 1, "projplot": 1}

    ax = FakeAx()
    _overlay_tracks(ax, tracks, 5)  # satellite absent at frame 5: nothing drawn
    assert ax.c == {"projscatter": 0, "projplot": 0}


def test_render_frames_overlay_uses_axes_not_drawing_wrappers(tmp_path, monkeypatch):
    import healpy as hp

    import kremetart.core.smoovie as sm

    # Root-cause guard: the overlay must use the projection-axes methods, NOT the module-level
    # hp.proj* wrappers -- each wrapper forces a full pylab.draw(), turning an N-satellite overlay into
    # ~N full-figure re-rasterizations per frame (the cause of ~15 s/frame rendering). If render_frames
    # ever calls these wrappers again, the overlay raises here.
    def boom(*a, **k):
        raise AssertionError("overlay must call ax.proj* (no pylab.draw), not the hp.proj* wrappers")

    monkeypatch.setattr(hp, "projscatter", boom)
    monkeypatch.setattr(hp, "projplot", boom)

    nside = 8
    npix = 12 * nside * nside
    maps = [np.arange(npix, dtype=float), np.arange(npix, dtype=float) + 1.0]
    stamps = ["t0 UTC", "t1 UTC"]
    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.0), (1, 12.0, -19.0, 1.0)]}

    pngs = sm.render_frames(maps, stamps, nside, "inferno", tmp_path, rot=(0.0, -30.0), tracks=tracks)
    assert len(pngs) == 2


def test_smoovie_produces_movie(tmp_path, hdf_dir, catalog_cache, catalog_elevation):
    """Short, catalogue-backed end-to-end: nframes caps the work and the bundled cache means the
    overlay never queries the TART API."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from kremetart.core.smoovie import smoovie

    out = tmp_path / "movie.mp4"
    smoovie(
        hdf_dir=hdf_dir,
        movie=out,
        nside=16,
        fps=2,
        nframes=4,
        overlay_catalog=True,
        catalog_cache=catalog_cache,
        catalog_elevation_deg=catalog_elevation,
    )
    assert out.exists() and out.stat().st_size > 0


def test_common_phase_direction_dec_matches_latitude(hdf_paths):
    from kremetart.core.smoovie import common_phase_direction
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    ra, dec = common_phase_direction(hdf_paths)
    info = partition_datatree(read_hdf_as_msv4(hdf_paths[0])).ds.attrs["observation_info"]
    lat = info["site_latitude_deg"]
    # The declination of the local zenith equals the observer's geodetic latitude, up to the
    # geodetic-vs-geocentric difference (~0.2 deg). Independent physical check, not a re-derivation.
    assert abs(dec - lat) < 0.3
    assert 0.0 <= ra < 360.0
    # Deterministic.
    assert (ra, dec) == common_phase_direction(hdf_paths)


def test_common_phase_direction_empty_raises():
    from kremetart.core.smoovie import common_phase_direction

    with pytest.raises(ValueError, match="no HDF files"):
        common_phase_direction([])


def test_smoovie_requires_both_phase_components(tmp_path):
    from kremetart.core.smoovie import smoovie

    # Validation happens before any data access, so no HDFs are needed.
    with pytest.raises(ValueError, match="both or neither"):
        smoovie(hdf_dir=tmp_path, movie=tmp_path / "m.mp4", phase_ra_deg=10.0)


def test_smoovie_honors_explicit_phase_direction(tmp_path, hdf_dir, sm_cpu, monkeypatch):
    captured = {}

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["rot"] = rot
        return [Path("frame_0000.png")]

    def fail_cpd(paths):
        raise AssertionError("common_phase_direction must not be called when phase is supplied")

    monkeypatch.setattr(sm_cpu, "render_frames", fake_render)
    monkeypatch.setattr(sm_cpu, "common_phase_direction", fail_cpd)

    sm_cpu.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1, phase_ra_deg=12.0, phase_dec_deg=-20.0)
    assert captured["rot"] == (12.0, -20.0)


def test_smoovie_auto_phase_direction_used(tmp_path, hdf_dir, sm_cpu, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_cpu, "common_phase_direction", lambda paths: (123.0, 45.0))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["rot"] = rot
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm_cpu, "render_frames", fake_render)

    sm_cpu.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1)
    assert captured["rot"] == (123.0, 45.0)


def test_smoovie_overlay_passes_tracks(tmp_path, hdf_dir, sm_cpu, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_cpu, "common_phase_direction", lambda paths: (0.0, 0.0))
    # Patch the name smoovie actually calls (sm.satellite_tracks), so no API/catalogue read happens.
    monkeypatch.setattr(sm_cpu, "satellite_tracks", lambda paths, elev, **k: {"SAT": [(0, 1.0, 2.0, 1.0)]})

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm_cpu, "render_frames", fake_render)

    sm_cpu.smoovie(
        hdf_dir=hdf_dir,
        movie=tmp_path / "m.mp4",
        nside=1,
        overlay_catalog=True,
        catalog_elevation_deg=30.0,
    )
    assert captured["tracks"] == {"SAT": [(0, 1.0, 2.0, 1.0)]}


def test_smoovie_no_overlay_passes_none_tracks(tmp_path, hdf_dir, sm_cpu, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_cpu, "common_phase_direction", lambda paths: (0.0, 0.0))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm_cpu, "render_frames", fake_render)

    sm_cpu.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1)
    assert captured["tracks"] is None


@pytest.mark.skipif(
    os.environ.get("KREMETART_MS_ORACLE") != "1",
    reason="opt-in: cross-checks tart2ms calibration convention (set KREMETART_MS_ORACLE=1)",
)
def test_weighted_corrected_vis_matches_calibrated_ms(data_dir):
    """tart2ms writes the *weighted* corrected visibility into DATA: ``V_corr * |g_p g_q|**2``.

    ``correct_file_gains`` returns ``(V_corr, W_corr)`` with ``W_corr = |g_p g_q|**2``, and the
    imaging step forms ``W_corr * V_corr`` -- which equals ``V_raw * conj(g_p) * g_q``, exactly the
    calibrated DATA tart2ms writes. (The MS WEIGHT column is a separate constant nominal value, not
    this gain weight.) The small residual tail is the known ~0.3% ITRF baseline-position convention
    difference -- the same source as the ~cm UVW tolerance in ``test_rephasing.py``.
    """
    import xarray as xr

    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    hdf = data_dir / "vis_2026-06-09_08_11_43.476804.hdf"
    ms = data_dir / "vis_2026-06-09_08_11_43.476804.ms"  # calibrated (NOT _nocal)
    if not (hdf.exists() and ms.exists()):
        pytest.skip("HDF or calibrated MS not present")

    node = partition_datatree(read_hdf_as_msv4(hdf))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]
    vis_c, wgt_c = correct_file_gains(node, vis, wgt)

    ref = partition_datatree(xr.open_datatree(str(ms), engine="xarray-ms:msv2"))
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


def test_smoovie_profile_prints(tmp_path, hdf_dir, sm_cpu, monkeypatch, capsys):
    monkeypatch.setattr(sm_cpu, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm_cpu, "render_frames", lambda *a, **k: [Path("frame_0000.png")])

    sm_cpu.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1, profile=True)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out and "render" in out


def test_smoovie_nframes_flows_to_imaging(tmp_path, hdf_dir, sm_cpu, monkeypatch):
    captured = {}

    def fake_fdm(paths, nside, **k):
        captured.update(k)
        return ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))

    monkeypatch.setattr(sm_cpu, "frame_dirty_maps", fake_fdm)
    monkeypatch.setattr(sm_cpu, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm_cpu, "render_frames", lambda *a, **k: [Path("frame_0000.png")])

    sm_cpu.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1, nframes=7)
    assert captured.get("nframes") == 7


def test_smoovie_default_catalog_cache_path(tmp_path, hdf_dir, sm_cpu, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_cpu, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm_cpu, "render_frames", lambda *a, **k: [Path("frame_0000.png")])

    def fake_tracks(paths, elevation_deg, **k):
        captured.update(k)
        return {}

    monkeypatch.setattr(sm_cpu, "satellite_tracks", fake_tracks)

    movie = tmp_path / "m.mp4"
    sm_cpu.smoovie(hdf_dir=hdf_dir, movie=movie, nside=1, overlay_catalog=True, nframes=3)
    assert captured["cache_path"] == str(movie) + ".catalog.zarr"
    assert captured["nframes"] == 3


def test_gpu_available_false_without_cuda_device(monkeypatch):
    """gpu_available() is False when no CUDA device is present (the CPU-fallback trigger)."""
    import kremetart.utils as ku

    # gpu_available() calls cupy.cuda.runtime.getDeviceCount(); 0 devices -> CPU path.
    monkeypatch.setattr(ku.cupy.cuda.runtime, "getDeviceCount", lambda: 0)
    assert ku.gpu_available() is False
