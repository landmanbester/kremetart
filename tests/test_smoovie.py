"""Tests for the ``smoovie`` command and its Holoscan imaging app.

``smoovie`` is inherently GPU/Holoscan-driven (mirrors :mod:`kremetart.core.stream_msv4`): there is
no CPU fallback, so the whole suite is gated on a CUDA device plus the cupy/holoscan/healpy stack
(assumed installed via the ``[full]`` extra). On GitHub CPU runners these skip cleanly; run them
locally on a GPU box. If a test segfaults during the holoscan import, raise the stack limit first:
``ulimit -s 32768``. Test data comes from the shared ``hdf_paths`` / ``hdf_dir`` / ``catalog_cache`` /
``catalog_elevation`` fixtures (``conftest.py``).

The host-wiring tests use the ``sm_stub`` fixture, which stubs the single imaging seam
(``image_via_app``) and neutralises the ffmpeg encode so they exercise ``smoovie()``'s orchestration
(phase resolution, track overlay, cache-path default, nframes, profiling) without running the GPU
pipeline; the end-to-end tests run the real Holoscan app.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest


def _gpu() -> bool:
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        import healpy  # noqa: F401
        import holoscan  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="requires a CUDA device + cupy/holoscan/healpy")


@pytest.fixture
def sm():
    """The smoovie module (imported lazily so collection never needs the GPU stack)."""
    import kremetart.core.smoovie as sm

    return sm


@pytest.fixture
def sm_stub(sm, monkeypatch):
    """smoovie module with the GPU imaging + ffmpeg encode stubbed, for host-wiring tests.

    Tests add their own render / tracks / phase patches via ``monkeypatch.setattr(sm_stub, ...)``.
    """
    monkeypatch.setattr(
        sm, "image_via_app", lambda paths, nside, **k: ([np.zeros(12)], [np.zeros(12)], [np.zeros(12)], ["t UTC"])
    )
    monkeypatch.setattr(sm.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(sm.subprocess, "run", lambda *a, **k: None)
    return sm


# --- the GPU imaging app -------------------------------------------------------------------------


def test_gpu_operators_import():
    from kremetart.operators.dft_healpix import HealpixDFTOperator
    from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator
    from kremetart.operators.iwp_kalman import IWPKalmanOperator

    assert HealpixDFTOperator and HealpixZarrReaderOperator and HealpixWriterOperator
    assert IWPKalmanOperator


def test_image_via_app_end_to_end(hdf_paths):
    """The real Holoscan app images one frame per sub-integration into finite ``(npix,)`` maps."""
    from kremetart.core.smoovie import image_via_app

    nside = 8
    npix = 12 * nside * nside
    maps, stamps = image_via_app(hdf_paths[:1], nside, correct_gains=True, nframes=3)
    assert len(maps) == len(stamps) == 3
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
    assert "UTC" in stamps[0]


# --- host rendering / overlay --------------------------------------------------------------------


def test_overlay_tracks_marker_trail_label(sm):
    class FakeAx:
        def __init__(self):
            self.c = {"projscatter": 0, "projplot": 0}

        def projscatter(self, *a, **k):
            self.c["projscatter"] += 1

        def projplot(self, *a, **k):
            self.c["projplot"] += 1

    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.0), (1, 12.0, -19.0, 1.0)]}

    ax = FakeAx()
    sm._overlay_tracks(ax, tracks, 0)  # frame 0: marker, no trail yet
    assert ax.c == {"projscatter": 1, "projplot": 0}

    ax = FakeAx()
    sm._overlay_tracks(ax, tracks, 1)  # frame 1: marker + trail (>1 past point)
    assert ax.c == {"projscatter": 1, "projplot": 1}

    ax = FakeAx()
    sm._overlay_tracks(ax, tracks, 5)  # satellite absent at frame 5: nothing drawn
    assert ax.c == {"projscatter": 0, "projplot": 0}


def test_render_frames_overlay_uses_axes_not_drawing_wrappers(tmp_path, sm, monkeypatch):
    # Root-cause guard: the overlay must use the projection-axes methods, NOT the module-level
    # hp.proj* wrappers -- each wrapper forces a full pylab.draw(), turning an N-satellite overlay
    # into ~N full-figure re-rasterizations per frame (the cause of ~15 s/frame rendering). If
    # render_frames ever calls these wrappers again, the overlay raises here.
    import healpy as hp

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


# --- smoovie() orchestration (imaging + encode stubbed) ------------------------------------------


def test_smoovie_requires_both_phase_components(tmp_path, sm):
    # Validation happens before any data access, so no HDFs are needed.
    with pytest.raises(ValueError, match="both or neither"):
        sm.smoovie(hdf_dir=tmp_path, movie=tmp_path / "m.mp4", phase_ra_deg=10.0)


def test_smoovie_honors_explicit_phase_direction(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None, diverging=False):
        captured["rot"] = rot
        return [Path("frame_0000.png")]

    def fail_cpd(paths):
        raise AssertionError("common_phase_direction must not be called when phase is supplied")

    monkeypatch.setattr(sm_stub, "render_frames", fake_render)
    monkeypatch.setattr(sm_stub, "common_phase_direction", fail_cpd)

    sm_stub.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1, phase_ra_deg=12.0, phase_dec_deg=-20.0)
    assert captured["rot"] == (12.0, -20.0)


def test_smoovie_auto_phase_direction_used(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (123.0, 45.0))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None, diverging=False):
        captured["rot"] = rot
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm_stub, "render_frames", fake_render)

    sm_stub.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1)
    assert captured["rot"] == (123.0, 45.0)


def test_smoovie_overlay_passes_tracks(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    # Patch the name smoovie actually calls (sm.satellite_tracks), so no API/catalogue read happens.
    monkeypatch.setattr(sm_stub, "satellite_tracks", lambda paths, elev, **k: {"SAT": [(0, 1.0, 2.0, 1.0)]})

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None, diverging=False):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm_stub, "render_frames", fake_render)

    sm_stub.smoovie(
        hdf_dir=hdf_dir,
        movie=tmp_path / "m.mp4",
        nside=1,
        overlay_catalog=True,
        catalog_elevation_deg=30.0,
    )
    assert captured["tracks"] == {"SAT": [(0, 1.0, 2.0, 1.0)]}


def test_smoovie_no_overlay_passes_none_tracks(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None, diverging=False):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm_stub, "render_frames", fake_render)

    sm_stub.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1)
    assert captured["tracks"] is None


def test_smoovie_nframes_flows_to_imaging(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    def fake_image(paths, nside, **k):
        captured.update(k)
        return ([np.zeros(12)], [np.zeros(12)], [np.zeros(12)], ["t UTC"])

    monkeypatch.setattr(sm_stub, "image_via_app", fake_image)
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm_stub, "render_frames", lambda *a, **k: [Path("frame_0000.png")])

    sm_stub.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1, nframes=7)
    assert captured.get("nframes") == 7


def test_smoovie_default_catalog_cache_path(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm_stub, "render_frames", lambda *a, **k: [Path("frame_0000.png")])

    def fake_tracks(paths, elevation_deg, **k):
        captured.update(k)
        return {}

    monkeypatch.setattr(sm_stub, "satellite_tracks", fake_tracks)

    movie = tmp_path / "m.mp4"
    sm_stub.smoovie(hdf_dir=hdf_dir, movie=movie, nside=1, overlay_catalog=True, nframes=3)
    assert captured["cache_path"] == str(movie) + ".catalog.zarr"
    assert captured["nframes"] == 3


def test_smoovie_profile_prints(tmp_path, hdf_dir, sm_stub, monkeypatch, capsys):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm_stub, "render_frames", lambda *a, **k: [Path("frame_0000.png")])

    sm_stub.smoovie(hdf_dir=hdf_dir, movie=tmp_path / "m.mp4", nside=1, profile=True)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out and "render" in out


# --- full end-to-end (real Holoscan app + real ffmpeg) -------------------------------------------


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
