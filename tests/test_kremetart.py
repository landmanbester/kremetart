"""Tests for ``kremetart()``: host orchestration around the runner seam.

These run on any machine: the ``cpu`` fixture forces the CPU loop runner by monkeypatching the device
detector, so ``kremetart()`` images real HDFs end to end at ``nside=1``. Only the web server is faked
(``FrameServer`` + ``_wait_for_interrupt``) so no port is bound and the frozen session returns at once.
The runners themselves are tested in ``test_runners.py``. Test data comes from the shared ``hdf_dir``
fixture (``conftest.py``).
"""

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

import kremetart.core.kremetart as km
import kremetart.core.runner as runner_module
from kremetart.core.loop_runner import run_loop


class FakeServer:
    """Records its constructor kwargs (``FakeServer.last``) and never binds a port."""

    last: dict = {}

    def __init__(self, holder: object, **kwargs: object) -> None:
        FakeServer.last = {"holder": holder, **kwargs}

    def start(self) -> str:
        return "http://localhost:8080/"

    def stop(self) -> None:
        FakeServer.last["stopped"] = True


@pytest.fixture
def cpu(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The kremetart command module on the CPU loop runner, with the web server faked."""
    monkeypatch.setattr(runner_module, "cuda_available", lambda: False)
    monkeypatch.setattr(km, "FrameServer", FakeServer)
    monkeypatch.setattr(km, "_wait_for_interrupt", lambda: None)
    FakeServer.last = {}
    return km


@pytest.fixture
def prepare_kwargs(cpu: ModuleType, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Records the keyword arguments ``kremetart()`` passes to the prepare step (which still runs)."""
    calls: dict = {}
    real = cpu.prepare_msv4_zarr

    def spy(paths: list[Path], out: Path, **kwargs: object) -> Path:
        calls.update(kwargs)
        return real(paths, out, **kwargs)

    monkeypatch.setattr(cpu, "prepare_msv4_zarr", spy)
    return calls


def test_kremetart_runs_the_requested_runner(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    calls: list[str] = []

    def fake_cyberether(
        prepared: Path, output: Path, settings: object, holder: object, *, device: str, window: bool
    ) -> None:
        calls.append(f"{device}, window={window}")
        run_loop(prepared, output, settings, holder)

    monkeypatch.setattr(runner_module, "run_cyberether", fake_cyberether)
    monkeypatch.setattr(runner_module, "cupy_device_available", lambda: False)
    monkeypatch.setattr(runner_module, "find_cyberether", lambda: "/opt/cyberether")
    cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=1, serve=False, runner="cyberether")
    assert calls == ["cpu, window=False"]
    assert "CyberEther (CPU)" in capsys.readouterr().out


def test_kremetart_fails_fast_when_cyberether_is_missing(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_prepare(*args: object, **kwargs: object) -> None:
        raise AssertionError("snapshots must not be prepared when the runner cannot start")

    monkeypatch.setattr(cpu, "prepare_msv4_zarr", no_prepare)
    monkeypatch.setenv("KREMETART_CYBERETHER", str(tmp_path / "missing.AppImage"))
    with pytest.raises(FileNotFoundError, match="is not a file"):
        cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, runner="cyberether")
    assert FakeServer.last == {}  # the web server never started


def test_kremetart_requires_both_phase_components(tmp_path: Path) -> None:
    # Validation happens before any data access or server start, so no HDFs are needed.
    with pytest.raises(ValueError, match="both or neither"):
        km.kremetart(hdf_dir=tmp_path, output=tmp_path / "out.zarr", phase_ra_deg=10.0)


def test_kremetart_images_on_cpu(cpu: ModuleType, hdf_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Without a GPU the loop runner images every snapshot into the map store."""
    out = tmp_path / "out.zarr"
    result = cpu.kremetart(hdf_dir=hdf_dir, output=out, nside=1, nframes=2, serve=False)

    assert result == out
    assert "CPU (NumPy loop)" in capsys.readouterr().out
    ds = xr.open_zarr(out)
    assert set(ds.data_vars) == {"dirty", "filtered", "znorm"}  # deconvolution is opt-in
    for var in ds.data_vars:
        assert ds[var].shape == (2, 12)
        assert np.all(np.isfinite(ds[var].values))


def test_kremetart_honors_explicit_phase_direction(
    cpu: ModuleType, prepare_kwargs: dict, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_cpd(paths: list[Path]) -> tuple[float, float]:
        raise AssertionError("common_phase_direction must not be called when phase is supplied")

    monkeypatch.setattr(cpu, "common_phase_direction", fail_cpd)
    cpu.kremetart(
        hdf_dir=hdf_dir,
        output=tmp_path / "out.zarr",
        nside=1,
        nframes=1,
        phase_ra_deg=12.0,
        phase_dec_deg=-20.0,
        serve=False,
    )
    assert prepare_kwargs["phase_ra_deg"] == 12.0 and prepare_kwargs["phase_dec_deg"] == -20.0


def test_kremetart_auto_phase_direction_used(
    cpu: ModuleType, prepare_kwargs: dict, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cpu, "common_phase_direction", lambda paths: (123.0, 45.0))
    cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=1, serve=False)
    assert prepare_kwargs["phase_ra_deg"] == 123.0 and prepare_kwargs["phase_dec_deg"] == 45.0


def test_kremetart_overlay_passes_tracks_to_server(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the name kremetart() calls (km.satellite_tracks) so no API/catalogue read happens.
    monkeypatch.setattr(cpu, "satellite_tracks", lambda paths, elev, **k: {"SAT": [(0, 1.0, 2.0, 1.0)]})
    cpu.kremetart(
        hdf_dir=hdf_dir,
        output=tmp_path / "out.zarr",
        nside=1,
        nframes=1,
        overlay_catalog=True,
        catalog_elevation_deg=30.0,
    )
    assert FakeServer.last["tracks"] == {"SAT": [(0, 1.0, 2.0, 1.0)]}


def test_kremetart_no_overlay_passes_none_tracks(cpu: ModuleType, hdf_dir: Path, tmp_path: Path) -> None:
    cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=1)
    assert FakeServer.last["tracks"] is None


def test_kremetart_labels_the_smoothed_dirty_maps_in_dirty_units(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path
) -> None:
    """Without Tikhonov the IWP filter smooths the dirty maps, so the viewer labels them Jy/beam."""
    cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=1)
    assert FakeServer.last["units"] == {"dirty": "Jy/beam", "smooth": "Jy/beam", "znorm": ""}


def test_kremetart_serve_streams_finishes_and_stops_server(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waited = {"n": 0}
    monkeypatch.setattr(cpu, "_wait_for_interrupt", lambda: waited.__setitem__("n", waited["n"] + 1))

    cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=2, l2=0.01, l1=0.01)

    holder = FakeServer.last["holder"]
    assert FakeServer.last["names"] == ("dirty", "tikhonov", "l1", "smooth", "znorm")
    assert FakeServer.last["nest"] is True
    assert FakeServer.last["units"]["smooth"] == "Jy/pixel"  # smooths the Tikhonov maps
    assert all(slot is not None for slot in holder.snapshot().values())  # every stream published
    assert holder.current_seq == 1  # one shared sequence number per snapshot
    assert holder.finished is True  # frozen-session signal sent
    assert FakeServer.last.get("stopped") is True  # server torn down
    assert waited["n"] == 1  # blocked for inspection exactly once


def test_kremetart_stops_server_when_imaging_fails(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(cpu, "prepare_msv4_zarr", boom)
    with pytest.raises(RuntimeError, match="prepare failed"):
        cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1)
    assert FakeServer.last["holder"].finished is True
    assert FakeServer.last.get("stopped") is True


def test_kremetart_no_serve_skips_server(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        def __init__(self, *a: object, **k: object) -> None:
            raise AssertionError("FrameServer must not be constructed when serve=False")

    monkeypatch.setattr(cpu, "FrameServer", Boom)
    cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=1, serve=False)


def test_kremetart_nframes_caps_the_snapshots(cpu: ModuleType, hdf_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.zarr"
    cpu.kremetart(hdf_dir=hdf_dir, output=out, nside=1, nframes=3, serve=False)
    assert xr.open_zarr(out).sizes["TIME"] == 3


def test_kremetart_default_catalog_cache_path(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_tracks(paths: list[Path], elevation_deg: float, **k: object) -> dict:
        captured.update(k)
        return {}

    monkeypatch.setattr(cpu, "satellite_tracks", fake_tracks)
    output = tmp_path / "out.zarr"
    cpu.kremetart(hdf_dir=hdf_dir, output=output, nside=1, overlay_catalog=True, nframes=3)
    assert captured["cache_path"] == str(output) + ".catalog.zarr"
    assert captured["nframes"] == 3


def test_kremetart_profile_prints(
    cpu: ModuleType, hdf_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    cpu.kremetart(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=1, profile=True, serve=False)
    out = capsys.readouterr().out
    assert "kremetart profile" in out
    assert "imaging" in out


def test_kremetart_overwrite_fail_fast(cpu: ModuleType, hdf_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.zarr"
    output.mkdir()  # pre-existing output

    with pytest.raises(FileExistsError):
        cpu.kremetart(hdf_dir=hdf_dir, output=output, nside=1, serve=False)

    # With overwrite set, the guard passes and the run replaces the directory with a map store.
    cpu.kremetart(hdf_dir=hdf_dir, output=output, nside=1, nframes=1, overwrite=True, serve=False)
    assert xr.open_zarr(output).sizes["TIME"] == 1
