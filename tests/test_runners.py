"""The runner seam: both adapters image a prepared snapshot zarr into the same map store.

The loop runner and runner selection run everywhere. The Holoscan runner, and its agreement with the
loop runner on CuPy, need a CUDA device (run after ``ulimit -s 32768``).

Agreement tolerance: both runners call the same stages on CuPy with the same inputs and store float32
maps; measured on the test HDFs (nside 8) they are bit-identical. The bound is the CuPy parity
test's 1e-6 relative L2 per snapshot, leaving room for reduction-order changes on other GPUs.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import kremetart.core.runner as runner_module
from kremetart.core.cyberether_runner import run_cyberether
from kremetart.core.loop_runner import run_loop
from kremetart.core.runner import select_runner
from kremetart.core.settings import ImagingSettings
from kremetart.utils.healpix_viz import LatestFrameHolder
from tests.conftest import PREPARED_NFRAMES as NFRAMES

requires_gpu = pytest.mark.skipif(not runner_module.cuda_available(), reason="requires a CUDA device + holoscan")


def test_select_runner_without_gpu_is_the_loop_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "cuda_available", lambda: False)
    assert select_runner().run is run_loop
    assert select_runner("loop").run is run_loop


def test_select_runner_refuses_holoscan_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs a CUDA device"):
        select_runner("holoscan")


@pytest.mark.parametrize("name, window", [("cyberether", False), ("cyberether-gui", True)])
@pytest.mark.parametrize("gpu, device", [(False, "cpu"), (True, "cuda")])
def test_select_runner_cyberether_detects_the_device(
    monkeypatch: pytest.MonkeyPatch, gpu: bool, device: str, name: str, window: bool
) -> None:
    monkeypatch.setattr(runner_module, "cupy_device_available", lambda: gpu)
    choice = select_runner(name)
    assert choice.run.func is run_cyberether and choice.run.keywords == {"device": device, "window": window}


def test_select_runner_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown runner"):
        select_runner("gnuradio")


@pytest.mark.parametrize(
    "l2, l1, stored, streamed",
    [
        (None, None, ("dirty", "filtered", "znorm"), ("dirty", "smooth", "znorm")),
        (0.01, None, ("dirty", "tikhonov", "filtered", "znorm"), ("dirty", "tikhonov", "smooth", "znorm")),
        (
            0.01,
            0.01,
            ("dirty", "tikhonov", "l1", "filtered", "znorm"),
            ("dirty", "tikhonov", "l1", "smooth", "znorm"),
        ),
    ],
    ids=["default", "l2", "l2-l1"],
)
def test_settings_name_the_enabled_maps(
    l2: float | None, l1: float | None, stored: tuple[str, ...], streamed: tuple[str, ...]
) -> None:
    settings = ImagingSettings(nside=1, l2=l2, l1=l1)
    assert settings.stored_names == stored
    assert settings.stream_names == streamed


@pytest.mark.parametrize("l2, l1", [(None, None), (0.01, 0.01)], ids=["default", "l2-l1"])
def test_loop_runner_writes_and_publishes_every_map(
    prepared: Path, tmp_path: Path, l2: float | None, l1: float | None
) -> None:
    settings = ImagingSettings(nside=2, l2=l2, l1=l1)
    holder = LatestFrameHolder(settings.stream_names)
    out = tmp_path / "maps.zarr"

    run_loop(prepared, out, settings, holder)

    ds = xr.open_zarr(out)
    assert set(ds.data_vars) == set(settings.stored_names)
    np.testing.assert_array_equal(ds.TIME.values, xr.open_zarr(prepared).time.values)
    for var in ds.data_vars:
        assert ds[var].shape == (NFRAMES, settings.npix)
        assert np.all(np.isfinite(ds[var].values))
    slots = holder.snapshot()
    assert all(slots[name] is not None and len(slots[name].data) == settings.npix * 4 for name in slots)
    assert holder.current_seq == NFRAMES - 1


@requires_gpu
def test_select_runner_with_gpu_is_the_holoscan_runner() -> None:
    # GPU-only dependency: holoscan_runner imports holoscan and cupy at module top.
    from kremetart.core.holoscan_runner import run_holoscan

    assert select_runner().run is run_holoscan


@requires_gpu
@pytest.mark.parametrize("l2, l1", [(None, None), (0.01, 0.01)], ids=["default", "l2-l1"])
def test_holoscan_runner_agrees_with_the_loop_runner(
    prepared: Path, tmp_path: Path, l2: float | None, l1: float | None
) -> None:
    # GPU-only dependency: this module must import on CPU-only runners.
    import cupy as cp

    # GPU-only dependency: holoscan_runner imports holoscan and cupy at module top.
    from kremetart.core.holoscan_runner import run_holoscan

    settings = ImagingSettings(nside=8, l2=l2, l1=l1)
    holder = LatestFrameHolder(settings.stream_names)
    run_holoscan(prepared, tmp_path / "holoscan.zarr", settings, holder)
    run_loop(prepared, tmp_path / "loop.zarr", settings, xp=cp)

    got = xr.open_zarr(tmp_path / "holoscan.zarr")
    ref = xr.open_zarr(tmp_path / "loop.zarr")
    assert set(got.data_vars) == set(settings.stored_names)
    assert holder.current_seq == NFRAMES - 1
    for var in settings.stored_names:
        g, r = got[var].values.astype(np.float64), ref[var].values.astype(np.float64)
        err = np.linalg.norm(g - r, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-30)
        assert np.all(err <= 1e-6), f"{var}: relative L2 per snapshot {err}"
