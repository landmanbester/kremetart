"""The CyberEther runner: the flowgraph it writes, and its agreement with the loop runner.

The flowgraph tests run everywhere. The run tests need CyberEther (``$KREMETART_CYBERETHER`` or
``cyberether`` on ``PATH``) and skip otherwise; the CUDA run also needs a CUDA device (not Holoscan). Each run costs
about 15 s of CyberEther start-up.

Agreement tolerance: the blocks call the same stages as the loop runner on the same device and inputs,
and every map crosses the graph as float64, so the maps must be bit-identical (measured: identical on
CuPy and NumPy at nside 16 over 100 snapshots).
"""

import ast
import json
import os
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import xarray as xr

import kremetart.core.cyberether_runner as cyberether_runner
from kremetart.core.cyberether_runner import find_cyberether, find_nvrtc, flowgraph, run_cyberether
from kremetart.core.loop_runner import run_loop
from kremetart.core.settings import ImagingSettings
from kremetart.utils.healpix_viz import LatestFrameHolder
from tests.conftest import PREPARED_NFRAMES

requires_cyberether = pytest.mark.skipif(
    not (os.environ.get("KREMETART_CYBERETHER") or shutil.which("cyberether")),
    reason="set KREMETART_CYBERETHER or put `cyberether` on PATH",
)


def _config(block: dict[str, Any]) -> dict[str, Any]:
    """The JSON config a block's stub constructs its adapter from."""
    line = next(line for line in block["config"]["code"].splitlines() if line.startswith("CONFIG = "))
    return json.loads(ast.literal_eval(line.removeprefix("CONFIG = json.loads(").removesuffix(")")))


def _graph(prepared: Path, tmp_path: Path, settings: ImagingSettings) -> dict[str, dict[str, Any]]:
    graph = flowgraph(prepared, tmp_path / "out.zarr", settings, tmp_path, device="cpu", dataset=xr.open_zarr(prepared))
    return {block["name"]: block for block in graph["graph"]}


@pytest.mark.parametrize("l2, l1", [(None, None), (0.01, 0.01)], ids=["default", "l2-l1"])
def test_flowgraph_wires_one_block_per_enabled_stage(
    prepared: Path, tmp_path: Path, l2: float | None, l1: float | None
) -> None:
    settings = ImagingSettings(nside=2, l2=l2, l1=l1)
    blocks = _graph(prepared, tmp_path, settings)

    solvers = {name for name, eta in (("tikhonov", l2), ("l1", l1)) if eta is not None}
    assert set(blocks) == {"source", "imager", "iwp", "sink"} | solvers
    # The IWP filter smooths the Tikhonov stream, or the dirty stream when Tikhonov is off.
    smoothed = "tikhonov" if l2 is not None else "imager"
    assert blocks["iwp"]["input"]["input0"] == f"${{graph.{smoothed}.output.output0}}"
    # The sink takes one input per stored map, in stored-name order, then the time.
    sink_inputs = list(blocks["sink"]["input"].values())
    assert len(sink_inputs) == len(settings.stored_names) + 1
    assert sink_inputs[-1] == "${graph.source.output.output4}"
    assert _config(blocks["sink"])["names"] == list(settings.stored_names)


@pytest.mark.parametrize("note", [False, True])
def test_flowgraph_lays_out_every_block_for_the_window(prepared: Path, tmp_path: Path, note: bool) -> None:
    settings = ImagingSettings(nside=2, l2=0.01, l1=0.01)
    graph = flowgraph(
        prepared, tmp_path / "o.zarr", settings, tmp_path, device="cpu", dataset=xr.open_zarr(prepared), note=note
    )
    positions = [(b["meta"]["node"]["x"], b["meta"]["node"]["y"]) for b in graph["graph"]]
    assert len(set(positions)) == len(positions)  # no two nodes stacked on each other
    assert ("about" in {b["name"] for b in graph["graph"]}) == note


class _RunningProcess:
    """Stands in for a CyberEther process that is alive but never writes a snapshot."""

    returncode = None

    def poll(self) -> None:
        return None


def test_a_stalled_run_raises_instead_of_waiting_forever(tmp_path: Path) -> None:
    (tmp_path / "errors").mkdir()
    log = tmp_path / "cyberether.log"
    log.write_text("JETSTREAM [INFO]  | [INSTANCE] Starting instance.\n")
    with pytest.raises(RuntimeError, match="wrote no snapshot for 0 s"):
        cyberether_runner._follow(_RunningProcess(), tmp_path, tmp_path / "o.zarr", 3, None, log, stall_seconds=0.2)


def test_flowgraph_output_specs_match_the_prepared_snapshots(prepared: Path, tmp_path: Path) -> None:
    ds = xr.open_zarr(prepared)
    nbl, nchan = ds.VISIBILITY.shape[1:]
    specs = _graph(prepared, tmp_path, ImagingSettings(nside=2))["source"]["config"]["outputTensorSpecs"]
    assert [(s["shape"], s["dtype"]) for s in specs] == [
        (f"[{nbl}, {nchan}]", "CF32"),
        (f"[{nbl}, {nchan}]", "F32"),
        (f"[{nbl}, 3]", "F64"),
        ("[3]", "F64"),
        ("[1]", "F64"),
    ]


def test_find_cyberether_names_the_appimages_next_to_a_wrong_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "CyberEther-linux-x64.AppImage").touch()
    monkeypatch.setenv("KREMETART_CYBERETHER", str(tmp_path / "CyberEther-linux-x64-cuda.AppImage"))
    with pytest.raises(FileNotFoundError, match="that directory has CyberEther-linux-x64.AppImage"):
        find_cyberether()


def test_find_cyberether_asks_for_chmod_on_a_non_executable_appimage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appimage = tmp_path / "CyberEther-linux-x64-cuda.AppImage"
    appimage.touch()
    monkeypatch.setenv("KREMETART_CYBERETHER", str(appimage))
    with pytest.raises(PermissionError, match="chmod \\+x"):
        find_cyberether()


def test_find_nvrtc_takes_the_newest_from_a_cuda_toolkit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = tmp_path / "cuda" / "targets" / "sbsa-linux" / "lib"
    lib.mkdir(parents=True)
    for name in ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so.13.0.88"):
        (lib / name).touch()
    monkeypatch.setattr(cyberether_runner.importlib.util, "find_spec", lambda name: None)  # no CUDA wheels
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "cuda"))
    assert find_nvrtc() == lib / "libnvrtc.so.13"


def _backends() -> list:
    """CPU always; CUDA when CuPy sees a device (the CyberEther runner does not need Holoscan)."""
    out = [pytest.param("cpu", np, id="cpu")]
    try:
        # GPU-only dependency: this module must import on CPU-only runners.
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            out.append(pytest.param("cuda", cupy, id="cuda"))
    except Exception:
        pass
    return out


@requires_cyberether
@pytest.mark.parametrize("device, xp", _backends())
def test_cyberether_runner_matches_the_loop_runner(prepared: Path, tmp_path: Path, device: str, xp: ModuleType) -> None:
    settings = ImagingSettings(nside=8, l2=0.01, l1=0.01)
    holder = LatestFrameHolder(settings.stream_names)
    run_cyberether(prepared, tmp_path / "cyberether.zarr", settings, holder, device=device)
    run_loop(prepared, tmp_path / "loop.zarr", settings, xp=xp)

    got = xr.open_zarr(tmp_path / "cyberether.zarr")
    ref = xr.open_zarr(tmp_path / "loop.zarr")
    for name in settings.stored_names:
        np.testing.assert_array_equal(got[name].values, ref[name].values, err_msg=name)
    assert holder.current_seq == PREPARED_NFRAMES - 1
    assert all(slot is not None for slot in holder.snapshot().values())


@requires_cyberether
def test_a_failing_block_raises_its_traceback(prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        graph = flowgraph(*args, **kwargs)
        source = graph["graph"][0]
        source["config"]["code"] = source["config"]["code"].replace(str(prepared), str(tmp_path / "missing.zarr"))
        return graph

    monkeypatch.setattr(cyberether_runner, "flowgraph", broken)
    with pytest.raises(RuntimeError, match="block 'source' failed"):
        run_cyberether(prepared, tmp_path / "out.zarr", ImagingSettings(nside=2), device="cpu")


@requires_cyberether
def test_a_rejected_flowgraph_raises_instead_of_hanging(
    prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        graph = flowgraph(*args, **kwargs)
        graph["graph"][-1]["input"]["input0"] = "${graph.nowhere.output.output0}"
        return graph

    monkeypatch.setattr(cyberether_runner, "flowgraph", broken)
    with pytest.raises(RuntimeError, match="rejected the flowgraph"):
        run_cyberether(prepared, tmp_path / "out.zarr", ImagingSettings(nside=2), device="cpu")
