"""The CyberEther runner: image a prepared snapshot zarr through a headless CyberEther flowgraph.

One Python block per stage, mirroring the Holoscan graph: source -> dirty imager -> Tikhonov and L1 (each
when enabled) -> map-store sink, with the IWP filter on the Tikhonov stream (the dirty stream when
Tikhonov is off). The flowgraph runs in a separate ``cyberether --headless`` process on this Python
interpreter, so this module never imports CyberEther; it follows the sink's progress file, publishes each
written snapshot to the live viewer from the map store, and stops the process once the last snapshot is
written.

The executable is ``$KREMETART_CYBERETHER`` if set, else ``cyberether`` on ``PATH``. A CyberEther block
that raises would otherwise skip silently forever, so each block stub records its traceback and the runner
raises it. CyberEther also keeps running when it rejects part of a flowgraph, so any ``[ERROR]`` line in
its log fails the run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import xarray as xr
import yaml
import zarr

from kremetart.core.settings import STREAM_NAME, ImagingSettings
from kremetart.utils.healpix_viz import LatestFrameHolder, publish_maps
from kremetart.utils.map_store import create_map_store

POLL_SECONDS = 0.05
SHUTDOWN_SECONDS = 10.0

_STUB = """\
import json
import os
import traceback

CONFIG = json.loads({config!r})
_block = None
_failed = False


def compute(ctx):
    global _block, _failed
    if _failed:
        return SKIP
    try:
        if _block is None:
            # Built on the first cycle: importing pyarrow, dask or astropy while CyberEther loads block code hangs it.
            from {module} import {cls}

            _block = {cls}(**CONFIG)
        if _block.compute(ctx) is False:
            return SKIP
    except Exception:
        _failed = True
        with open({error!r} + ".partial", "w") as f:
            f.write(traceback.format_exc())
        os.replace({error!r} + ".partial", {error!r})  # atomic: the runner never reads half a traceback
        return SKIP
"""


class _NoAliasDumper(yaml.SafeDumper):
    """Writes repeated objects out in full: CyberEther's parser does not resolve YAML aliases."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def find_cyberether() -> str:
    """The CyberEther executable: ``$KREMETART_CYBERETHER``, else ``cyberether`` on ``PATH``.

    Raises:
        FileNotFoundError: if neither is found.
    """
    exe = os.environ.get("KREMETART_CYBERETHER") or shutil.which("cyberether")
    if exe is None:
        raise FileNotFoundError("CyberEther not found: set KREMETART_CYBERETHER or put `cyberether` on PATH")
    return exe


def run_cyberether(
    prepared: Path,
    output: Path,
    settings: ImagingSettings,
    holder: LatestFrameHolder | None = None,
    *,
    device: str = "cuda",
) -> None:
    """Image every snapshot in ``prepared`` into the map store at ``output`` (the runner interface).

    Args:
        prepared: the prepared snapshot zarr.
        output: the map store to create.
        settings: the imaging settings.
        holder: when given, also publish each snapshot's maps to the live viewer.
        device: where the blocks keep their tensors and run the stages (``cuda`` or ``cpu``).

    Raises:
        RuntimeError: if a block fails or CyberEther exits before the last snapshot is written.
    """
    ds = xr.open_zarr(prepared)
    times = ds.time.values
    create_map_store(output, times, settings.npix, settings.stored_names)

    with tempfile.TemporaryDirectory(prefix="kremetart-cyberether-") as td:
        run_dir = Path(td)
        (run_dir / "errors").mkdir()
        graph = flowgraph(prepared, output, settings, run_dir, device=device, dataset=ds)
        flowgraph_path = run_dir / "flowgraph.yml"
        flowgraph_path.write_text("---\n" + yaml.dump(graph, Dumper=_NoAliasDumper, sort_keys=False, width=1000))
        log_path = run_dir / "cyberether.log"
        command = [
            find_cyberether(),
            "--headless",
            "--dependency-policy",
            "deny",
            "--python-runtime",
            sys.executable,
            str(flowgraph_path),
        ]
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=_environment(device))
            try:
                _follow(process, run_dir, output, len(times), holder, log_path)
            finally:
                _stop(process)


def flowgraph(
    prepared: Path, output: Path, settings: ImagingSettings, run_dir: Path, *, device: str, dataset: xr.Dataset
) -> dict[str, Any]:
    """The CyberEther flowgraph (as a YAML-ready dict) that images ``prepared`` into ``output``."""
    nbl, nchan = dataset.VISIBILITY.shape[1:]
    grid = {
        "device": device,
        "nside": settings.nside,
        "nest": settings.nest,
        "apply_beam": settings.apply_beam,
        "ground_plane_diameter": settings.ground_plane_diameter,
        "freqs": [float(f) for f in dataset.frequency.values],
    }
    npix = f"[{settings.npix}]"
    snapshot_specs = [
        _spec(f"[{nbl}, {nchan}]", "CF32", device),
        _spec(f"[{nbl}, {nchan}]", "F32", device),
        _spec(f"[{nbl}, 3]", "F64", device),
        _spec("[3]", "F64", device),
        _spec("[1]", "F64", device),
    ]
    snapshot = [_port("source", i) for i in range(len(snapshot_specs))]
    time_port = snapshot[-1]

    blocks = [
        _block(
            run_dir,
            "source",
            "kremetart.blocks.io",
            "SnapshotSource",
            {"prepared": str(prepared), "device": device},
            [],
            snapshot_specs,
        ),
        _block(
            run_dir,
            "imager",
            "kremetart.blocks.stages",
            "DirtyImagerBlock",
            grid,
            snapshot,
            [_spec(npix, "F64", device)],
        ),
    ]
    sink_inputs = {"dirty": _port("imager")}
    for name, cls, eta in (("tikhonov", "TikhonovSolverBlock", settings.l2), ("l1", "L1SolverBlock", settings.l1)):
        if eta is not None:
            config = {**grid, "eta": eta}
            blocks.append(
                _block(
                    run_dir,
                    name,
                    "kremetart.blocks.stages",
                    cls,
                    config,
                    [*snapshot, _port("imager")],
                    [_spec(npix, "F64", device)],
                )
            )
            sink_inputs[name] = _port(name)
    smoothed = sink_inputs.get("tikhonov", sink_inputs["dirty"])
    iwp_config = {"device": device, "npix": settings.npix, "sigma2": settings.iwp_sigma, "noise": settings.iwp_noise}
    blocks.append(
        _block(
            run_dir,
            "iwp",
            "kremetart.blocks.stages",
            "IWPFilterBlock",
            iwp_config,
            [smoothed, time_port],
            [_spec(npix, "F64", device), _spec(npix, "F64", device)],
        )
    )
    sink_inputs["filtered"] = _port("iwp", 0)
    sink_inputs["znorm"] = _port("iwp", 1)

    names = list(settings.stored_names)
    sink_config = {"output": str(output), "names": names, "progress": str(run_dir / "progress")}
    blocks.append(
        _block(
            run_dir,
            "sink",
            "kremetart.blocks.io",
            "MapStoreSink",
            sink_config,
            [sink_inputs[n] for n in names] + [time_port],
            [],
        )
    )
    return {"version": 2, "title": "kremetart smoovie", "graph": blocks}


def _spec(shape: str, dtype: str, device: str) -> dict[str, str]:
    return {"shape": shape, "dtype": dtype, "device": device}


def _port(block: str, index: int = 0) -> str:
    return f"${{graph.{block}.output.output{index}}}"


def _block(
    run_dir: Path,
    name: str,
    module: str,
    cls: str,
    config: dict[str, Any],
    inputs: list[str],
    output_specs: list[dict[str, str]],
) -> dict[str, Any]:
    code = _STUB.format(
        config=json.dumps(config), module=module, cls=cls, error=str(run_dir / "errors" / f"{name}.txt")
    )
    block: dict[str, Any] = {
        "name": name,
        "module": "python",
        "device": "cpu",  # the Python block's own module is CPU-only; its tensors may live on CUDA
        "runtime": "python",
        "provider": "generic",
        "config": {
            "code": code,
            "inputCount": len(inputs),
            "outputCount": len(output_specs),
            "outputTensorSpecs": output_specs,
            "throttled": False,
        },
    }
    if inputs:
        block["input"] = {f"input{i}": port for i, port in enumerate(inputs)}
    return block


def _follow(
    process: subprocess.Popen,
    run_dir: Path,
    output: Path,
    ntime: int,
    holder: LatestFrameHolder | None,
    log_path: Path,
) -> None:
    """Wait for the sink to write ``ntime`` snapshots, publishing each one to ``holder`` as it lands."""
    progress = run_dir / "progress"
    # Read rows straight from zarr: a lazy xarray read per row is slower than the graph itself.
    store = zarr.open_group(str(output), mode="r") if holder is not None else None
    times = store["TIME"][:] if store is not None else None
    names = [name for name in STREAM_NAME if store is not None and name in store]
    published = 0
    while True:
        graph_errors = [line for line in log_path.read_text().splitlines() if "[ERROR]" in line]
        if graph_errors:
            raise RuntimeError("CyberEther rejected the flowgraph:\n" + "\n".join(graph_errors))
        failures = sorted((run_dir / "errors").glob("*.txt"))
        if failures:
            raise RuntimeError(f"CyberEther block {failures[0].stem!r} failed:\n{failures[0].read_text()}")
        written = int(progress.read_text()) if progress.exists() else 0
        if store is not None:
            for k in range(published, written):
                publish_maps(holder, k, float(times[k]), {STREAM_NAME[name]: store[name][k] for name in names})
            published = written
        if written >= ntime:
            return
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            raise RuntimeError(
                f"CyberEther exited with code {process.returncode} after {written}/{ntime} snapshots:\n{tail}"
            )
        time.sleep(POLL_SECONDS)


def _environment(device: str) -> dict[str, str]:
    """CyberEther's environment: ours, plus a CUDA 13+ NVRTC preloaded for CUDA flowgraphs.

    The CyberEther AppImages bundle CUDA 12's NVRTC, which cannot compile CuPy's kernels against CUDA 13
    headers (nor for newer GPUs such as the DGX Spark's GB10). Preloading the NVRTC that CuPy uses outside
    CyberEther makes it compile the same way inside.
    """
    env = dict(os.environ)
    nvrtc = find_nvrtc() if device == "cuda" else None
    if nvrtc is not None:
        env["LD_PRELOAD"] = ":".join(p for p in (str(nvrtc), env.get("LD_PRELOAD")) if p)
    return env


def find_nvrtc() -> Path | None:
    """The newest ``libnvrtc.so.<major>`` in the first place that has one.

    Searched in order: this interpreter's ``nvidia`` CUDA wheels (what a pip-installed CuPy loads), then
    the CUDA toolkits at ``$CUDA_PATH``, ``$CUDA_HOME`` and ``/usr/local/cuda*``.
    """
    roots: list[Path] = []
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        roots += [Path(root) for root in spec.submodule_search_locations]
    toolkits = [os.environ.get(var) for var in ("CUDA_PATH", "CUDA_HOME")]
    roots += [Path(t) for t in toolkits if t] + sorted(Path("/usr/local").glob("cuda*"))
    for root in roots:
        found = [
            lib
            for pattern in ("*/lib/libnvrtc.so.*", "lib64/libnvrtc.so.*", "targets/*/lib/libnvrtc.so.*")
            for lib in root.glob(pattern)
            if lib.name.removeprefix("libnvrtc.so.").isdigit()
        ]
        if found:
            return max(found, key=lambda lib: int(lib.name.rsplit(".", 1)[-1]))
    return None


def _stop(process: subprocess.Popen) -> None:
    """End CyberEther with SIGTERM, killing it if it lingers.

    Nothing is lost: the sink writes each snapshot before reporting it, and the blocks hold no resources
    that need a cleanup hook. SIGINT would run those hooks, but once the blocks have loaded kremetart
    CyberEther takes several seconds to honour it, or ignores it.
    """
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
