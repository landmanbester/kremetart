"""The runner seam: what drives the snapshot imager's stages over a prepared snapshot stream.

A **runner** reads every snapshot of a prepared zarr (:func:`kremetart.utils.read_tart_hdf.prepare_msv4_zarr`)
in time order, images it through the stages, writes the maps to the durable map store and, when given
a holder, publishes them to the live viewer. Two adapters satisfy it:

- :func:`kremetart.core.loop_runner.run_loop`: an in-process loop over :class:`SnapshotImager` (NumPy;
  runs anywhere).
- :func:`kremetart.core.holoscan_runner.run_holoscan`: the Holoscan graph, one operator per stage
  (needs a CUDA device).
- :func:`kremetart.core.cyberether_runner.run_cyberether`: a headless CyberEther flowgraph, one Python
  block per stage (needs CyberEther; uses the GPU when CuPy sees one).

:func:`select_runner` maps a runner name to an adapter; ``auto`` picks from the environment, and the
device is always detected rather than chosen. Tests force a path by monkeypatching the detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

from kremetart.core.cyberether_runner import run_cyberether
from kremetart.core.loop_runner import run_loop
from kremetart.core.settings import ImagingSettings
from kremetart.utils.healpix_viz import LatestFrameHolder


class Runner(Protocol):
    """Image every snapshot in ``prepared`` into the map store at ``output``.

    Args:
        prepared: the prepared snapshot zarr.
        output: the map store to create (replaced if present; the caller guards overwrites).
        settings: the imaging settings.
        holder: when given, every snapshot's maps are also published to it under a shared sequence
            number (0, 1, ...), using the live stream names.
    """

    def __call__(
        self, prepared: Path, output: Path, settings: ImagingSettings, holder: LatestFrameHolder | None = None
    ) -> None: ...


@dataclass(frozen=True)
class RunnerChoice:
    """The runner :func:`select_runner` picked, and a one-line description for the user."""

    run: Runner
    description: str


RUNNERS: tuple[str, ...] = ("auto", "loop", "holoscan", "cyberether")
"""Runner names :func:`select_runner` accepts (the ``--runner`` choices)."""


def cupy_device_available() -> bool:
    """Whether CuPy is installed and sees a CUDA device."""
    try:
        # GPU-only dependency: absent or unusable on CPU-only machines.
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def cuda_available() -> bool:
    """Whether a CUDA device and the whole GPU stack (CuPy, Holoscan) are usable here."""
    try:
        # GPU-only dependency: absent on CPU-only machines.
        import holoscan  # noqa: F401
    except Exception:
        return False
    return cupy_device_available()


def select_runner(name: str = "auto") -> RunnerChoice:
    """The runner called ``name`` (one of ``RUNNERS``).

    ``auto`` picks Holoscan when :func:`cuda_available`, otherwise the CPU loop runner. ``cyberether``
    keeps its tensors on the GPU when CuPy sees a device, otherwise on the CPU.

    Raises:
        ValueError: if ``name`` is not a runner.
        RuntimeError: if ``holoscan`` is asked for where it cannot run.
    """
    if name not in RUNNERS:
        raise ValueError(f"unknown runner {name!r}; choose one of {', '.join(RUNNERS)}")
    if name == "cyberether":
        device = "cuda" if cupy_device_available() else "cpu"
        return RunnerChoice(
            partial(run_cyberether, device=device), f"CyberEther ({'GPU' if device == 'cuda' else 'CPU'})"
        )
    if name == "holoscan" or (name == "auto" and cuda_available()):
        if not cuda_available():
            raise RuntimeError("the holoscan runner needs a CUDA device with CuPy and Holoscan installed")
        # GPU-only dependency: holoscan_runner imports holoscan and cupy at module top.
        from kremetart.core.holoscan_runner import run_holoscan

        return RunnerChoice(run_holoscan, "GPU (Holoscan)")
    return RunnerChoice(run_loop, "CPU (NumPy loop)")
