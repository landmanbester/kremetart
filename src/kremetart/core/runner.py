"""The runner seam: what drives the snapshot imager's stages over a prepared snapshot stream.

A **runner** reads every snapshot of a prepared zarr (:func:`kremetart.utils.read_tart_hdf.prepare_msv4_zarr`)
in time order, images it through the stages, writes the maps to the durable map store and, when given
a holder, publishes them to the live viewer. Two adapters satisfy it:

- :func:`kremetart.core.loop_runner.run_loop`: an in-process loop over :class:`SnapshotImager` (NumPy;
  runs anywhere).
- :func:`kremetart.core.holoscan_runner.run_holoscan`: the Holoscan graph, one operator per stage
  (needs a CUDA device).

:func:`select_runner` picks one from the environment, so callers never choose a device; tests force a
path by monkeypatching :func:`cuda_available`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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


def cuda_available() -> bool:
    """Whether a CUDA device and the GPU stack (CuPy, Holoscan) are usable here."""
    try:
        # GPU-only dependencies: absent or unusable on CPU-only machines.
        import cupy
        import holoscan  # noqa: F401

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def select_runner() -> RunnerChoice:
    """The Holoscan runner when a CUDA device is usable, otherwise the CPU loop runner."""
    if cuda_available():
        # GPU-only dependency: holoscan_runner imports holoscan and cupy at module top.
        from kremetart.core.holoscan_runner import run_holoscan

        return RunnerChoice(run_holoscan, "GPU (Holoscan)")
    return RunnerChoice(run_loop, "CPU (NumPy loop)")
