"""The graph's ends: a source that streams a prepared snapshot zarr, and a sink that writes the map store."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from kremetart.blocks.snapshot_ports import SNAPSHOT_PORTS, array_module, synchronize
from kremetart.utils.map_store import write_maps


class SnapshotSource:
    """Emit one snapshot of the prepared zarr per cycle on outputs 0-4; report exhaustion afterwards.

    Args:
        prepared: the prepared snapshot zarr.
        device: device of the output tensors (``cpu`` or ``cuda``).
    """

    def __init__(self, *, prepared: str, device: str) -> None:
        self.xp = array_module(device)
        ds = xr.open_zarr(prepared)
        self.arrays = [ds[name].values for name in SNAPSHOT_PORTS[:-1]] + [ds.time.values[:, None]]
        self.next = 0

    def compute(self, ctx: Any) -> bool:
        """Emit the next snapshot; ``False`` once every snapshot has been emitted."""
        if self.next >= len(self.arrays[-1]):
            return False
        for port, values in enumerate(self.arrays):
            ctx.outputs[port][...] = self.xp.asarray(values[self.next])
        synchronize(self.xp)
        self.next += 1
        return True


class MapStoreSink:
    """Write each snapshot's maps (one input per stored name, then the time) into the map store.

    After each snapshot it rewrites ``progress`` with the number of snapshots written, so a runner can
    follow the run from outside the CyberEther process.

    Args:
        output: the map store, already created with every snapshot's time.
        names: the stored name of each map input, in port order.
        progress: path of the progress file.
    """

    def __init__(self, *, output: str, names: list[str], progress: str) -> None:
        self.output = output
        self.names = names
        self.progress = Path(progress)
        self.written = 0

    def compute(self, ctx: Any) -> None:
        maps = {name: _to_host(ctx.inputs[port]) for port, name in enumerate(self.names)}
        time = float(_to_host(ctx.inputs[len(self.names)])[0])
        write_maps(self.output, time, maps)
        self.written += 1
        partial = self.progress.with_suffix(".partial")
        partial.write_text(str(self.written), encoding="utf-8")
        os.replace(partial, self.progress)  # atomic: a reader never sees a half-written count


def _to_host(a: Any) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)
