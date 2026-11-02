"""Block port plumbing for a Snapshot: the source's five outputs <-> one :class:`Snapshot`.

A snapshot crosses the graph as five tensors on ports 0-4 of every stage block, in ``SNAPSHOT_PORTS``
order; ``time`` is a length-1 tensor. Tensors on a ``cuda`` device arrive as CuPy arrays, ``cpu`` ones as
NumPy arrays.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from kremetart.imaging.snapshot import Array, Snapshot

SNAPSHOT_PORTS = ("VISIBILITY", "WEIGHT", "B_ROT", "BORESIGHT", "time")


def array_module(device: str) -> ModuleType:
    """NumPy for ``cpu`` tensors, CuPy for ``cuda`` ones."""
    if device == "cuda":
        # GPU-only dependency: CPU flowgraphs never import CuPy.
        import cupy

        return cupy
    return np


def receive_snapshot(ctx: Any, freqs: Array, xp: ModuleType) -> Snapshot:
    """The snapshot on input ports 0-4, copied so it outlives the cycle (block inputs are read-only views)."""
    vis, weight, b_rot, boresight, time = (xp.array(ctx.inputs[i]) for i in range(len(SNAPSHOT_PORTS)))
    return Snapshot(vis=vis, weight=weight, b_rot=b_rot, boresight=boresight, time=float(time[0]), freqs=freqs)


def synchronize(xp: ModuleType) -> None:
    """Finish queued GPU work so downstream blocks see complete outputs (a no-op on NumPy)."""
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
