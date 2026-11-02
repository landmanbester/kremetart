"""Holoscan port plumbing for a Snapshot: the reader's five output ports <-> one :class:`Snapshot`."""

import cupy as cp
from holoscan.core import InputContext, OperatorSpec

from kremetart.imaging.snapshot import Snapshot

SNAPSHOT_PORTS = ("VISIBILITY", "WEIGHT", "B_ROT", "BORESIGHT", "time")


def declare_snapshot_inputs(spec: OperatorSpec) -> None:
    """Declare the inputs a stage operator needs to receive a whole snapshot."""
    for port in SNAPSHOT_PORTS:
        spec.input(port)


def snapshot_flow() -> set[tuple[str, str]]:
    """The ``add_flow`` port map from the reader to a stage operator's snapshot inputs."""
    return {(port, port) for port in SNAPSHOT_PORTS}


def receive_snapshot(op_input: InputContext, freqs: cp.ndarray) -> Snapshot:
    """Receive one snapshot (the reader emits each array with a leading length-1 time axis)."""
    vis, weight, b_rot, boresight, time = (cp.asarray(op_input.receive(port)) for port in SNAPSHOT_PORTS)
    return Snapshot(
        vis=vis[0], weight=weight[0], b_rot=b_rot[0], boresight=boresight[0], time=float(time[0]), freqs=freqs
    )
