"""Holoscan adapter for the dirty imager stage (:class:`kremetart.imaging.dirty_imager.DirtyImager`, CuPy)."""

from typing import Any

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.core import ExecutionContext, Fragment, InputContext, Operator, OperatorSpec, OutputContext

from kremetart.imaging.dirty_imager import DirtyImager
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.operators.snapshot_ports import declare_snapshot_inputs, receive_snapshot
from kremetart.utils.beam import GROUND_PLANE_DIAMETER


class HealpixDFTOperator(Operator):
    """Snapshot in; dirty map (``cube``, ``(1, npix)``) out.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        freqs: ``(nchan,)`` frequencies in Hz.
        nest: NESTED HEALPix ordering.
        apply_beam: fold the Airy primary beam into the measurement operator.
        ground_plane_diameter: Airy aperture diameter in metres.
    """

    def __init__(
        self,
        fragment: Fragment,
        nside: int,
        freqs: np.ndarray,
        *args: Any,
        nest: bool = True,
        apply_beam: bool = True,
        ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
        **kwargs: Any,
    ) -> None:
        self.grid_args = dict(nest=nest, apply_beam=apply_beam, ground_plane_diameter=ground_plane_diameter)
        self.nside = nside
        self.freqs_host = freqs
        super().__init__(fragment, *args, **kwargs)

    def start(self) -> None:
        self.freqs = cp.asarray(self.freqs_host)
        self.stage = DirtyImager(SkyGrid.build(self.nside, xp=cp, **self.grid_args))

    def setup(self, spec: OperatorSpec) -> None:
        declare_snapshot_inputs(spec)
        spec.output("cube")

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext) -> None:
        snapshot = receive_snapshot(op_input, self.freqs)
        dirty = self.stage.image(snapshot)
        op_output.emit(hs.as_tensor(dirty[None, :]), "cube")
