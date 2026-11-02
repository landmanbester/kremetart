"""Holoscan adapter for the dirty imager stage (:class:`kremetart.imaging.dirty_imager.DirtyImager`, CuPy)."""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

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
        fragment,
        nside,
        freqs,
        *args,
        nest=True,
        apply_beam=True,
        ground_plane_diameter=GROUND_PLANE_DIAMETER,
        **kwargs,
    ):
        self.grid_args = dict(nest=nest, apply_beam=apply_beam, ground_plane_diameter=ground_plane_diameter)
        self.nside = nside
        self.freqs_host = freqs
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        self.freqs = cp.asarray(self.freqs_host)
        self.stage = DirtyImager(SkyGrid.build(self.nside, xp=cp, **self.grid_args))

    def setup(self, spec: OperatorSpec):
        declare_snapshot_inputs(spec)
        spec.output("cube")

    def compute(self, op_input, op_output, context):
        snapshot = receive_snapshot(op_input, self.freqs)
        dirty = self.stage.image(snapshot)
        op_output.emit(hs.as_tensor(dirty[None, :]), "cube")
