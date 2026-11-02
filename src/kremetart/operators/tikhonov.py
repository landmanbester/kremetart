"""Holoscan adapter for the Tikhonov solver stage (:class:`kremetart.imaging.tikhonov_solver.TikhonovSolver`, CuPy)."""

from typing import Any

import cupy as cp
import holoscan as hs
import numpy as np
from holoscan.core import ExecutionContext, Fragment, InputContext, Operator, OperatorSpec, OutputContext

from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.tikhonov_solver import TikhonovSolver
from kremetart.operators.snapshot_ports import declare_snapshot_inputs, receive_snapshot
from kremetart.utils.beam import GROUND_PLANE_DIAMETER


class TikhonovOperator(Operator):
    """Snapshot plus dirty map (``cube``) in; Tikhonov map (``cube``) out.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        freqs: ``(nchan,)`` frequencies in Hz.
        eta: regularisation strength as a fraction of the total weight.
        nest, apply_beam, ground_plane_diameter: sky grid settings (must match the dirty imager's).
        maxiter, tol, use_preconditioner, use_warm_start: :class:`TikhonovSolver` options.
    """

    def __init__(
        self,
        fragment: Fragment,
        nside: int,
        freqs: np.ndarray,
        eta: float,
        *args: Any,
        nest: bool = True,
        apply_beam: bool = True,
        ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
        maxiter: int = 100,
        tol: float = 1e-5,
        use_preconditioner: bool = True,
        use_warm_start: bool = True,
        **kwargs: Any,
    ) -> None:
        self.grid_args = dict(nest=nest, apply_beam=apply_beam, ground_plane_diameter=ground_plane_diameter)
        self.solver_args = dict(
            eta=eta, maxiter=maxiter, tol=tol, use_preconditioner=use_preconditioner, use_warm_start=use_warm_start
        )
        self.nside = nside
        self.freqs_host = freqs
        super().__init__(fragment, *args, **kwargs)

    def start(self) -> None:
        self.freqs = cp.asarray(self.freqs_host)
        self.stage = TikhonovSolver(SkyGrid.build(self.nside, xp=cp, **self.grid_args), **self.solver_args)

    def setup(self, spec: OperatorSpec) -> None:
        declare_snapshot_inputs(spec)
        spec.input("cube")  # dirty map from the dirty imager
        spec.output("cube")

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext) -> None:
        snapshot = receive_snapshot(op_input, self.freqs)
        dirty = cp.asarray(op_input.receive("cube"))[0]
        x = self.stage.solve(snapshot, dirty)
        op_output.emit(hs.as_tensor(x[None, :]), "cube")
