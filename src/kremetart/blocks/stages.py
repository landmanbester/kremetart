"""Stage blocks: receive a snapshot (ports 0-4, plus the dirty map on port 5 for the solvers), call the stage,
write its map(s) to the outputs.
"""

from __future__ import annotations

from typing import Any

from kremetart.blocks.snapshot_ports import SNAPSHOT_PORTS, array_module, receive_snapshot, synchronize
from kremetart.imaging.dirty_imager import DirtyImager
from kremetart.imaging.iwp_filter import IWPFilter
from kremetart.imaging.l1_solver import L1Solver
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.tikhonov_solver import TikhonovSolver

DIRTY_PORT = len(SNAPSHOT_PORTS)
"""The solvers' input port for the dirty map (after the snapshot's ports)."""


class _GridBlock:
    """A block whose stage needs the sky grid and the channel frequencies."""

    def __init__(
        self,
        *,
        device: str,
        nside: int,
        nest: bool,
        apply_beam: bool,
        ground_plane_diameter: float,
        freqs: list[float],
    ) -> None:
        self.xp = array_module(device)
        self.freqs = self.xp.asarray(freqs)
        self.grid = SkyGrid.build(
            nside, nest=nest, apply_beam=apply_beam, ground_plane_diameter=ground_plane_diameter, xp=self.xp
        )


class DirtyImagerBlock(_GridBlock):
    """Snapshot in; dirty map out."""

    def __init__(self, **grid: Any) -> None:
        super().__init__(**grid)
        self.stage = DirtyImager(self.grid)

    def compute(self, ctx: Any) -> None:
        ctx.outputs[0][...] = self.stage.image(receive_snapshot(ctx, self.freqs, self.xp))
        synchronize(self.xp)


class TikhonovSolverBlock(_GridBlock):
    """Snapshot and dirty map in; Tikhonov map out."""

    def __init__(self, *, eta: float, **grid: Any) -> None:
        super().__init__(**grid)
        self.stage = TikhonovSolver(self.grid, eta=eta)

    def compute(self, ctx: Any) -> None:
        snapshot = receive_snapshot(ctx, self.freqs, self.xp)
        ctx.outputs[0][...] = self.stage.solve(snapshot, self.xp.array(ctx.inputs[DIRTY_PORT]))
        synchronize(self.xp)


class L1SolverBlock(_GridBlock):
    """Snapshot and dirty map in; L1 map out."""

    def __init__(self, *, eta: float, **grid: Any) -> None:
        super().__init__(**grid)
        self.stage = L1Solver(self.grid, eta=eta)

    def compute(self, ctx: Any) -> None:
        snapshot = receive_snapshot(ctx, self.freqs, self.xp)
        ctx.outputs[0][...] = self.stage.solve(snapshot, self.xp.array(ctx.inputs[DIRTY_PORT]))
        synchronize(self.xp)


class IWPFilterBlock:
    """Map (port 0) and time (port 1) in; smooth map (output 0) and normalised innovation (output 1) out."""

    def __init__(self, *, device: str, npix: int, sigma2: float, noise: float) -> None:
        self.xp = array_module(device)
        self.stage = IWPFilter(npix, sigma2=sigma2, noise=noise, xp=self.xp)

    def compute(self, ctx: Any) -> None:
        smooth, innovation = self.stage.update(self.xp.array(ctx.inputs[0]), float(ctx.inputs[1][0]))
        ctx.outputs[0][...] = smooth
        ctx.outputs[1][...] = innovation
        synchronize(self.xp)
