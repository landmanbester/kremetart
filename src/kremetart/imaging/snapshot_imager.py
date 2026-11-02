"""The snapshot imager: the fixed chain of stages that turns each snapshot into all of its maps.

dirty imager -> Tikhonov solver -> IWP filter, with the optional L1 solver alongside the Tikhonov
solver. The Tikhonov solver is optional too: the IWP filter smooths the Tikhonov maps when they are
produced and the dirty maps otherwise. This is the in-process composition used by the CPU
loop runner, scripts and end-to-end tests; streaming runtimes (Holoscan today, CyberEther later) wire
the same stages one per operator/block instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np

from kremetart.imaging.dirty_imager import DirtyImager
from kremetart.imaging.iwp_filter import IWPFilter
from kremetart.imaging.l1_solver import L1Solver
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.snapshot import Array, Snapshot
from kremetart.imaging.tikhonov_solver import TikhonovSolver
from kremetart.utils.beam import GROUND_PLANE_DIAMETER


@dataclass(frozen=True)
class SnapshotMaps:
    """Every map the snapshot imager produces for one snapshot (each ``(npix,)``; ``None`` when off)."""

    time: float
    dirty: Array
    tikhonov: Array | None
    smooth: Array
    normalised_innovation: Array
    l1: Array | None = None


class SnapshotImager:
    """Run the stages over snapshots in time order, keeping each stage's state between snapshots.

    Args:
        dirty_imager: the dirty imager stage.
        tikhonov_solver: optional Tikhonov solver stage; ``None`` disables Tikhonov maps.
        iwp_filter: the IWP filter stage (smooths the Tikhonov maps, or the dirty maps without them).
        l1_solver: optional L1 solver stage; ``None`` disables L1 maps.
    """

    def __init__(
        self,
        dirty_imager: DirtyImager,
        tikhonov_solver: TikhonovSolver | None,
        iwp_filter: IWPFilter,
        l1_solver: L1Solver | None = None,
    ) -> None:
        self.dirty_imager = dirty_imager
        self.tikhonov_solver = tikhonov_solver
        self.iwp_filter = iwp_filter
        self.l1_solver = l1_solver

    @classmethod
    def build(
        cls,
        nside: int,
        *,
        l2: float | None = None,
        l1: float | None = None,
        sigma2: float = 1e-3,
        noise: float = 1e-2,
        nest: bool = True,
        apply_beam: bool = True,
        ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
        xp: ModuleType = np,
    ) -> SnapshotImager:
        """The standard chain with default solver settings (matching the ``smoovie`` pipeline)."""
        grid = SkyGrid.build(
            nside, nest=nest, apply_beam=apply_beam, ground_plane_diameter=ground_plane_diameter, xp=xp
        )
        return cls(
            DirtyImager(grid),
            None if l2 is None else TikhonovSolver(grid, eta=l2),
            IWPFilter(grid.npix, sigma2=sigma2, noise=noise, xp=xp),
            None if l1 is None else L1Solver(grid, eta=l1),
        )

    def image(self, snapshot: Snapshot) -> SnapshotMaps:
        """Produce every map for ``snapshot`` (call in time order)."""
        dirty = self.dirty_imager.image(snapshot)
        tikhonov = None if self.tikhonov_solver is None else self.tikhonov_solver.solve(snapshot, dirty)
        l1 = None if self.l1_solver is None else self.l1_solver.solve(snapshot, dirty)
        smooth, innovation = self.iwp_filter.update(dirty if tikhonov is None else tikhonov, snapshot.time)
        return SnapshotMaps(snapshot.time, dirty, tikhonov, smooth, innovation, l1)
