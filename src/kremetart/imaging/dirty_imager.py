"""The dirty imager stage: snapshot -> dirty map on the sky grid."""

from __future__ import annotations

from kremetart.imaging.healpix import dirty_map
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.snapshot import Array, Snapshot


class DirtyImager:
    """Weighted adjoint DFT of a snapshot onto the sky grid, beam-weighted when the grid has a beam.

    The map is normalised by the snapshot's total weight; a snapshot with no data yields an all-zero
    map, which downstream stages read as "no data". Stateless across snapshots.
    """

    def __init__(self, grid: SkyGrid):
        self.grid = grid

    def image(self, snapshot: Snapshot) -> Array:
        """Return the ``(npix,)`` real dirty map of ``snapshot``."""
        xp = self.grid.xp
        beam = self.grid.beam(snapshot.boresight, snapshot.freqs)
        return dirty_map(
            snapshot.vis, snapshot.weight, snapshot.b_rot, self.grid.pix_vec, snapshot.freqs, beam=beam, xp=xp
        )
