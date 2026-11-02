"""The sky grid: the HEALPix pixelisation and primary-beam model every map in a run lives on."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np

from kremetart.imaging.healpix import make_pixel_grid
from kremetart.imaging.snapshot import Array
from kremetart.utils.beam import GROUND_PLANE_DIAMETER, airy_power_beam


@dataclass(frozen=True)
class SkyGrid:
    """Run-level imaging geometry, built once on the device given by ``xp``.

    Attributes:
        nside: HEALPix resolution.
        nest: NESTED pixel ordering.
        pix_vec: ``(npix, 3)`` pixel unit vectors (equatorial ICRS frame).
        apply_beam: fold the Airy primary beam into the measurement operator.
        ground_plane_diameter: Airy aperture diameter in metres.
        xp: array module (``numpy`` or ``cupy``) holding ``pix_vec`` and every map on this grid.
    """

    nside: int
    nest: bool
    pix_vec: Array
    apply_beam: bool
    ground_plane_diameter: float
    xp: ModuleType

    @classmethod
    def build(
        cls,
        nside: int,
        *,
        nest: bool = True,
        apply_beam: bool = True,
        ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
        xp: ModuleType = np,
    ) -> SkyGrid:
        """Build the pixel grid for ``nside`` on the device of ``xp``."""
        pix_vec = make_pixel_grid(nside, nest=nest, xp=xp)
        return cls(nside, nest, pix_vec, apply_beam, float(ground_plane_diameter), xp)

    @property
    def npix(self) -> int:
        return int(self.pix_vec.shape[0])

    def beam(self, boresight: Array, freqs: Array) -> Array | None:
        """The ``(nchan, npix)`` primary beam for one pointing, or ``None`` when the beam is off."""
        if not self.apply_beam:
            return None
        return airy_power_beam(self.pix_vec, boresight, freqs, diameter=self.ground_plane_diameter, xp=self.xp)

    def zeros(self) -> Array:
        """An all-zero float64 map: what every stage emits for a snapshot that carries no data."""
        return self.xp.zeros(self.npix, dtype=self.xp.float64)
