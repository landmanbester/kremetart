"""Imaging settings shared by every runner, and the names their maps are stored and streamed under."""

from __future__ import annotations

from dataclasses import dataclass

from kremetart.utils.beam import GROUND_PLANE_DIAMETER

STORED_NAMES: tuple[str, ...] = ("dirty", "tikhonov", "l1", "filtered", "znorm")
"""Map-store variable names, in order. The live stream calls ``filtered`` ``smooth`` (see ``STREAM_NAME``)."""

STREAM_NAME: dict[str, str] = {
    "dirty": "dirty",
    "tikhonov": "tikhonov",
    "l1": "l1",
    "filtered": "smooth",
    "znorm": "znorm",
}
"""Live-stream name of each stored map."""


@dataclass(frozen=True)
class ImagingSettings:
    """Everything a runner needs to build the stages.

    Attributes:
        nside: HEALPix resolution.
        nest: NESTED pixel ordering (RING if ``False``); the viewer must use the same.
        l2: Tikhonov strength as a fraction of the snapshot weight sum; ``None`` disables Tikhonov maps
            (the IWP filter then smooths the dirty maps).
        l1: reweighted-L1 strength as a fraction of the weight sum; ``None`` disables L1 maps.
        iwp_sigma: IWP driving variance (sigma^2).
        iwp_noise: IWP measurement-noise variance (R).
        apply_beam: fold the Airy primary beam into the measurement operator.
        ground_plane_diameter: Airy aperture diameter in metres.
    """

    nside: int
    nest: bool = True
    l2: float | None = None
    l1: float | None = None
    iwp_sigma: float = 1e-3
    iwp_noise: float = 1e-2
    apply_beam: bool = True
    ground_plane_diameter: float = GROUND_PLANE_DIAMETER

    @property
    def npix(self) -> int:
        """Pixels per map."""
        return 12 * self.nside * self.nside

    @property
    def stored_names(self) -> tuple[str, ...]:
        """The map-store variables this run writes (``tikhonov`` and ``l1`` only when enabled)."""
        off = {name for name, strength in (("tikhonov", self.l2), ("l1", self.l1)) if strength is None}
        return tuple(name for name in STORED_NAMES if name not in off)

    @property
    def stream_names(self) -> tuple[str, ...]:
        """The live-stream names this run publishes."""
        return tuple(STREAM_NAME[name] for name in self.stored_names)
