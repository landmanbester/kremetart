"""The Snapshot: everything one stage needs to image one timestamp (see ``GLOSSARY.md``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Array = Any  # a numpy or cupy array; every array in a Snapshot lives on the same device


@dataclass(frozen=True)
class Snapshot:
    """The correlations of every baseline at one timestamp, plus what imaging them needs.

    Flagged data is expressed as zero ``weight``; a snapshot whose weights are all zero carries no
    data, and every stage treats it as such (zero maps, state untouched or coasting).

    Attributes:
        vis: ``(nbl, nchan)`` complex visibilities.
        weight: ``(nbl, nchan)`` real, gain-corrected weights.
        b_rot: ``(nbl, 3)`` rotated baselines in metres (the celestial frame of the sky grid).
        boresight: ``(3,)`` unit vector of the primary-beam pointing in the same frame.
        time: timestamp in unix seconds.
        freqs: ``(nchan,)`` channel frequencies in Hz.
    """

    vis: Array
    weight: Array
    b_rot: Array
    boresight: Array
    time: float
    freqs: Array

    def total_weight(self) -> float:
        """Sum of all weights; ``0.0`` means the snapshot carries no data."""
        return float(self.weight.sum())
