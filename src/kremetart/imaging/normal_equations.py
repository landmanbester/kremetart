"""The per-snapshot normal equations shared by the deconvolving stages (Tikhonov and L1 solvers).

Each solver builds its own normal equations (and so its own DFT kernel) per snapshot; see
``docs/adr/0001-independent-imaging-stages.md``. This module only removes the duplicated arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kremetart.imaging.healpix import hessian_healpix
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.snapshot import Array, Snapshot


@dataclass(frozen=True)
class NormalEquations:
    """``H x = b`` for one snapshot, with ``λ = eta·Σw`` making ``eta`` snapshot-invariant.

    Attributes:
        hessian: ``x -> H x`` with ``H = B Mᴴ W M B`` (real, ``(npix,) -> (npix,)``).
        diagonal: closed-form ``diag(H)``.
        rhs: ``b``, the dirty map un-normalised by the total weight.
        lam: the regularisation weight ``eta·Σw``.
    """

    hessian: Callable[[Array], Array]
    diagonal: Array
    rhs: Array
    lam: Array


def build_normal_equations(grid: SkyGrid, snapshot: Snapshot, dirty: Array, eta: float) -> NormalEquations | None:
    """Normal equations for ``snapshot`` given its dirty map, or ``None`` if it carries no data.

    With every weight zero, ``λ`` is zero and ``H`` is singular, and the dirty map is the all-zero
    no-data map: there is nothing to solve.
    """
    w = grid.xp.asarray(snapshot.weight)
    wsum = w.sum()
    if float(wsum) == 0.0:
        return None
    beam = grid.beam(snapshot.boresight, snapshot.freqs)
    hmv, hdiag = hessian_healpix(snapshot.b_rot, grid.pix_vec, snapshot.freqs, w, beam=beam, xp=grid.xp)
    return NormalEquations(hessian=hmv, diagonal=hdiag, rhs=dirty * wsum, lam=eta * wsum)


def warm_start(previous: Array | None, enabled: bool, xp) -> Array | None:
    """The previous solution as a starting point, if enabled and finite; otherwise ``None`` (cold start)."""
    if enabled and previous is not None and bool(xp.all(xp.isfinite(previous))):
        return previous
    return None
