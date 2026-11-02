"""The L1 solver stage: snapshot + dirty map -> L1 map, via (reweighted) FISTA."""

from __future__ import annotations

from kremetart.imaging.normal_equations import build_normal_equations, warm_start
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.snapshot import Array, Snapshot
from kremetart.opt.fista import fista_quadratic


class L1Solver:
    """Minimise ``½⟨x,Hx⟩ − ⟨b,x⟩ + λ Σ wᵢ|xᵢ|`` per snapshot, with ``λ = eta·Σw``.

    The Lipschitz step is seeded from ``max diag(H)``. State across snapshots: the last solution, used
    as the next starting point when ``use_warm_start`` is set; a snapshot with no data yields an
    all-zero map and leaves that state untouched.

    Args:
        grid: the sky grid.
        eta: regularisation strength as a fraction of the total weight; must be > 0.
        max_iter: maximum inner FISTA iterations per reweighting round.
        tol: inner relative-change tolerance.
        max_reweight: outer Candès–Wakin–Boyd reweighting rounds.
        reweight_eps: ``ε`` in ``wᵢ = 1/(|xᵢ| + ε)``.
        positive: enforce ``x >= 0``.
        use_warm_start: start each snapshot's FISTA from the previous solution.
    """

    def __init__(
        self,
        grid: SkyGrid,
        *,
        eta: float,
        max_iter: int = 200,
        tol: float = 1e-5,
        max_reweight: int = 2,
        reweight_eps: float = 1e-3,
        positive: bool = True,
        use_warm_start: bool = True,
    ) -> None:
        self.grid = grid
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_reweight = int(max_reweight)
        self.reweight_eps = float(reweight_eps)
        self.positive = positive
        self.use_warm_start = use_warm_start
        self._x_prev = None
        self._last_info: dict | None = None

    @property
    def last_info(self) -> dict | None:
        """FISTA diagnostics (iterations, reweights, convergence) for the last snapshot that had data."""
        return self._last_info

    def solve(self, snapshot: Snapshot, dirty: Array) -> Array:
        """Return the ``(npix,)`` L1 map for ``snapshot``, given its dirty map."""
        xp = self.grid.xp
        eqs = build_normal_equations(self.grid, snapshot, dirty, self.eta)
        if eqs is None:
            return self.grid.zeros()

        x0 = warm_start(self._x_prev, self.use_warm_start, xp)
        x, self._last_info = fista_quadratic(
            eqs.hessian,
            eqs.rhs,
            lam=float(eqs.lam),
            x0=x0,
            positive=self.positive,
            L0=float(eqs.diagonal.max()),
            max_iter=self.max_iter,
            tol=self.tol,
            max_reweight=self.max_reweight,
            reweight_eps=self.reweight_eps,
            xp=xp,
        )
        self._x_prev = x
        return x
