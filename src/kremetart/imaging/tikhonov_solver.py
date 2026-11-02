"""The Tikhonov solver stage: snapshot + dirty map -> Tikhonov map, via preconditioned CG."""

from __future__ import annotations

from kremetart.imaging.normal_equations import build_normal_equations, warm_start
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.snapshot import Array, Snapshot
from kremetart.opt.cg import cg


class TikhonovSolver:
    """Solve ``(H + λI) x = b`` per snapshot, with ``λ = eta·Σw``.

    State across snapshots: the last solution, used as the next snapshot's CG starting point when
    ``use_warm_start`` is set. A snapshot with no data yields an all-zero map and leaves that state
    untouched, so the next snapshot with data still starts from the last good solution.

    Args:
        grid: the sky grid.
        eta: regularisation strength as a fraction of the total weight; must be > 0.
        maxiter: maximum CG iterations per snapshot.
        tol: CG relative-residual tolerance.
        use_preconditioner: apply the Jacobi preconditioner ``1/(diag(H) + λ)``.
        use_warm_start: start each snapshot's CG from the previous solution.
    """

    def __init__(
        self,
        grid: SkyGrid,
        *,
        eta: float,
        maxiter: int = 100,
        tol: float = 1e-5,
        use_preconditioner: bool = True,
        use_warm_start: bool = True,
    ):
        self.grid = grid
        self.eta = float(eta)
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.use_preconditioner = use_preconditioner
        self.use_warm_start = use_warm_start
        self._x_prev = None

    def solve(self, snapshot: Snapshot, dirty: Array) -> Array:
        """Return the ``(npix,)`` Tikhonov map for ``snapshot``, given its dirty map."""
        xp = self.grid.xp
        eqs = build_normal_equations(self.grid, snapshot, dirty, self.eta)
        if eqs is None:
            return self.grid.zeros()

        lam = eqs.lam

        def a_matvec(x):
            return eqs.hessian(x) + lam * x

        precond = None
        if self.use_preconditioner:
            inv_mdiag = 1.0 / (eqs.diagonal + lam)

            def precond(r):
                return r * inv_mdiag

        x0 = warm_start(self._x_prev, self.use_warm_start, xp)
        x = cg(a_matvec, eqs.rhs, x0=x0, M=precond, maxiter=self.maxiter, tol=self.tol, xp=xp)
        self._x_prev = x
        return x
