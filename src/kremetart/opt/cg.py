"""Generic (preconditioned) conjugate-gradient solver for SPD systems.

The solver is ``xp``-injectable (``numpy`` on the host, ``cupy`` on the GPU) and operator-agnostic:
the linear operator ``A`` and the preconditioner ``M`` are plain callables ``x -> A @ x`` and
``r -> M^{-1} r``. Plain callables (rather than a ``scipy.LinearOperator``, which is host/NumPy
oriented) keep the same code running on cupy arrays without host round-trips.

Used by the streaming Tikhonov imager to solve ``(H + lambda I) x = b`` per frame, where ``H`` is
the HEALPix image-space Hessian (:func:`kremetart.utils.healpix_dft.hessian_healpix`). The optional
``x0`` warm-start and ``M`` preconditioner default off so the adjointness/reference tests compare
against the unpreconditioned, zero-initialised solution.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import numpy as np


def cg(
    A: Callable,
    b,
    *,
    x0=None,
    M: Callable | None = None,
    maxiter: int = 100,
    tol: float = 1e-5,
    xp: ModuleType = np,
):
    """Solve ``A x = b`` for symmetric positive-definite ``A`` via (preconditioned) CG.

    Args:
        A: linear operator, a callable ``x -> A @ x`` (must be SPD over the reals).
        b: right-hand side, shape ``(n,)``.
        x0: optional warm-start initial guess (same shape as ``b``); ``None`` -> zeros.
        M: optional preconditioner, a callable ``r -> M^{-1} r`` (e.g. Jacobi); ``None`` ->
            unpreconditioned.
        maxiter: maximum CG iterations.
        tol: relative-residual stopping tolerance; iterate until ``||r|| <= tol * ||b||``.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(n,)`` solution ``x`` (same dtype/precision as ``b``).
    """
    b = xp.asarray(b)
    x = xp.zeros_like(b) if x0 is None else xp.asarray(x0).copy()

    r = b - A(x)
    z = r if M is None else M(r)
    p = z.copy()
    rz = xp.vdot(r, z).real

    b_norm = xp.sqrt(xp.vdot(b, b).real)
    if float(b_norm) == 0.0:
        return x  # b == 0 -> x = 0 (or the warm start, which the data does not constrain)
    thresh = tol * b_norm

    for _ in range(maxiter):
        if xp.sqrt(xp.vdot(r, r).real) <= thresh:
            break
        ap = A(p)
        alpha = rz / xp.vdot(p, ap).real
        x = x + alpha * p
        r = r - alpha * ap
        z = r if M is None else M(r)
        rz_new = xp.vdot(r, z).real
        p = z + (rz_new / rz) * p
        rz = rz_new

    return x
