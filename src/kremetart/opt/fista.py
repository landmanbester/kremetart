"""Reweighted-L1 FISTA solver (xp-injectable; CPU via numpy, GPU via cupy).

Generic and operator-agnostic, like :mod:`kremetart.opt.cg`: the forward operator ``A`` and its
Hermitian adjoint ``AH`` are plain callables ``x -> A @ x`` and ``r -> Aᴴ @ r``. Minimises, over a
**real** vector ``x``, ``0.5 ||W^½ (A x - y)||² + λ Σ wᵢ |xᵢ|`` with FISTA + backtracking and an
outer Candès–Wakin–Boyd reweighting loop. Used at the outset of self-calibration to recover
non-negative source fluxes from calibrated visibilities; the caller wires ``A`` to the imaging DFT
or the per-source sky model (out of scope here, mirroring how the imager wires up ``cg``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np


def _soft_threshold(z, tau, *, positive: bool, xp: ModuleType = np):
    """Proximal operator of the (optionally non-negative) weighted-L1 penalty.

    Args:
        z: real array, the point at which to evaluate the prox.
        tau: scalar or array (broadcastable to ``z``) of non-negative thresholds.
        positive: if True, clamp at 0 (prox of L1 + non-negativity indicator).
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``max(z - tau, 0)`` if ``positive`` else ``sign(z) * max(|z| - tau, 0)``.
    """
    if positive:
        return xp.maximum(z - tau, 0.0)
    return xp.sign(z) * xp.maximum(xp.abs(z) - tau, 0.0)


def _objective(A: Callable, y, weight, x, lam: float, xp: ModuleType) -> float:
    """Full objective ``0.5 Σ W|A x − y|² + λ Σ|x|`` (plain L1, for diagnostics)."""
    r = A(x) - y
    sq = xp.abs(r) ** 2
    if weight is not None:
        sq = weight * sq
    return 0.5 * float(sq.sum()) + lam * float(xp.abs(x).sum())


def _fista_single(
    A: Callable,
    AH: Callable,
    y,
    w,
    x_start,
    *,
    lam: float,
    weight,
    positive: bool,
    L0: float | None,
    eta: float,
    max_iter: int,
    tol: float,
    xp: ModuleType,
):
    """FISTA with backtracking for one fixed reweighting vector ``w``.

    Returns:
        ``(x, iters, converged, lipschitz)``.
    """

    def fval(r):
        sq = xp.abs(r) ** 2
        if weight is not None:
            sq = weight * sq
        return 0.5 * float(sq.sum())

    def grad_at(r):
        wr = r if weight is None else weight * r
        return xp.real(AH(wr))  # x is real -> take the real part of the complex adjoint

    threshold_w = lam * w  # per-element L1 weight (λ · reweight weight)
    lipschitz = 1.0 if L0 is None else float(L0)
    x = xp.asarray(x_start).copy()
    x_prev = x.copy()
    v = x.copy()
    t = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        r_v = A(v) - y
        f_v = fval(r_v)
        g_v = grad_at(r_v)
        bt = 0
        while True:
            z = v - g_v / lipschitz
            x_new = _soft_threshold(z, threshold_w / lipschitz, positive=positive, xp=xp)
            diff = x_new - v
            rhs = f_v + float((diff * g_v).sum()) + 0.5 * lipschitz * float((diff * diff).sum())
            # +1e-12: numerical slack so float round-off near equality doesn't force a needless backtrack
            if fval(A(x_new) - y) <= rhs + 1e-12 or bt >= 100:
                break
            lipschitz *= eta
            bt += 1
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        v = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x_prev = x
        x = x_new
        t = t_new
        change = float(xp.linalg.norm(x - x_prev))
        scale = max(float(xp.linalg.norm(x)), 1e-12)
        if change / scale < tol:
            converged = True
            break
    return x, iters, converged, lipschitz


def fista(
    A: Callable,
    AH: Callable,
    y,
    *,
    lam: float,
    weight=None,
    x0=None,
    positive: bool = True,
    L0: float | None = None,
    eta: float = 2.0,
    max_iter: int = 500,
    tol: float = 1e-5,
    max_reweight: int = 0,
    reweight_eps: float = 1e-3,
    reweight_tol: float = 1e-3,
    xp: ModuleType = np,
):
    """Minimise ``0.5 ||W^½(A x − y)||² + λ Σ wᵢ|xᵢ|`` over real ``x`` via reweighted-L1 FISTA.

    Args:
        A: forward operator, a callable ``x -> A @ x`` (real ``x`` -> complex output).
        AH: Hermitian adjoint of ``A``, a callable ``r -> Aᴴ @ r``.
        y: ``(m,)`` complex data (any shape ``A`` returns).
        lam: L1 strength ``λ`` (``0`` allowed -> plain least squares).
        weight: optional real inverse-variance ``W`` broadcastable to ``y``; ``None`` -> ones.
        x0: optional real warm start; ``None`` -> zeros.
        positive: enforce ``x >= 0`` in the prox.
        L0: initial Lipschitz estimate; ``None`` -> ``1.0``.
        eta: backtracking growth factor (> 1).
        max_iter: max inner FISTA iterations per reweight round.
        tol: inner relative-change stopping tolerance.
        max_reweight: outer reweighting rounds (``0`` -> plain L1).
        reweight_eps: ``ε`` in ``wᵢ = 1/(|xᵢ| + ε)``.
        reweight_tol: outer between-round relative-change stop.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(x, info)``: real ``x`` and a dict with ``iterations``, ``reweights``, ``objective``,
        ``lipschitz``, ``converged``.
    """
    y = xp.asarray(y)
    weight = None if weight is None else xp.asarray(weight)
    real_dtype = y.real.dtype
    if x0 is None:
        x = xp.zeros(AH(y).shape, dtype=real_dtype)
    else:
        x = xp.asarray(x0, dtype=real_dtype).copy()
    w = xp.ones(x.shape, dtype=real_dtype)

    iterations: list[int] = []
    converged = False
    lipschitz = 1.0 if L0 is None else float(L0)
    for ell in range(max(max_reweight, 0) + 1):  # negative count -> a single plain-L1 solve
        x_prev_round = x.copy()
        x, iters, converged, lipschitz = _fista_single(
            A,
            AH,
            y,
            w,
            x,
            lam=lam,
            weight=weight,
            positive=positive,
            L0=L0,
            eta=eta,
            max_iter=max_iter,
            tol=tol,
            xp=xp,
        )
        iterations.append(iters)
        if ell == max_reweight:
            break
        w = 1.0 / (xp.abs(x) + reweight_eps)  # Candès–Wakin–Boyd reweighting
        denom = max(float(xp.linalg.norm(x_prev_round)), 1e-12)
        if float(xp.linalg.norm(x - x_prev_round)) / denom < reweight_tol:
            break  # support / values have stabilised

    info = {
        "iterations": iterations,
        "reweights": len(iterations) - 1,
        "objective": _objective(A, y, weight, x, lam, xp),
        "lipschitz": lipschitz,
        "converged": converged,
    }
    return x, info
