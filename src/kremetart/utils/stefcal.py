"""Acquisition StEFCal: alternating per-antenna complex-gain least squares.

The cold-start solver of the Stage-1 calibration operator
(docs/superpowers/specs/2026-06-24-stefcal-calibration-core-design.md). Holding all other antenna
gains fixed, ``V_pq = g_p M_pq conj(g_q)`` is linear in ``g_p``; alternating that antenna-by-antenna
gives a robust, phase-wrap-free cold start. Every source is modelled at unit flux, so the solved
amplitudes are unreliable and only the gauge-referenced phases are kept downstream. ``xp``-injectable
(``xp=numpy`` CPU / ``xp=cupy`` GPU); the per-antenna reduction is a segment sum over the bipartite
baseline-antenna incidence.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def _seg_sum_complex(idx, vals, n: int, *, xp: ModuleType):
    """Per-segment complex sum: ``out[k] = sum(vals[idx == k])`` over ``n`` segments."""
    real = xp.bincount(idx, weights=vals.real, minlength=n)
    imag = xp.bincount(idx, weights=vals.imag, minlength=n)
    return real + 1j * imag


def stefcal_solve(
    vis,
    model,
    a1,
    a2,
    n_ant: int,
    *,
    ref_ant: int = 0,
    weight=None,
    g0=None,
    max_iter: int = 100,
    tol: float = 1e-8,
    xp: ModuleType = np,
):
    """Solve per-antenna complex gains by alternating least squares (StEFCal).

    Args:
        vis: ``(nbl, nchan)`` observed visibilities.
        model: ``(nbl, nchan)`` unit-flux model visibilities
            (:func:`kremetart.utils.skymodel.model_visibilities`).
        a1: ``(nbl,)`` int antenna index of the first antenna of each baseline.
        a2: ``(nbl,)`` int antenna index of the second antenna of each baseline.
        n_ant: number of antennas.
        ref_ant: reference antenna whose gain is pinned to 1 (fixes both gauges). Must be live.
        weight: optional ``(nbl,)`` or ``(nbl, nchan)`` real weights; ``0`` flags a baseline out.
        g0: optional ``(n_ant,)`` complex initial gains (warm start); defaults to unity.
        max_iter: maximum alternating iterations.
        tol: convergence threshold on the max relative gain change over live antennas.
        xp: array module.

    Returns:
        ``(gains, info)``: ``gains`` is ``(n_ant,)`` complex with ``gains[ref_ant] == 1`` and dead
        antennas ``NaN``; ``info`` has ``iterations`` (int), ``converged`` (bool), ``max_change``
        (float, the last iteration's max relative change).

    Raises:
        ValueError: if ``ref_ant`` has no live (non-zero-weight) baselines.
    """
    vis = xp.asarray(vis)
    model = xp.asarray(model)
    a1 = xp.asarray(a1)
    a2 = xp.asarray(a2)
    nbl, nchan = vis.shape
    if weight is None:
        weight = xp.ones((nbl, nchan))
    else:
        weight = xp.asarray(weight)
        if weight.ndim == 1:
            weight = weight[:, None]
        weight = xp.broadcast_to(weight, (nbl, nchan))

    # Directed baselines: forward (p=a1, partner=a2, V, M) + reverse (p=a2, partner=a1, conjV, conjM).
    p_idx = xp.concatenate([a1, a2])
    q_idx = xp.concatenate([a2, a1])
    vdir = xp.concatenate([vis, xp.conj(vis)], axis=0)  # (2*nbl, nchan)
    mdir = xp.concatenate([model, xp.conj(model)], axis=0)
    wdir = xp.concatenate([weight, weight], axis=0)

    # Live antennas: those carrying at least one non-zero-weight baseline.
    deg = xp.bincount(p_idx, weights=wdir.sum(axis=1), minlength=n_ant)  # (n_ant,) total weight
    live = deg > 0
    if not bool(live[ref_ant]):
        raise ValueError(f"ref_ant {ref_ant} has no live baselines")

    g = xp.ones(n_ant, dtype=xp.complex128) if g0 is None else xp.asarray(g0).astype(xp.complex128)
    converged = False
    change = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        z = mdir * xp.conj(g[q_idx])[:, None]  # (2*nbl, nchan): V_dir = g[p] * z
        num = _seg_sum_complex(p_idx, (wdir * xp.conj(z) * vdir).sum(axis=1), n_ant, xp=xp)
        den = xp.bincount(p_idx, weights=(wdir * (z.real**2 + z.imag**2)).sum(axis=1), minlength=n_ant)
        g_new = xp.where(den > 0, num / xp.where(den > 0, den, 1.0), g)
        if it % 2 == 0:
            g_new = 0.5 * (g_new + g)  # StEFCal even-iteration stabiliser
        delta = xp.abs(g_new - g)
        scale = xp.where(xp.abs(g) > 1e-12, xp.abs(g), 1.0)
        change = float(xp.max(xp.where(live, delta / scale, 0.0)))
        g = g_new
        if change < tol:
            converged = True
            break

    nan_c = complex(float("nan"), float("nan"))
    g = xp.where(live, g, nan_c)
    g = g / g[ref_ant]  # gauge: g_ref -> 1 (live, finite)
    return g, {"iterations": it, "converged": converged, "max_change": change}


def referenced_phases(gains, ref_ant: int, *, xp: ModuleType = np):
    """Gauge-referenced gain phases (amplitudes discarded).

    Args:
        gains: ``(n_ant,)`` complex gains.
        ref_ant: reference antenna; its phase is subtracted from all phases.
        xp: array module.

    Returns:
        ``(n_ant,)`` real phases ``angle(g_p) - angle(g_ref)`` in radians (``NaN`` for dead antennas).
    """
    gains = xp.asarray(gains)
    return xp.angle(gains) - xp.angle(gains[ref_ant])
