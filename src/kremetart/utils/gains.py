"""Apply the inverse of a per-antenna gain solution to visibilities.

The TART measurement equation for a baseline ``(p, q)`` is
``V_pq = g_p · conj(g_q) · V_pq^true`` with per-antenna complex gains ``g``. Correcting
(calibrating) the data divides that factor out; the matching inverse-variance weight
transform scales the weight by ``|g_p · g_q|**2``. Dead antennas carry ``g == 0`` and are
already weight-flagged; the correction guards the division and keeps them at zero weight.

Pure array math, ``xp``-injectable (``numpy`` on CPU, ``cupy`` on GPU) like the rest of the
imaging pipeline. No I/O.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def apply_inverse_gains(vis, weight, gains, a1_idx, a2_idx, *, xp: ModuleType = np):
    """Correct ``vis``/``weight`` by the inverse per-antenna gain product.

    Args:
        vis: ``(n_time, nbl, nchan)`` complex visibilities (baseline on the middle axis).
        weight: ``(n_time, nbl, nchan)`` real weights, same layout as ``vis``.
        gains: ``(n_ant,)`` complex per-antenna gains, ordered by antenna index.
        a1_idx: ``(nbl,)`` integer antenna index of the first antenna of each baseline.
        a2_idx: ``(nbl,)`` integer antenna index of the second antenna of each baseline.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(vis_corr, weight_corr)`` with the same shapes as the inputs. Baselines touching a
        dead antenna (gain ``0``) come out as ``0`` vis and ``0`` weight (never ``inf``/``nan``).
    """
    vis = xp.asarray(vis)
    weight = xp.asarray(weight)
    gains = xp.asarray(gains)

    factor = gains[a1_idx] * xp.conj(gains[a2_idx])  # (nbl,)
    ok = xp.abs(factor) > 0  # dead antennas -> factor == 0
    safe = xp.where(ok, factor, 1.0 + 0.0j)  # avoid divide-by-zero before masking

    factor_b = factor[None, :, None]
    ok_b = ok[None, :, None]
    safe_b = safe[None, :, None]

    vis_corr = xp.where(ok_b, vis / safe_b, 0.0)
    weight_corr = xp.where(ok_b, weight * xp.abs(factor_b) ** 2, 0.0)
    return vis_corr, weight_corr
