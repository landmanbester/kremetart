"""Reweighted-L1 FISTA solver (xp-injectable; CPU via numpy, GPU via cupy).

Generic and operator-agnostic, like :mod:`kremetart.opt.cg`: the forward operator ``A`` and its
Hermitian adjoint ``AH`` are plain callables ``x -> A @ x`` and ``r -> Aᴴ @ r``. Minimises, over a
**real** vector ``x``, ``0.5 ||W^½ (A x - y)||² + λ Σ wᵢ |xᵢ|`` with FISTA + backtracking and an
outer Candès–Wakin–Boyd reweighting loop. Used at the outset of self-calibration to recover
non-negative source fluxes from calibrated visibilities; the caller wires ``A`` to the imaging DFT
or the per-source sky model (out of scope here, mirroring how the imager wires up ``cg``).
"""

from __future__ import annotations

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
