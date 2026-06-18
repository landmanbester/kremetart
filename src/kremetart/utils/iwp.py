"""Integrated-Wiener-process (IWP) state-space model and the per-pixel Kalman recursion.

The quiescent prior of the kremetart design note (sec:iwp, sec:kf): each pixel's light curve is
a q=1 integrated Wiener process observed in noise, and the Kalman filter whitens it. All
functions are ``xp``-injectable -- pass ``xp=numpy`` (CPU tests) or ``xp=cupy`` (the GPU operator)
-- and vectorise over the leading pixel axis. See
docs/superpowers/specs/2026-06-17-smoovie-iwp-filter-design.md.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def iwp_transition(dt: float, sigma2: float, *, xp: ModuleType = np):
    """Exact q=1 IWP discrete transition matrices for an interval ``dt`` (eq. AQ).

    Args:
        dt: inter-frame interval Delta (seconds), read from the timestamp stream every frame
            (no constant-step assumption -- the design note's hard requirement).
        sigma2: scalar driving variance sigma^2.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(A, Q)``: the ``(2, 2)`` transition matrix ``A(dt)`` and process-noise covariance
        ``Q(dt)``.
    """
    dt = float(dt)
    a = xp.asarray([[1.0, dt], [0.0, 1.0]])
    q = sigma2 * xp.asarray([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    return a, q


def kalman_predict(X, P, A, Q, *, xp: ModuleType = np):
    """IWP predict step, vectorised over pixels.

    Args:
        X: ``(npix, 2)`` posterior means x_{k-1|k-1}.
        P: ``(npix, 2, 2)`` posterior covariances P_{k-1|k-1}.
        A: ``(2, 2)`` transition matrix.
        Q: ``(2, 2)`` process-noise covariance.
        xp: array module.

    Returns:
        ``(X_pred, P_pred)``: predicted means ``(npix, 2)`` and covariances ``(npix, 2, 2)``.
    """
    x_pred = X @ A.T
    p_pred = xp.einsum("ij,pjk,lk->pil", A, P, A) + Q
    return x_pred, p_pred


def kalman_update(X_pred, P_pred, y, R, *, xp: ModuleType = np):
    """IWP update step with scalar observation y = H x + v, H = (1, 0), Joseph form.

    Args:
        X_pred: ``(npix, 2)`` predicted means.
        P_pred: ``(npix, 2, 2)`` predicted covariances.
        y: ``(npix,)`` observations (dirty-map pixel values).
        R: scalar measurement-noise variance.
        xp: array module.

    Returns:
        ``(X_kk, P_kk, e, S)``: posterior means ``(npix, 2)``, posterior covariances
        ``(npix, 2, 2)``, innovations ``(npix,)`` and innovation variances ``(npix,)``.
    """
    npix = X_pred.shape[0]
    e = y - X_pred[:, 0]  # innovation (npix,)
    s = P_pred[:, 0, 0] + R  # innovation variance (npix,)
    k = P_pred[:, :, 0] / s[:, None]  # gain (npix, 2): P_pred @ H^T is column 0
    x_kk = X_pred + k * e[:, None]
    # Joseph form: (I - K H) P_pred (I - K H)^T + K R K^T, with H = (1, 0).
    eye = xp.broadcast_to(xp.eye(2), (npix, 2, 2))
    kh = k[:, :, None] * xp.asarray([1.0, 0.0])[None, None, :]  # (npix, 2, 2)
    i_kh = eye - kh
    p_kk = xp.einsum("pij,pjk,plk->pil", i_kh, P_pred, i_kh) + R * (k[:, :, None] * k[:, None, :])
    return x_kk, p_kk, e, s
