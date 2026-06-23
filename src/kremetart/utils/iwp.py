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


def frame_has_observation(y, *, xp: ModuleType = np) -> bool:
    """Whether a frame carries a usable observation, or the filter should coast over it.

    A fully-flagged frame (all weights zero) is imaged as an all-zero map -- the no-data sentinel
    the imager and Tikhonov stage emit instead of a ``0/0`` NaN (see
    :func:`kremetart.utils.healpix_dft.dirty_map`) -- and any non-finite pixel likewise cannot
    inform the per-pixel filter. Either case means *no observation*: the IWP step predicts only.

    Args:
        y: ``(npix,)`` observation (dirty-map pixel values).
        xp: array module.

    Returns:
        ``True`` if ``y`` is finite and not identically zero, else ``False``.
    """
    return bool(xp.all(xp.isfinite(y))) and not bool(xp.all(y == 0.0))


def iwp_filter_step(X, P, *, dt, y, sigma2, R, has_obs: bool = True, xp: ModuleType = np):
    """One per-frame IWP-Kalman step: predict across ``dt``, then update unless the frame has no data.

    Wraps :func:`kalman_predict`/:func:`kalman_update` into the exact recursion the GPU operator
    runs, with one addition: a no-data frame (``has_obs=False``; see :func:`frame_has_observation`)
    advances the state by the predict step alone -- the principled treatment of a gap in the
    sidereally-fixed light curves (design note, sec:imaging) -- so a fully-flagged frame coasts on
    the prediction instead of poisoning the filter.

    Args:
        X: ``(npix, 2)`` prior means x_{k-1|k-1}.
        P: ``(npix, 2, 2)`` prior covariances P_{k-1|k-1}.
        dt: inter-frame interval (seconds); ``None`` on frame 0 (no predict, diffuse prior).
        y: ``(npix,)`` observation; ignored when ``has_obs`` is False.
        sigma2: IWP driving variance sigma^2.
        R: scalar measurement-noise variance.
        has_obs: when False the frame carries no usable data -> predict-only (coast), no update.
        xp: array module.

    Returns:
        ``(X, P, filtered, znorm)``: posterior means ``(npix, 2)`` and covariances ``(npix, 2, 2)``,
        the filtered flux ``X[:, 0]`` ``(npix,)``, and the normalised innovation ``(npix,)`` -- the
        latter all zeros on a no-data frame, where no innovation is defined.
    """
    x, p = X, P
    if dt is not None:
        a, q = iwp_transition(dt, sigma2, xp=xp)
        x, p = kalman_predict(x, p, a, q, xp=xp)
    if has_obs:
        x, p, e, s = kalman_update(x, p, y, R, xp=xp)
        znorm = e / xp.sqrt(s)
    else:
        znorm = xp.zeros(x.shape[0], dtype=x.dtype)
    return x, p, x[:, 0], znorm
