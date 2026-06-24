"""Tests for the reweighted-L1 FISTA solver (src/kremetart/opt/fista.py)."""

from __future__ import annotations

import numpy as np

from kremetart.opt.fista import _soft_threshold, fista


def test_soft_threshold_signed():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    out = _soft_threshold(z, 1.0, positive=False, xp=np)
    np.testing.assert_allclose(out, [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_soft_threshold_positive():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    out = _soft_threshold(z, 1.0, positive=True, xp=np)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 0.0, 2.0])


def test_soft_threshold_vector_tau():
    z = np.array([2.0, 2.0, 2.0])
    tau = np.array([0.5, 1.0, 3.0])
    out = _soft_threshold(z, tau, positive=False, xp=np)
    np.testing.assert_allclose(out, [1.5, 1.0, 0.0])


def _identity_ops():
    return (lambda x: x), (lambda r: r)


def test_identity_recovers_soft_threshold():
    # A = I, real data: argmin 0.5||x - y||² + λ||x||₁  ==  soft_threshold(y, λ)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    a, ah = _identity_ops()
    x, info = fista(a, ah, y, lam=0.3, positive=False, tol=1e-10, max_iter=2000)
    expect = np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0)
    np.testing.assert_allclose(x, expect, atol=1e-5)
    assert info["converged"]


def test_backtracking_recovers_from_tiny_l0():
    # A badly underestimated L0 must still converge (backtracking inflates lipschitz).
    rng = np.random.default_rng(1)
    y = rng.standard_normal(50)
    a, ah = _identity_ops()
    x, info = fista(a, ah, y, lam=0.3, positive=False, L0=1e-6, tol=1e-10, max_iter=2000)
    expect = np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0)
    np.testing.assert_allclose(x, expect, atol=1e-5)
    assert info["lipschitz"] > 1e-6  # grew toward the true Lipschitz constant (~1)


def test_positive_constraint():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(50)  # has negative entries
    a, ah = _identity_ops()
    x, _ = fista(a, ah, y, lam=0.1, positive=True, tol=1e-10, max_iter=2000)
    assert np.all(x >= 0.0)
    np.testing.assert_allclose(x, np.maximum(y - 0.1, 0.0), atol=1e-5)


def test_zero_data_returns_zeros():
    a, ah = _identity_ops()
    y = np.zeros(10)
    x, info = fista(a, ah, y, lam=0.5, positive=True)
    np.testing.assert_allclose(x, 0.0)
    assert info["converged"]
    assert info["reweights"] == 0
