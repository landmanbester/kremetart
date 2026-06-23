"""Unit tests for the generic conjugate-gradient solver (CPU, xp=numpy)."""

import numpy as np

from kremetart.opt.cg import cg


def _spd(n, rng, cond=None):
    """A random symmetric positive-definite matrix; optionally with a target condition number."""
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    if cond is None:
        d = rng.uniform(1.0, 3.0, size=n)
    else:
        d = np.geomspace(1.0, cond, n)  # eigenvalues spanning [1, cond]
    return (q * d) @ q.T


def test_cg_solves_spd_system():
    """Unpreconditioned, zero-initialised CG matches a direct solve (the reference path)."""
    rng = np.random.default_rng(0)
    n = 40
    a = _spd(n, rng)
    b = rng.standard_normal(n)
    x = cg(lambda v: a @ v, b, maxiter=200, tol=1e-12, xp=np)
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-6, atol=1e-8)


def test_cg_zero_rhs_returns_zero():
    rng = np.random.default_rng(1)
    a = _spd(10, rng)
    x = cg(lambda v: a @ v, np.zeros(10), xp=np)
    np.testing.assert_array_equal(x, np.zeros(10))


def test_jacobi_preconditioner_reduces_residual_for_fixed_budget():
    """On an ill-conditioned SPD system, Jacobi PCG beats plain CG at a fixed small iteration budget."""
    rng = np.random.default_rng(2)
    n = 60
    a = _spd(n, rng, cond=1e4)
    b = rng.standard_normal(n)
    inv_diag = 1.0 / np.diag(a)

    budget = 15
    x_plain = cg(lambda v: a @ v, b, maxiter=budget, tol=0.0, xp=np)
    x_pre = cg(lambda v: a @ v, b, M=lambda r: r * inv_diag, maxiter=budget, tol=0.0, xp=np)
    assert np.linalg.norm(a @ x_pre - b) < np.linalg.norm(a @ x_plain - b)

    # Both still converge to the same solution given enough iterations.
    x_full = cg(lambda v: a @ v, b, M=lambda r: r * inv_diag, maxiter=500, tol=1e-12, xp=np)
    np.testing.assert_allclose(x_full, np.linalg.solve(a, b), rtol=1e-6, atol=1e-8)


def test_warm_start_reduces_residual_for_fixed_budget():
    """Seeding x0 near the solution lands closer than a zero start at a fixed budget."""
    rng = np.random.default_rng(3)
    n = 50
    a = _spd(n, rng, cond=1e3)
    b = rng.standard_normal(n)
    truth = np.linalg.solve(a, b)
    x0 = truth + 1e-3 * rng.standard_normal(n)  # a good warm start

    budget = 5
    x_cold = cg(lambda v: a @ v, b, maxiter=budget, tol=0.0, xp=np)
    x_warm = cg(lambda v: a @ v, b, x0=x0, maxiter=budget, tol=0.0, xp=np)
    assert np.linalg.norm(a @ x_warm - b) < np.linalg.norm(a @ x_cold - b)


def test_warm_start_at_solution_is_fixed_point():
    rng = np.random.default_rng(4)
    a = _spd(20, rng)
    b = rng.standard_normal(20)
    truth = np.linalg.solve(a, b)
    x = cg(lambda v: a @ v, b, x0=truth, maxiter=50, tol=1e-12, xp=np)
    np.testing.assert_allclose(x, truth, rtol=1e-8, atol=1e-10)
