"""Tests for the reweighted-L1 FISTA solver (src/kremetart/opt/fista.py)."""

from __future__ import annotations

import numpy as np

from kremetart.opt.fista import _soft_threshold, fista, fista_quadratic
from kremetart.utils.skymodel import enu_direction_cosines, model_visibilities


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
    assert info["lipschitz"] > 0.5  # grew from 1e-6 toward the true Lipschitz constant (~1)


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


def test_complex_operator_recovers_real_solution():
    # A maps real x -> COMPLEX data; guards the Re{Aᴴ(...)} gradient handling.
    # Dropping the real-part wrapper makes the iterate complex and this recovery fails.
    rng = np.random.default_rng(7)
    n, m = 6, 24
    mat = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))

    def A(x):
        return mat @ x

    def AH(r):
        return mat.conj().T @ r

    x_true = np.array([0.0, 2.0, 0.0, 1.0, 0.5, 0.0])
    y = mat @ x_true
    x, _ = fista(A, AH, y, lam=1e-4, positive=True, tol=1e-12, max_iter=8000)
    np.testing.assert_allclose(x, x_true, atol=1e-2)
    assert not np.iscomplexobj(x)  # x stays real through the complex adjoint


def _gaussian_operator(rng, m, n):
    mat = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    return mat, (lambda x: mat @ x), (lambda r: mat.conj().T @ r)


def test_reweighting_debiases_sparse_recovery():
    rng = np.random.default_rng(3)
    n, m = 20, 60
    mat, a, ah = _gaussian_operator(rng, m, n)
    x_true = np.zeros(n)
    x_true[[2, 7, 15]] = [1.0, 2.0, 1.5]
    y = mat @ x_true

    x_l1, info_l1 = fista(a, ah, y, lam=0.2, positive=True, tol=1e-9, max_iter=4000)
    x_rw, info_rw = fista(
        a,
        ah,
        y,
        lam=0.2,
        positive=True,
        tol=1e-9,
        max_iter=4000,
        max_reweight=8,
        reweight_eps=1e-3,
    )

    err_l1 = np.linalg.norm(x_l1 - x_true)
    err_rw = np.linalg.norm(x_rw - x_true)
    assert err_rw <= err_l1 + 1e-9  # reweighting never worse than plain L1
    assert err_rw < 1e-2  # and essentially exact here
    assert info_l1["reweights"] == 0
    assert info_rw["reweights"] >= 1
    np.testing.assert_array_equal(np.where(x_rw > 1e-3)[0], [2, 7, 15])  # exact support


def test_exported_from_opt_package():
    from kremetart.opt import fista as fista_pkg

    assert fista_pkg is fista


def test_recover_source_fluxes_through_model_visibilities():
    # Per-source forward operator: column j is the unit-flux model visibility of source j.
    az = np.radians([10.0, 120.0, 250.0])
    el = np.radians([70.0, 40.0, 55.0])
    s = enu_direction_cosines(az, el)
    bl_enu = np.array(
        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 0.0], [5.0, 1.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.0, 0.0]]
    )
    freqs = np.array([1.575e9])
    cols = np.stack(
        [model_visibilities(s[j : j + 1], bl_enu, freqs).ravel() for j in range(s.shape[0])],
        axis=1,
    )  # (nbl*nchan, nsrc)

    def A(x):
        return cols @ x

    def AH(r):
        return cols.conj().T @ r

    flux_true = np.array([1.0, 0.5, 2.0])
    y = cols @ flux_true
    x, info = fista(A, AH, y, lam=1e-3, positive=True, tol=1e-10, max_iter=5000, max_reweight=4)
    np.testing.assert_allclose(x, flux_true, atol=1e-2)
    assert np.all(x >= 0.0)


def test_negative_max_reweight_clamps_to_plain_l1():
    # A negative reweight count must behave like max_reweight=0 (one plain-L1 solve), not a no-op.
    a, ah = _identity_ops()
    rng = np.random.default_rng(11)
    y = rng.standard_normal(20)
    x, info = fista(a, ah, y, lam=0.2, positive=False, max_reweight=-1)
    assert info["reweights"] == 0
    assert len(info["iterations"]) == 1


def test_fista_quadratic_matches_least_squares():
    # Quadratic form H = Aᴴ W A, b = Re{Aᴴ W y} must reach the same optimum as the A/AH path.
    rng = np.random.default_rng(21)
    n, m = 12, 40
    mat = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    weight = rng.uniform(0.5, 1.5, size=m)
    x_true = np.zeros(n)
    x_true[[1, 4, 9]] = [1.0, 2.0, 0.5]
    y = mat @ x_true

    def A(x):
        return mat @ x

    def AH(r):
        return mat.conj().T @ r

    def H(x):
        return AH(weight * A(x))  # complex; fista_quadratic takes the real part

    b = np.real(AH(weight * y))
    x_ls, _ = fista(A, AH, y, weight=weight, lam=0.2, positive=False, tol=1e-11, max_iter=8000)
    x_q, _ = fista_quadratic(H, b, lam=0.2, positive=False, tol=1e-11, max_iter=8000, max_reweight=0)
    np.testing.assert_allclose(x_q, x_ls, atol=1e-4)
    assert not np.iscomplexobj(x_q)


def test_fista_quadratic_identity_recovers_soft_threshold():
    # H = I, b = y, signed L1: argmin ½xᵀx − bᵀx + λ|x|₁ == soft_threshold(y, λ).
    rng = np.random.default_rng(22)
    y = rng.standard_normal(40)
    x, info = fista_quadratic(lambda x: x, y, lam=0.3, positive=False, tol=1e-11, max_iter=4000)
    np.testing.assert_allclose(x, np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0), atol=1e-5)
    assert info["converged"]


def test_fista_quadratic_backtracking_from_tiny_l0():
    rng = np.random.default_rng(23)
    y = rng.standard_normal(40)
    x, info = fista_quadratic(lambda x: x, y, lam=0.3, positive=False, L0=1e-6, tol=1e-11, max_iter=4000)
    np.testing.assert_allclose(x, np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0), atol=1e-5)
    assert info["lipschitz"] > 0.5  # grew from 1e-6 toward ‖H‖₂ = 1


def test_fista_quadratic_positive_constraint():
    rng = np.random.default_rng(24)
    y = rng.standard_normal(40)  # has negatives
    x, _ = fista_quadratic(lambda x: x, y, lam=0.1, positive=True, tol=1e-11, max_iter=4000)
    assert np.all(x >= 0.0)
    np.testing.assert_allclose(x, np.maximum(y - 0.1, 0.0), atol=1e-5)


def test_fista_quadratic_reweighting_debiases():
    rng = np.random.default_rng(25)
    n, m = 20, 60
    mat = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[[3, 8, 14]] = [1.0, 2.0, 1.5]
    y = mat @ x_true

    def H(x):
        return mat.conj().T @ (mat @ x)

    b = np.real(mat.conj().T @ y)
    x_l1, info_l1 = fista_quadratic(H, b, lam=0.2, positive=True, tol=1e-10, max_iter=6000, max_reweight=0)
    x_rw, info_rw = fista_quadratic(H, b, lam=0.2, positive=True, tol=1e-10, max_iter=6000, max_reweight=8)
    assert np.linalg.norm(x_rw - x_true) <= np.linalg.norm(x_l1 - x_true) + 1e-9
    assert np.linalg.norm(x_rw - x_true) < 1e-2
    assert info_l1["reweights"] == 0 and info_rw["reweights"] >= 1
    np.testing.assert_array_equal(np.where(x_rw > 1e-3)[0], [3, 8, 14])


def test_fista_quadratic_zero_rhs_short_circuit():
    x0 = np.array([0.4, 0.0, 1.1])
    x, info = fista_quadratic(lambda x: x, np.zeros(3), lam=0.5, positive=True, x0=x0)
    np.testing.assert_allclose(x, x0)  # b == 0 -> the warm start is unchanged
    assert info["converged"] and info["reweights"] == 0


def test_fista_quadratic_recovers_sparse_sky_through_hessian():
    # End-to-end on the real image-space Hessian: plant point sources, build H/b, recover with FISTA.
    from kremetart.utils.beam import airy_power_beam
    from kremetart.utils.healpix_dft import dft_adjoint, dft_forward, hessian_healpix, make_pixel_grid

    nside = 8
    pix = make_pixel_grid(nside, nest=True, xp=np)  # (npix, 3)
    npix = pix.shape[0]
    rng = np.random.default_rng(26)
    bl = rng.standard_normal((40, 3)) * 2.0  # metres
    freqs = np.array([1.575e9])
    boresight = np.array([0.0, 0.0, 1.0])
    beam = airy_power_beam(pix, boresight, freqs, xp=np)  # (1, npix)
    weights = np.ones((bl.shape[0], 1))

    sky = np.zeros(npix)
    up = np.where(pix @ boresight > 0.6)[0]  # well inside the beam
    src = up[rng.choice(up.size, size=3, replace=False)]
    sky[src] = np.array([1.0, 2.0, 1.5])

    vis = dft_forward(sky, bl, pix, freqs, beam=beam, xp=np)  # (nbl, nchan) complex
    hmv, hdiag = hessian_healpix(bl, pix, freqs, weights, beam=beam, xp=np)
    b = np.real(dft_adjoint(weights * vis, bl, pix, freqs, beam=beam, xp=np))  # un-normalised dirty = Re{B Mᴴ W y}

    x, info = fista_quadratic(
        hmv,
        b,
        lam=1e-3 * float(weights.sum()),
        positive=True,
        L0=float(hdiag.max()),
        tol=1e-9,
        max_iter=4000,
        max_reweight=3,
    )
    assert np.all(x >= 0.0)
    # The three planted pixels are the brightest recovered, in the right flux order.
    top3 = np.argsort(x)[-3:]
    assert set(top3.tolist()) == set(src.tolist())


def test_fista_quadratic_exported_from_opt_package():
    from kremetart.opt import fista_quadratic as fq_pkg

    assert fq_pkg is fista_quadratic
