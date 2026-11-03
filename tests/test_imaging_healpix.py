"""The image-space Hessian the Tikhonov and L1 solvers rely on (CPU, xp=numpy).

The dirty map itself is tested through the stages (``test_imaging_stages.py``); these pin the two
Hessian properties the solvers assume and the stage tests do not check (self-adjointness and the
closed-form diagonal used as the CG preconditioner and the FISTA Lipschitz seed), and that the forward
operator the tests simulate visibilities with is the adjoint of the imager's.
"""

import numpy as np

from kremetart.imaging.healpix import dft_adjoint, dft_forward, hessian_healpix, make_pixel_grid


def _dense_hessian(matvec, npix):
    """Materialise H as a dense (npix, npix) matrix by applying matvec to each unit vector."""
    cols = np.empty((npix, npix))
    for j in range(npix):
        e = np.zeros(npix)
        e[j] = 1.0
        cols[:, j] = matvec(e)
    return cols


def test_hessian_healpix_is_self_adjoint():
    """H = B Mᴴ W M B is symmetric over the reals, with and without a beam."""
    rng = np.random.default_rng(20)
    pix = make_pixel_grid(4, xp=np)
    npix = pix.shape[0]
    nrow, nchan = 12, 2
    baselines = rng.standard_normal((nrow, 3)) * 2.0
    freqs = np.array([1.40e9, 1.575e9])
    weights = rng.random((nrow, nchan)) + 0.1
    x = rng.standard_normal(npix)
    y = rng.standard_normal(npix)
    for beam in (None, rng.random((nchan, npix)) + 0.1):
        matvec, _ = hessian_healpix(baselines, pix, freqs, weights, beam=beam, xp=np)
        np.testing.assert_allclose(np.dot(matvec(x), y), np.dot(x, matvec(y)), rtol=1e-9, atol=1e-9)


def test_hessian_diagonal_matches_dense():
    """The closed-form diagonal equals the dense Hessian's diagonal (beam and no-beam)."""
    rng = np.random.default_rng(21)
    pix = make_pixel_grid(2, xp=np)  # npix = 48, small enough to densify
    npix = pix.shape[0]
    nrow, nchan = 9, 2
    baselines = rng.standard_normal((nrow, 3)) * 2.0
    freqs = np.array([1.40e9, 1.575e9])
    weights = rng.random((nrow, nchan)) + 0.1
    for beam in (None, rng.random((nchan, npix)) + 0.1):
        matvec, diagonal = hessian_healpix(baselines, pix, freqs, weights, beam=beam, xp=np)
        dense = _dense_hessian(matvec, npix)
        np.testing.assert_allclose(diagonal, np.diag(dense), rtol=1e-9, atol=1e-9)
        if beam is None:
            # diag(H)_j = Σ_c Σ_r W[r,c] for every pixel when there is no beam.
            np.testing.assert_allclose(diagonal, np.full(npix, weights.sum()), rtol=1e-9, atol=1e-9)


def test_hessian_diagonal_zero_below_beam_null():
    """Pixels where the beam is zero have zero Hessian diagonal."""
    rng = np.random.default_rng(22)
    pix = make_pixel_grid(4, xp=np)
    npix = pix.shape[0]
    baselines = rng.standard_normal((7, 3))
    freqs = np.array([1.575e9])
    weights = rng.random((7, 1)) + 0.1
    beam = rng.random((1, npix)) + 0.1
    beam[0, [1, 5, 17]] = 0.0
    _, diagonal = hessian_healpix(baselines, pix, freqs, weights, beam=beam, xp=np)
    np.testing.assert_allclose(diagonal[[1, 5, 17]], 0.0, atol=1e-12)


def test_forward_adjoint_hermitian_with_beam():
    """With a real per-channel beam, dft_adjoint is still the exact Hermitian transpose of dft_forward."""
    rng = np.random.default_rng(11)
    pix = make_pixel_grid(8, xp=np)
    npix = pix.shape[0]
    nrow, nchan = 10, 3
    baselines = rng.standard_normal((nrow, 3))
    freqs = np.array([1.40e9, 1.50e9, 1.575e9])
    beam = rng.random((nchan, npix)) + 0.1  # real, positive primary beam
    image = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    data = rng.standard_normal((nrow, nchan)) + 1j * rng.standard_normal((nrow, nchan))
    lhs = np.vdot(dft_forward(image, baselines, pix, freqs, beam=beam, xp=np), data)
    rhs = np.vdot(image, dft_adjoint(data, baselines, pix, freqs, beam=beam, xp=np))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
