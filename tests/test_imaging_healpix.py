"""Unit tests for the HEALPix gridless DFT operator (CPU, xp=numpy)."""

import numpy as np
import pytest

from kremetart.imaging.healpix import (
    dft_adjoint,
    dft_forward,
    dirty_map,
    hessian_healpix,
    image_frame,
    image_frame_prerotated,
    make_pixel_grid,
)
from kremetart.utils.geometry import equatorial_baselines

LIGHTSPEED = 299792458.0


def test_make_pixel_grid_shape_and_unit_norm():
    nside = 8
    pix = make_pixel_grid(nside, xp=np)
    assert pix.shape == (12 * nside**2, 3)
    np.testing.assert_allclose(np.linalg.norm(pix, axis=1), 1.0, atol=1e-12)


def test_make_pixel_grid_nested_matches_healpy() -> None:
    hp = pytest.importorskip("healpy")  # the reference implementation; no Windows wheels

    nside = 4
    pix = make_pixel_grid(nside, xp=np)  # nest=True default
    expected = np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), axis=1)
    np.testing.assert_allclose(pix, expected, atol=1e-12)


def test_forward_matches_explicit_fringe():
    """forward of a single-pixel source is the geometric fringe on every row/channel."""
    rng = np.random.default_rng(1)
    pix = make_pixel_grid(4, xp=np)
    npix = pix.shape[0]
    baselines = rng.standard_normal((5, 3))
    freqs = np.array([1.40e9, 1.575e9])
    image = np.zeros(npix)
    image[3] = 2.0
    vis = dft_forward(image, baselines, pix, freqs, xp=np)
    assert vis.shape == (5, 2)
    g = baselines @ pix[3]  # (nrow,)
    expected = 2.0 * np.exp(2j * np.pi * (freqs[None, :] / LIGHTSPEED) * g[:, None])
    np.testing.assert_allclose(vis, expected, rtol=1e-12, atol=1e-12)


def test_forward_adjoint_are_hermitian_transposes():
    """<forward(image), data> == <image, adjoint(data)>  (the adjointness dot-product test)."""
    rng = np.random.default_rng(0)
    pix = make_pixel_grid(8, xp=np)
    npix = pix.shape[0]
    nrow, nchan = 12, 3
    baselines = rng.standard_normal((nrow, 3))
    freqs = np.array([1.40e9, 1.50e9, 1.575e9])
    image = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    data = rng.standard_normal((nrow, nchan)) + 1j * rng.standard_normal((nrow, nchan))
    lhs = np.vdot(dft_forward(image, baselines, pix, freqs, xp=np), data)
    rhs = np.vdot(image, dft_adjoint(data, baselines, pix, freqs, xp=np))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_dirty_map_recovers_point_source():
    """dirty_map of a forward-modelled single-pixel source peaks (value 1) at that pixel."""
    rng = np.random.default_rng(2)
    pix = make_pixel_grid(16, xp=np)
    npix = pix.shape[0]
    nrow = 300
    baselines = rng.standard_normal((nrow, 3)) * 2.0
    freqs = np.array([1.575e9])
    src = 1234
    image = np.zeros(npix)
    image[src] = 1.0
    vis = dft_forward(image, baselines, pix, freqs, xp=np)
    weights = np.ones((nrow, 1))
    dmap = dirty_map(vis, weights, baselines, pix, freqs, xp=np)
    assert dmap.shape == (npix,)
    assert dmap.dtype == np.float64
    assert int(np.argmax(dmap)) == src
    np.testing.assert_allclose(dmap[src], 1.0, atol=1e-12)


def test_image_frame_recovers_source_through_ctime():
    """End-to-end: model a source per timestamp with C(t), image it back to the right pixel."""
    rng = np.random.default_rng(5)
    nside = 16
    pix = make_pixel_grid(nside, xp=np)
    itrs_bl = rng.standard_normal((20, 3)) * 3.0
    times = np.array([1.6e9, 1.6e9 + 60.0, 1.6e9 + 120.0])
    freqs = np.array([1.575e9])
    src = 1500  # valid pixel index for nside=16 (npix=3072)

    b_rot = equatorial_baselines(itrs_bl, times, xp=np)  # (n_time, nbl, 3)
    n_time, nbl = b_rot.shape[:2]
    s = pix[src]
    vis = np.empty((n_time, nbl, 1), dtype=complex)
    for t in range(n_time):
        vis[t, :, 0] = np.exp(2j * np.pi * (freqs[0] / LIGHTSPEED) * (b_rot[t] @ s))
    wgt = np.ones((n_time, nbl, 1))

    dmap = image_frame(vis, wgt, times, itrs_bl, pix, freqs, xp=np)
    assert dmap.shape == (pix.shape[0],)
    assert int(np.argmax(dmap)) == src
    np.testing.assert_allclose(dmap[src], 1.0, atol=1e-12)


def test_image_frame_prerotated_matches_image_frame():
    """The device-pure core equals the full image_frame (which now wraps it)."""
    from kremetart.imaging.healpix import image_frame
    from kremetart.utils.geometry import equatorial_baselines

    rng = np.random.default_rng(7)
    nside = 8
    pix = make_pixel_grid(nside, xp=np)
    itrs_bl = rng.standard_normal((15, 3)) * 3.0
    times = np.array([1.6e9, 1.6e9 + 60.0])
    freqs = np.array([1.575e9])
    nbl = itrs_bl.shape[0]
    vis = rng.standard_normal((2, nbl, 1)) + 1j * rng.standard_normal((2, nbl, 1))
    wgt = np.ones((2, nbl, 1))

    ref = image_frame(vis, wgt, times, itrs_bl, pix, freqs, xp=np)
    b_rot = equatorial_baselines(itrs_bl, times, xp=np)
    got = image_frame_prerotated(vis, wgt, b_rot, pix, freqs, xp=np)

    assert got.shape == (pix.shape[0],)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


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


def test_dirty_map_with_beam_scales_and_masks():
    """Beam measurement operator: a point source images to beam**2 at its pixel; beam==0 -> 0."""
    rng = np.random.default_rng(12)
    pix = make_pixel_grid(16, xp=np)
    npix = pix.shape[0]
    nrow = 300
    baselines = rng.standard_normal((nrow, 3)) * 2.0
    freqs = np.array([1.575e9])
    beam = rng.random((1, npix)) + 0.2  # positive
    zeroed = np.array([10, 2000, 3000])  # stand-in for below-horizon pixels
    beam[0, zeroed] = 0.0
    src = 1234
    image = np.zeros(npix)
    image[src] = 1.0
    vis = dft_forward(image, baselines, pix, freqs, beam=beam, xp=np)
    weights = np.ones((nrow, 1))
    dmap = dirty_map(vis, weights, baselines, pix, freqs, beam=beam, xp=np)
    assert int(np.argmax(dmap)) == src
    np.testing.assert_allclose(dmap[src], beam[0, src] ** 2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dmap[zeroed], 0.0, atol=1e-12)


def test_dirty_map_beam_none_matches_no_beam():
    """beam=None reproduces the no-beam dirty map exactly (regression)."""
    rng = np.random.default_rng(13)
    pix = make_pixel_grid(8, xp=np)
    nrow = 50
    baselines = rng.standard_normal((nrow, 3))
    freqs = np.array([1.50e9, 1.575e9])
    vis = rng.standard_normal((nrow, 2)) + 1j * rng.standard_normal((nrow, 2))
    weights = rng.random((nrow, 2))
    a = dirty_map(vis, weights, baselines, pix, freqs, xp=np)
    b = dirty_map(vis, weights, baselines, pix, freqs, beam=None, xp=np)
    np.testing.assert_array_equal(a, b)


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


def test_cg_solves_tikhonov_system_against_dense():
    """(H + λI) x = b solved by preconditioned CG matches a dense direct solve."""
    from kremetart.opt.cg import cg

    rng = np.random.default_rng(23)
    pix = make_pixel_grid(2, xp=np)
    npix = pix.shape[0]
    nrow, nchan = 11, 1
    baselines = rng.standard_normal((nrow, nchan + 2)) * 2.0  # (nrow, 3)
    freqs = np.array([1.575e9])
    weights = rng.random((nrow, nchan)) + 0.1
    beam = rng.random((nchan, npix)) + 0.2
    matvec, diagonal = hessian_healpix(baselines, pix, freqs, weights, beam=beam, xp=np)

    wsum = weights.sum()
    lam = 0.1 * wsum  # eta = 0.1
    b = rng.standard_normal(npix)

    dense = _dense_hessian(matvec, npix) + lam * np.eye(npix)
    x_dense = np.linalg.solve(dense, b)
    x_cg = cg(
        lambda v: matvec(v) + lam * v,
        b,
        M=lambda r: r / (diagonal + lam),
        maxiter=500,
        tol=1e-12,
        xp=np,
    )
    np.testing.assert_allclose(x_cg, x_dense, rtol=1e-6, atol=1e-7)


def test_dirty_map_all_zero_weights_returns_finite_zeros():
    """A fully-flagged frame (all weights 0) must not divide 0/0 -> NaN; it emits a clean zero map
    so the downstream IWP filter sees a no-data frame, not a poisoned (NaN) observation."""
    rng = np.random.default_rng(7)
    pix = make_pixel_grid(2, xp=np)
    nrow, nchan = 6, 1
    baselines = rng.standard_normal((nrow, 3))
    freqs = np.array([1.575e9])
    vis = rng.standard_normal((nrow, nchan)) + 1j * rng.standard_normal((nrow, nchan))
    weights = np.zeros((nrow, nchan))
    dmap = dirty_map(vis, weights, baselines, pix, freqs, xp=np)
    assert np.all(np.isfinite(dmap))
    np.testing.assert_array_equal(dmap, np.zeros(pix.shape[0]))
