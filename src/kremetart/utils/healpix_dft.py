"""Gridless forward/adjoint DFT on a full-sky equatorial HEALPix grid.

The imaging step of the TART streaming pipeline. Pixels are represented by their Cartesian
unit vectors (direction cosines); the dirty-map kernel is the bare geometric delay
``exp(-2pi i (nu/c) b . s)`` with no ``(n-1)`` reference term and no ``1/n`` Jacobian (the
HEALPix grid is equal-area). See docs/superpowers/specs/2026-06-15-healpix-dft-operator-design.md.

The math is ``xp``-injectable: pass ``xp=numpy`` (CPU tests) or ``xp=cupy`` (GPU pipeline).
Only the per-frame frame-rotation ``C(t)`` is computed on the host with astropy (O(n_time)),
exactly the host/device split used in :mod:`kremetart.utils.rephasing`.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

LIGHTSPEED = 299792458.0


def make_pixel_grid(nside: int, *, nest: bool = True, xp: ModuleType = np):
    """Return the HEALPix pixel unit vectors (direction cosines).

    Args:
        nside: HEALPix resolution; ``npix = 12 * nside**2``.
        nest: Use NESTED ordering (default; index locality for the streaming detector).
        xp: Array module for the returned array (``numpy`` or ``cupy``).

    Returns:
        ``(npix, 3)`` array of unit vectors, declared to live in the equatorial (ICRS) frame.
    """
    import healpy as hp

    npix = hp.nside2npix(nside)
    vec = hp.pix2vec(nside, np.arange(npix), nest=nest)  # tuple of three (npix,) arrays
    grid = np.stack(vec, axis=1).astype(np.float64)  # (npix, 3)
    return xp.asarray(grid)


def _delay(baselines, pix_vec):
    """Geometric delay matrix b . s in metres, shape (nrow, npix)."""
    return baselines @ pix_vec.T


def _phase(baselines, pix_vec, freqs, xp):
    """2*pi*(nu/c)*(b . s), shape (nrow, nchan, npix)."""
    g = _delay(baselines, pix_vec)  # (nrow, npix)
    inv_wl = xp.asarray(freqs) / LIGHTSPEED  # (nchan,) cycles per metre
    return 2.0 * xp.pi * inv_wl[None, :, None] * g[:, None, :]


def dft_forward(image, baselines, pix_vec, freqs, *, xp: ModuleType = np):
    """Image -> visibilities (phasesign +1).

    Args:
        image: ``(npix,)`` sky (real in production; complex accepted for the adjoint test).
        baselines: ``(nrow, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        xp: Array module.

    Returns:
        ``(nrow, nchan)`` complex visibilities.
    """
    kernel = xp.exp(1j * _phase(baselines, pix_vec, freqs, xp))  # (nrow, nchan, npix)
    return kernel @ xp.asarray(image)  # (nrow, nchan)


def dft_adjoint(vis, baselines, pix_vec, freqs, *, xp: ModuleType = np):
    """Visibilities -> image (phasesign -1); the exact Hermitian transpose of :func:`dft_forward`.

    Args:
        vis: ``(nrow, nchan)`` complex visibilities.
        baselines, pix_vec, freqs, xp: as in :func:`dft_forward`.

    Returns:
        ``(npix,)`` complex image (caller takes ``Re`` / normalises; see :func:`dirty_map`).
    """
    kernel = xp.exp(-1j * _phase(baselines, pix_vec, freqs, xp))  # conj of forward
    return xp.einsum("rcj,rc->j", kernel, xp.asarray(vis))  # (npix,)


def dirty_map(vis, weights, baselines, pix_vec, freqs, *, xp: ModuleType = np):
    """Weighted adjoint dirty map: ``Re{ adjoint(weights * vis) } / sum(weights)``.

    Implements the design-doc dirty-map equation directly (equal-area grid: no 1/n factor).

    Args:
        vis: ``(nrow, nchan)`` complex residual visibilities.
        weights: ``(nrow, nchan)`` gain-corrected weights ``w_corr``.
        baselines, pix_vec, freqs, xp: as in :func:`dft_forward`.

    Returns:
        ``(npix,)`` real dirty image.
    """
    vis = xp.asarray(vis)
    weights = xp.asarray(weights)
    img = dft_adjoint(weights * vis, baselines, pix_vec, freqs, xp=xp)
    return img.real / weights.sum()


def _icrs_to_itrs_matrices(times: np.ndarray) -> np.ndarray:
    """Per-timestamp ICRS->ITRS rotation matrices R(t), shape (n_time, 3, 3) (host, O(n_time)).

    Column ``i`` of ``R(t)`` is the ITRS image of the ``i``-th ICRS axis, so
    ``s_itrs(t) = R(t) @ s_icrs``. The axes are transformed as unit-sphere directions (no distance)
    via the same astropy path as :func:`kremetart.utils.rephasing._itrs_unit_vectors`, folding in
    frame bias, precession, nutation and Earth rotation.

    This is a *pure rotation* (the design's ``C(t)`` = latitude tilt then rotation about the polar
    axis by LST). It therefore reproduces the full ICRS->ITRS source transform only up to stellar
    aberration -- the non-rotational ICRS<->GCRS term (~20 arcsec) cannot be captured by any single
    matrix -- which is negligible against the ~0.9 deg HEALPix pixel.
    """
    import astropy.units as u
    from astropy.coordinates import ICRS, ITRS, UnitSphericalRepresentation
    from astropy.time import Time

    tt = Time(np.asarray(times), format="unix", scale="utc")
    # The three ICRS axes (+x, +y, +z) as directions at infinity.
    axes = ICRS(UnitSphericalRepresentation(lon=[0.0, 90.0, 0.0] * u.deg, lat=[0.0, 0.0, 90.0] * u.deg))
    mats = np.empty((tt.size, 3, 3), dtype=np.float64)
    for k in range(tt.size):
        itrs = axes.transform_to(ITRS(obstime=tt[k]))
        mats[k] = itrs.cartesian.xyz.value  # (component, axis) -> columns of R(t)
    return mats


def equatorial_baselines(itrs_baselines, times, *, backend: str = "astropy", xp: ModuleType = np):
    """Rotate fixed ITRS baselines into the equatorial frame for each timestamp.

    Args:
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors (e.g. from rephasing's ``itrs_baselines``).
        times: ``(n_time,)`` unix-second timestamps.
        backend: ``"astropy"`` (the oracle, host-side) or ``"native"`` (GPU polynomial; later phase).
        xp: Array module for the returned array.

    Returns:
        ``(n_time, nbl, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
    """
    if backend == "astropy":
        b = np.asarray(itrs_baselines)
        rot = _icrs_to_itrs_matrices(times)  # (n_time, 3, 3)
        b_rot = np.einsum("bi,tik->tbk", b, rot)  # b_itrs @ R(t)
        return xp.asarray(b_rot)
    if backend == "native":
        raise NotImplementedError("GPU-native C(t) backend is a later phase; use backend='astropy'.")
    raise ValueError(f"unknown backend {backend!r}")


def image_frame(
    vis, weights, times, itrs_baselines, pix_vec, freqs, *, ctime_backend: str = "astropy", xp: ModuleType = np
):
    """Per-frame dirty image from unstopped residual visibilities.

    Rotates the ITRS baselines by ``C(t)``, flattens ``(time, baseline)`` into the row axis
    (the rotated baseline differs per timestamp), and adjoint-DFTs onto the fixed grid.

    Args:
        vis: ``(n_time, nbl, nchan)`` complex residual visibilities (scalar pol).
        weights: ``(n_time, nbl, nchan)`` gain-corrected weights.
        times: ``(n_time,)`` unix-second timestamps.
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        ctime_backend: passed to :func:`equatorial_baselines`.
        xp: Array module.

    Returns:
        ``(npix,)`` real dirty image.
    """
    b_rot = equatorial_baselines(itrs_baselines, times, backend=ctime_backend, xp=xp)  # (n_time, nbl, 3)
    n_time, nbl, nchan = vis.shape
    rows = b_rot.reshape(n_time * nbl, 3)
    vis_rows = xp.asarray(vis).reshape(n_time * nbl, nchan)
    wgt_rows = xp.asarray(weights).reshape(n_time * nbl, nchan)
    return dirty_map(vis_rows, wgt_rows, rows, pix_vec, freqs, xp=xp)
