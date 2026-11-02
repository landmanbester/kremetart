"""Gridless forward/adjoint DFT on a full-sky equatorial HEALPix grid.

The imaging step of the TART streaming pipeline. Pixels are represented by their Cartesian
unit vectors (direction cosines); the dirty-map kernel is the bare geometric delay
``exp(-2pi i (nu/c) b . s)`` with no ``(n-1)`` reference term and no ``1/n`` Jacobian (the
HEALPix grid is equal-area). See docs/superpowers/specs/2026-06-15-healpix-dft-operator-design.md.

The math is ``xp``-injectable: pass ``xp=numpy`` (CPU tests) or ``xp=cupy`` (GPU pipeline).
The host-side frame rotation ``C(t)`` lives in :mod:`kremetart.utils.geometry`.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from kremetart.utils.geometry import equatorial_baselines

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


def dft_forward(image, baselines, pix_vec, freqs, *, beam=None, xp: ModuleType = np):
    """Image -> visibilities (phasesign +1).

    With ``beam`` given this is the beam measurement operator ``R = A_dft . diag(B)``: the sky is
    multiplied by the per-channel primary beam before the geometric DFT.

    Args:
        image: ``(npix,)`` sky (real in production; complex accepted for the adjoint test).
        baselines: ``(nrow, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        beam: optional ``(nchan, npix)`` real primary beam (e.g.
            :func:`kremetart.utils.beam.airy_power_beam`). ``None`` -> no beam (identity).
        xp: Array module.

    Returns:
        ``(nrow, nchan)`` complex visibilities.
    """
    kernel = xp.exp(1j * _phase(baselines, pix_vec, freqs, xp))  # (nrow, nchan, npix)
    if beam is not None:
        kernel = kernel * xp.asarray(beam)[None, :, :]  # attenuate the sky by the per-channel beam
    return kernel @ xp.asarray(image)  # (nrow, nchan)


def dft_adjoint(vis, baselines, pix_vec, freqs, *, beam=None, xp: ModuleType = np):
    """Visibilities -> image (phasesign -1); the exact Hermitian transpose of :func:`dft_forward`.

    Args:
        vis: ``(nrow, nchan)`` complex visibilities.
        beam, baselines, pix_vec, freqs, xp: as in :func:`dft_forward`. The same real ``beam`` is
            applied per channel (before the channel sum), preserving the Hermitian transpose.

    Returns:
        ``(npix,)`` complex image (caller takes ``Re`` / normalises; see :func:`dirty_map`).
    """
    kernel = xp.exp(-1j * _phase(baselines, pix_vec, freqs, xp))  # conj of forward
    vis = xp.asarray(vis)
    if beam is None:
        return xp.einsum("rcj,rc->j", kernel, vis)  # (npix,)
    return xp.einsum("rcj,rc,cj->j", kernel, vis, xp.asarray(beam))  # beam per channel before the channel sum


def dirty_map(vis, weights, baselines, pix_vec, freqs, *, beam=None, xp: ModuleType = np):
    """Weighted adjoint dirty map: ``Re{ adjoint(weights * vis) } / sum(weights)``.

    Implements the design-doc dirty-map equation directly (equal-area grid: no 1/n factor). With
    ``beam`` given the per-frame map is the beam-weighted ``B (.) (A_dft^H W vis)`` oriented toward
    the intrinsic sky (see ``docs/superpowers/specs/2026-06-23-beam-measurement-operator-design.md``).

    Args:
        vis: ``(nrow, nchan)`` complex residual visibilities.
        weights: ``(nrow, nchan)`` gain-corrected weights ``w_corr``.
        beam, baselines, pix_vec, freqs, xp: as in :func:`dft_forward`.

    Returns:
        ``(npix,)`` real dirty image.
    """
    vis = xp.asarray(vis)
    weights = xp.asarray(weights)
    img = dft_adjoint(weights * vis, baselines, pix_vec, freqs, beam=beam, xp=xp)
    wsum = weights.sum()
    # A fully-flagged frame (all weights 0) would divide 0/0 -> NaN; emit a clean all-zero map
    # instead so the downstream IWP filter reads it as a no-data frame (it coasts on its prediction)
    # rather than ingesting a NaN observation that would poison the Kalman state for the whole run.
    if float(wsum) == 0.0:
        return xp.zeros(pix_vec.shape[0], dtype=img.real.dtype)
    return img.real / wsum


def hessian_healpix(baselines, pix_vec, freqs, weights, *, beam=None, xp: ModuleType = np):
    """Per-frame image-space Hessian ``H = B Mᴴ W M B`` and its diagonal.

    ``H`` is the (un-normalised) normal operator of the beam measurement operator ``R = M diag(B)``:
    ``H x = Σ_c B[c] ⊙ Re{ M_cᴴ W_c M_c (B[c] ⊙ x) }``, summing channels into one MFS image. It is
    symmetric positive semi-definite over the reals, so ``H + λI`` is SPD and solvable by CG
    (:func:`kremetart.opt.cg.cg`). The geometric DFT kernel is built **once** here and reused by the
    returned ``matvec`` (forward then weighted adjoint), so each CG iteration is two contractions
    with no kernel rebuild.

    Args:
        baselines: ``(nrow, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres (one frame).
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        weights: ``(nrow, nchan)`` gain-corrected weights.
        beam: optional ``(nchan, npix)`` real primary beam; ``None`` -> identity beam.
        xp: Array module.

    Returns:
        ``(matvec, diagonal)`` where ``matvec`` is a callable ``x:(npix,) -> H x`` (real, ``(npix,)``)
        and ``diagonal`` is the closed-form ``diag(H)_j = Σ_c B[c,j]² · Σ_r W[r,c]`` (``(npix,)``;
        exact because the kernel has unit modulus), used to build the Jacobi preconditioner.
    """
    baselines = xp.asarray(baselines)
    weights = xp.asarray(weights)
    kernel = xp.exp(1j * _phase(baselines, pix_vec, freqs, xp))  # (nrow, nchan, npix), built once
    beam_a = None if beam is None else xp.asarray(beam)  # (nchan, npix)
    w_sum_c = weights.sum(axis=0)  # (nchan,)
    npix = pix_vec.shape[0]

    if beam_a is None:
        diagonal = xp.full(npix, w_sum_c.sum(), dtype=xp.float64)
    else:
        diagonal = (beam_a**2 * w_sum_c[:, None]).sum(axis=0)  # (npix,)

    def matvec(x):
        x = xp.asarray(x)
        bx = x[None, :] if beam_a is None else beam_a * x[None, :]  # (nchan, npix)
        vis = xp.einsum("rcj,cj->rc", kernel, bx)  # forward M (nrow, nchan)
        wv = weights * vis
        # Mᴴ via conj(kernel): conj(kernel) @ wv == conj(kernel @ conj(wv)); avoids a second kernel.
        adj = xp.conj(xp.einsum("rcj,rc->cj", kernel, xp.conj(wv)))  # (nchan, npix)
        if beam_a is not None:
            adj = beam_a * adj
        return adj.sum(axis=0).real  # (npix,)

    return matvec, diagonal


def image_frame_prerotated(vis, weights, b_rot, pix_vec, freqs, *, beam=None, xp: ModuleType = np):
    """Per-frame dirty image from already-rotated baselines (device-pure; no host astropy).

    The frame rotation ``C(t)`` has already been folded into ``b_rot``; this is the pure-``xp`` core of
    :func:`image_frame`, imaging several timestamps into one map. It flattens ``(time, baseline)`` into
    the row axis and adjoint-DFTs onto the fixed grid. (The streaming pipeline images one snapshot at
    a time with :class:`kremetart.imaging.dirty_imager.DirtyImager`.)

    Args:
        vis: ``(n_time, nbl, nchan)`` complex residual visibilities (scalar pol).
        weights: ``(n_time, nbl, nchan)`` gain-corrected weights.
        b_rot: ``(n_time, nbl, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        beam: optional ``(nchan, npix)`` real primary beam; see :func:`dft_forward`.
        xp: Array module.

    Returns:
        ``(npix,)`` real dirty image.
    """
    b_rot = xp.asarray(b_rot)
    vis = xp.asarray(vis)
    weights = xp.asarray(weights)
    n_time, nbl, _ = b_rot.shape
    nchan = vis.shape[-1]
    rows = b_rot.reshape(n_time * nbl, 3)
    vis_rows = vis.reshape(n_time * nbl, nchan)
    wgt_rows = weights.reshape(n_time * nbl, nchan)
    return dirty_map(vis_rows, wgt_rows, rows, pix_vec, freqs, beam=beam, xp=xp)


def image_frame(
    vis,
    weights,
    times,
    itrs_baselines,
    pix_vec,
    freqs,
    *,
    beam=None,
    ctime_backend: str = "astropy",
    xp: ModuleType = np,
):
    """Per-frame dirty image from unstopped residual visibilities.

    Rotates the ITRS baselines by ``C(t)`` on the host (:func:`equatorial_baselines`) and delegates
    the DFT to the device-pure :func:`image_frame_prerotated`. Signature and result are unchanged
    from the original single-function implementation, so existing callers/tests are unaffected.

    Args:
        vis: ``(n_time, nbl, nchan)`` complex residual visibilities (scalar pol).
        weights: ``(n_time, nbl, nchan)`` gain-corrected weights.
        times: ``(n_time,)`` unix-second timestamps.
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        beam: optional ``(nchan, npix)`` real primary beam; see :func:`dft_forward`.
        ctime_backend: passed to :func:`equatorial_baselines`.
        xp: Array module.

    Returns:
        ``(npix,)`` real dirty image.
    """
    b_rot = equatorial_baselines(itrs_baselines, times, backend=ctime_backend, xp=xp)  # (n_time, nbl, 3)
    return image_frame_prerotated(vis, weights, b_rot, pix_vec, freqs, beam=beam, xp=xp)
