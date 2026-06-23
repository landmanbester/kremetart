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
    return img.real / weights.sum()


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


def zenith_icrs_vectors(times, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Instantaneous-zenith ICRS unit vectors at each timestamp (host, astropy; ``O(n_time)``).

    These are the antenna-boresight directions for the Airy primary beam: the site local zenith
    (``AltAz`` alt=90 deg) expressed as a unit vector in the same equatorial ICRS frame as
    :func:`make_pixel_grid`, so that ``pix_vec @ boresight`` is the cosine of each pixel's zenith
    angle. Precomputed per frame by the host prepare-step and applied on the GPU by the imager,
    mirroring the ``b_rot`` host/device split.

    Args:
        times: ``(n_time,)`` unix-second timestamps.
        lat_deg: site geodetic latitude in degrees.
        lon_deg: site longitude in degrees.
        alt_m: site altitude in metres.

    Returns:
        ``(n_time, 3)`` array of ICRS unit vectors of the site zenith.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    tt = Time(np.asarray(times), format="unix", scale="utc")
    zen = SkyCoord(AltAz(az=0.0 * u.deg, alt=90.0 * u.deg, obstime=tt, location=loc)).icrs
    ra, dec = zen.ra.rad, zen.dec.rad
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


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


def image_frame_prerotated(vis, weights, b_rot, pix_vec, freqs, *, beam=None, xp: ModuleType = np):
    """Per-frame dirty image from already-rotated baselines (device-pure; no host astropy).

    The frame rotation ``C(t)`` has already been folded into ``b_rot``; this is the pure-``xp``
    core shared by the CPU :func:`image_frame` and the GPU
    :class:`kremetart.operators.dft_healpix.HealpixDFTOperator`. It flattens ``(time, baseline)``
    into the row axis and adjoint-DFTs onto the fixed grid.

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
