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
