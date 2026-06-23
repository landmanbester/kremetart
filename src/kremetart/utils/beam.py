"""Analytic Airy primary beam for the GPS patch antenna, evaluated on the HEALPix grid.

The GPS patch antenna's response is approximated by the far-field diffraction pattern of a
uniformly-illuminated circular aperture whose diameter equals the antenna ground plane
(125 mm). For an aperture of diameter ``D`` observed at angle ``theta`` from boresight, the
voltage pattern is the classic Airy function ``A(theta) = 2 J1(x) / x`` with
``x = (pi D / lambda) sin(theta)``, and the power pattern is ``A(theta)**2`` (peak 1 at
boresight). The ground plane shields the back hemisphere, so the power beam is zeroed for
pixels below the local horizon (``cos(theta) < 0``).

The beam is evaluated per frame on the equatorial HEALPix grid of :mod:`kremetart.utils.healpix_dft`:
``pix_vec`` are the same pixel unit vectors, and ``boresight`` is the antenna-zenith unit vector
expressed in that grid frame for the frame's timestamp (deriving it from astropy is the caller's
job, mirroring :func:`kremetart.utils.healpix_dft._icrs_to_itrs_matrices`).

The math is ``xp``-injectable: pass ``xp=numpy`` (CPU tests) or ``xp=cupy`` (GPU pipeline). The
Bessel term dispatches to ``scipy.special.j1`` on the host and ``cupyx.scipy.special.j1`` on the
device, the same host/device split used elsewhere in :mod:`kremetart.utils`.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
from scipy.special import j1 as _scipy_j1

LIGHTSPEED = 299792458.0
GROUND_PLANE_DIAMETER = 0.125  # metres -- GPS patch antenna ground plane / Airy aperture diameter


def _bessel_j1(x, xp: ModuleType):
    """First-order Bessel function ``J1`` on the array module ``xp``.

    Args:
        x: input array living on ``xp``.
        xp: ``numpy`` (host, via scipy) or ``cupy`` (device, via cupyx).

    Returns:
        ``J1(x)`` as an ``xp`` array.
    """
    if xp.__name__ == "cupy":
        from cupyx.scipy.special import j1 as _cupy_j1  # GPU-only dep: lazy so the module imports without cupy

        return _cupy_j1(x)
    return _scipy_j1(x)


def airy_power_beam(
    pix_vec,
    boresight,
    freqs,
    *,
    diameter: float = GROUND_PLANE_DIAMETER,
    xp: ModuleType = np,
):
    """Airy power beam of the GPS ground-plane aperture, sampled on the HEALPix grid.

    Computes ``B(theta) = [2 J1(x) / x]**2`` with ``x = (pi D / lambda) sin(theta)`` and
    ``cos(theta) = pix_vec . boresight``, normalised to 1 at boresight and zeroed below the
    local horizon. One call evaluates one frame across all channels.

    Args:
        pix_vec: ``(npix, 3)`` HEALPix pixel unit vectors from
            :func:`kremetart.utils.healpix_dft.make_pixel_grid`.
        boresight: ``(3,)`` antenna-zenith unit vector in the same frame as ``pix_vec`` for this
            frame's timestamp. Normalised internally.
        freqs: ``(nchan,)`` frequencies in Hz (same convention as
            :mod:`kremetart.utils.healpix_dft`).
        diameter: aperture (ground plane) diameter in metres. Defaults to
            :data:`GROUND_PLANE_DIAMETER`.
        xp: array module -- ``numpy`` (CPU) or ``cupy`` (GPU).

    Returns:
        ``(nchan, npix)`` real power beam, peak 1 at boresight, 0 below the horizon.
    """
    pix_vec = xp.asarray(pix_vec)
    boresight = xp.asarray(boresight)
    boresight = boresight / xp.linalg.norm(boresight)  # defensive: guarantee a unit boresight

    mu = xp.clip(pix_vec @ boresight, -1.0, 1.0)  # cos(theta), (npix,)
    sinth = xp.sqrt(1.0 - mu**2)  # sin(theta), (npix,)

    inv_wl = xp.asarray(freqs) / LIGHTSPEED  # cycles per metre, (nchan,)
    x = xp.pi * diameter * inv_wl[:, None] * sinth[None, :]  # (nchan, npix)

    # Airy voltage 2 J1(x) / x, with the x -> 0 boresight limit A(0) = 1 handled without 0/0.
    safe_x = xp.where(x == 0.0, 1.0, x)
    amp = xp.where(x == 0.0, 1.0, 2.0 * _bessel_j1(safe_x, xp) / safe_x)

    beam = amp**2
    return xp.where(mu[None, :] >= 0.0, beam, 0.0)  # ground plane blocks the back hemisphere
