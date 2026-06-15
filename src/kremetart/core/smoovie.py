"""Render a TART HDF sequence into a HEALPix all-sky movie.

Reads each HDF snapshot, images one representative sub-integration onto the fixed equatorial
HEALPix grid (full sphere), renders Mollweide frames with a fixed colour scale, and encodes them
to mp4 with ffmpeg. See docs/superpowers/specs/2026-06-15-smoovie-design.md.
"""

from __future__ import annotations

import datetime

import numpy as np


def _partition(dt):
    return dt[list(dt.children)[0]]


def _utc(unix_seconds) -> str:
    dt = datetime.datetime.fromtimestamp(float(unix_seconds), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def frame_dirty_maps(hdf_paths, nside: int, *, xp=np):
    """Return (maps, timestamps, pix_vec): one full-sphere dirty map per HDF (mid sub-integration).

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        xp: array module (numpy by default).

    Returns:
        ``(maps, timestamps, pix_vec)`` -- list of ``(npix,)`` real maps, list of UTC strings,
        and the ``(npix, 3)`` pixel unit vectors.
    """
    from kremetart.utils.healpix_dft import image_frame, make_pixel_grid
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.rephasing import itrs_baselines

    pix_vec = make_pixel_grid(nside, xp=xp)
    maps, stamps = [], []
    for path in hdf_paths:
        node = _partition(read_hdf_as_msv4(path))
        main = node.ds
        times = np.asarray(main.time.values)
        mid = times.size // 2
        bl = itrs_baselines(node, xp)  # (nbl, 3)
        vis = np.asarray(main.VISIBILITY.values)[mid : mid + 1, :, :, 0]  # drop single-pol axis
        wgt = np.asarray(main.WEIGHT.values)[mid : mid + 1, :, :, 0]
        freqs = np.asarray(main.frequency.values)
        dmap = image_frame(vis, wgt, times[mid : mid + 1], bl, pix_vec, freqs, xp=xp)
        maps.append(np.asarray(dmap))
        stamps.append(_utc(times[mid]))
    return maps, stamps, pix_vec
