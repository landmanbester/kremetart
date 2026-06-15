"""Render a TART HDF sequence into a HEALPix all-sky movie.

Reads each HDF snapshot, images one representative sub-integration onto the fixed equatorial
HEALPix grid (full sphere), renders Mollweide frames with a fixed colour scale, and encodes them
to mp4 with ffmpeg. See docs/superpowers/specs/2026-06-15-smoovie-design.md.
"""

from __future__ import annotations

import datetime
from pathlib import Path

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


def render_frames(maps, timestamps, nside: int, cmap: str, outdir, *, nest: bool = True):
    """Render each map as a Mollweide PNG with a fixed colour scale. Returns ordered PNG paths."""
    import matplotlib

    matplotlib.use("Agg")
    import healpy as hp
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    stacked = np.concatenate([np.asarray(m) for m in maps])
    vmin, vmax = np.percentile(stacked, [1.0, 99.0])
    paths = []
    for i, (m, ts) in enumerate(zip(maps, timestamps)):
        hp.mollview(np.asarray(m), nest=nest, title=ts, cmap=cmap, min=float(vmin), max=float(vmax))
        hp.graticule()
        out = outdir / f"frame_{i:04d}.png"
        plt.savefig(out, dpi=100)
        plt.close("all")
        paths.append(out)
    return paths


def encode_movie(png_paths, movie, fps: int):
    """Encode an ordered PNG sequence into an mp4 with ffmpeg. Returns the movie path."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required to encode the movie.")
    movie = Path(movie)
    pattern = str(Path(png_paths[0]).parent / "frame_%04d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(movie),
        ],
        check=True,
        capture_output=True,
    )
    return movie


def smoovie(hdf_dir, movie, nside: int = 128, fps: int = 2, cmap: str = "inferno"):
    """Render the HDF sequence in ``hdf_dir`` to an mp4 ``movie``. Returns the movie path."""
    import tempfile

    if hdf_dir is None or movie is None:
        raise ValueError("hdf_dir and movie are required")
    hdf_dir = Path(hdf_dir)
    movie = Path(movie)
    hdf_paths = sorted(hdf_dir.glob("*.hdf"))
    if not hdf_paths:
        raise FileNotFoundError(f"no .hdf files found in {hdf_dir}")
    maps, stamps, _ = frame_dirty_maps(hdf_paths, nside)
    with tempfile.TemporaryDirectory() as td:
        pngs = render_frames(maps, stamps, nside, cmap, Path(td))
        encode_movie(pngs, movie, fps)
    return movie
