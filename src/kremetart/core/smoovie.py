"""Render a TART HDF sequence into a HEALPix all-sky movie.

Images every sub-integration across the input HDFs onto the fixed equatorial (ICRS) HEALPix grid
(full sphere), all centered on a single common phase direction (the local zenith RA/Dec at the
global mid-time, reusable as the shared field center for multi-TART mosaicking). Renders Mollweide
frames with a fixed colour scale and encodes them to mp4 with ffmpeg. See
docs/superpowers/specs/2026-06-16-smoovie-common-frame-design.md (amending the original
docs/superpowers/specs/2026-06-15-smoovie-design.md).
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


def common_phase_direction(hdf_paths) -> tuple[float, float]:
    """Single shared ICRS phase direction: the local zenith RA/Dec at the global mid-time.

    Reads the first and last timestamps across all files, takes the midpoint, and converts the local
    zenith (AltAz alt=90 deg) at that time to ICRS. Reusable as the common field center for
    multi-TART mosaicking: compute once, hand the same value to every TART.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.

    Returns:
        ``(ra_deg, dec_deg)`` of the local zenith at the global mid-time, in ICRS.

    Raises:
        ValueError: if ``hdf_paths`` is empty.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    t_lo = t_hi = None
    info = None
    for path in hdf_paths:
        main = _partition(read_hdf_as_msv4(path)).ds
        times = np.asarray(main.time.values)
        lo, hi = float(times.min()), float(times.max())
        t_lo = lo if t_lo is None else min(t_lo, lo)
        t_hi = hi if t_hi is None else max(t_hi, hi)
        # Site info comes from the first file; all inputs are assumed to share the same array site.
        if info is None:
            info = main.attrs["observation_info"]
    if info is None:
        raise ValueError("no HDF files provided")

    t_mid = 0.5 * (t_lo + t_hi)
    loc = EarthLocation(
        lat=info["site_latitude_deg"] * u.deg,
        lon=info["site_longitude_deg"] * u.deg,
        height=info["site_altitude_m"] * u.m,
    )
    aa = AltAz(az=0.0 * u.deg, alt=90.0 * u.deg, obstime=Time(t_mid, format="unix", scale="utc"), location=loc)
    icrs = SkyCoord(aa).icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)


def frame_dirty_maps(hdf_paths, nside: int, *, xp=np):
    """Return (maps, timestamps, pix_vec): one full-sphere dirty map per sub-integration.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        xp: array module (numpy by default).

    Returns:
        ``(maps, timestamps, pix_vec)`` -- list of ``(npix,)`` real maps (one per sub-integration,
        across all files, in order), list of UTC strings, and the ``(npix, 3)`` pixel unit vectors.
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
        bl = itrs_baselines(node, xp)  # (nbl, 3)
        vis = np.asarray(main.VISIBILITY.values)[..., 0]  # (n_time, nbl, nchan), drop single-pol axis
        wgt = np.asarray(main.WEIGHT.values)[..., 0]
        freqs = np.asarray(main.frequency.values)
        for k in range(times.size):
            dmap = image_frame(vis[k : k + 1], wgt[k : k + 1], times[k : k + 1], bl, pix_vec, freqs, xp=xp)
            maps.append(np.asarray(dmap))
            stamps.append(_utc(times[k]))
    return maps, stamps, pix_vec


def render_frames(
    maps, timestamps, nside: int, cmap: str, outdir, *, rot: tuple[float, float] | None = None, nest: bool = True
):
    """Render each map as a Mollweide PNG with a fixed colour scale. Returns ordered PNG paths.

    ``rot=(lon, lat)`` (degrees) re-centers every frame on the common phase direction so the observed
    patch sits stably at the projection center across the movie.
    """
    import matplotlib

    matplotlib.use("Agg")
    import healpy as hp
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    stacked = np.concatenate([np.asarray(m) for m in maps])
    vmin, vmax = np.percentile(stacked, [1.0, 99.0])
    paths = []
    for i, (m, ts) in enumerate(zip(maps, timestamps)):
        hp.mollview(np.asarray(m), nest=nest, title=ts, cmap=cmap, min=float(vmin), max=float(vmax), rot=rot)
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


def smoovie(
    hdf_dir,
    movie,
    nside: int = 128,
    fps: int = 2,
    cmap: str = "inferno",
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
):
    """Render the HDF sequence in ``hdf_dir`` to an mp4 ``movie``. Returns the movie path.

    Every sub-integration becomes a frame, all imaged into the common ICRS frame and centered on the
    shared phase direction. If ``phase_ra_deg``/``phase_dec_deg`` are unset they default to the local
    zenith RA/Dec at the global mid-time (:func:`common_phase_direction`); supply both to override
    (the multi-TART mosaic hook). Supplying only one raises ``ValueError``.
    """
    import tempfile

    if hdf_dir is None or movie is None:
        raise ValueError("hdf_dir and movie are required")
    if (phase_ra_deg is None) != (phase_dec_deg is None):
        raise ValueError("phase_ra_deg and phase_dec_deg must be given together (both or neither)")
    hdf_dir = Path(hdf_dir)
    movie = Path(movie)
    hdf_paths = sorted(hdf_dir.glob("*.hdf"))
    if not hdf_paths:
        raise FileNotFoundError(f"no .hdf files found in {hdf_dir}")
    if phase_ra_deg is None:
        phase_ra_deg, phase_dec_deg = common_phase_direction(hdf_paths)
    maps, stamps, _ = frame_dirty_maps(hdf_paths, nside)
    with tempfile.TemporaryDirectory() as td:
        pngs = render_frames(maps, stamps, nside, cmap, Path(td), rot=(phase_ra_deg, phase_dec_deg))
        encode_movie(pngs, movie, fps)
    return movie
