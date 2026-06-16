"""Render a TART HDF sequence into a HEALPix all-sky movie.

Images every sub-integration across the input HDFs onto the fixed equatorial (ICRS) HEALPix grid
(full sphere), all centered on a single common phase direction (the local zenith RA/Dec at the
global mid-time, reusable as the shared field center for multi-TART mosaicking). Renders Mollweide
frames with a fixed colour scale and encodes them to mp4 with ffmpeg. See
docs/superpowers/specs/2026-06-16-smoovie-common-frame-design.md (amending the original
docs/superpowers/specs/2026-06-15-smoovie-design.md).
"""

from __future__ import annotations

import contextlib
import datetime
import time
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


def _correct_file_gains(node, vis, wgt, *, xp=np):
    """Divide a file's vis/weight by the per-antenna gain product (``gain_xds.GAIN``).

    Maps each baseline to its two antenna gains the same way :func:`itrs_baselines` maps
    antennas, then delegates to :func:`kremetart.utils.gains.apply_inverse_gains`. The gain
    snapshot is per-file (time-independent), so this runs once before the sub-integration loop.
    """
    from kremetart.utils.gains import apply_inverse_gains

    antenna = node["antenna_xds"].to_dataset(inherit=False)
    index = {name: i for i, name in enumerate(antenna.antenna_name.values)}
    a1 = np.array([index[n] for n in node.ds.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in node.ds.baseline_antenna2_name.values])
    gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values
    return apply_inverse_gains(vis, wgt, gains, a1, a2, xp=xp)


def frame_dirty_maps(hdf_paths, nside: int, *, correct_gains: bool = False, nframes: int | None = None, xp=np):
    """Return (maps, timestamps, pix_vec): one full-sphere dirty map per sub-integration.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        correct_gains: divide vis/weights by the per-antenna gain product before imaging.
        nframes: optional cap on the total number of frames produced (profiling/preview aid).
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
        if nframes is not None and len(maps) >= nframes:
            break
        node = _partition(read_hdf_as_msv4(path))
        main = node.ds
        times = np.asarray(main.time.values)
        bl = itrs_baselines(node, xp)  # (nbl, 3)
        vis = np.asarray(main.VISIBILITY.values)[..., 0]  # (n_time, nbl, nchan), drop single-pol axis
        wgt = np.asarray(main.WEIGHT.values)[..., 0]
        freqs = np.asarray(main.frequency.values)
        if correct_gains:
            vis, wgt = _correct_file_gains(node, vis, wgt, xp=xp)
        for k in range(times.size):
            if nframes is not None and len(maps) >= nframes:
                break
            dmap = image_frame(vis[k : k + 1], wgt[k : k + 1], times[k : k + 1], bl, pix_vec, freqs, xp=xp)
            maps.append(np.asarray(dmap))
            stamps.append(_utc(times[k]))
    return maps, stamps, pix_vec


def _overlay_tracks(hp, tracks, frame_index):
    """Draw each satellite present in ``frame_index``: trailing line, current marker, name label.

    ``tracks`` maps name -> list of ``(frame_index, ra_deg, dec_deg, flux_jy)``. Coordinates are
    plotted with ``lonlat=True`` (degrees, ``lon == RA``) so healpy applies the active Mollweide
    ``rot`` and the overlay lands in the same projected ICRS frame as the imaged pixels.
    """
    for name, points in tracks.items():
        trail = [(ra, dec) for (f, ra, dec, _jy) in points if f <= frame_index]
        current = [(ra, dec) for (f, ra, dec, _jy) in points if f == frame_index]
        if not current:
            continue  # satellite not above the cutoff in this frame
        if len(trail) > 1:
            hp.projplot(
                [ra for ra, _ in trail],
                [dec for _, dec in trail],
                lonlat=True,
                color="cyan",
                linewidth=0.7,
                alpha=0.6,
            )
        ra0, dec0 = current[0]
        hp.projscatter([ra0], [dec0], lonlat=True, color="cyan", marker="x", s=30)
        hp.projtext(ra0, dec0, name, lonlat=True, color="cyan", fontsize=6)


def render_frames(
    maps,
    timestamps,
    nside: int,
    cmap: str,
    outdir,
    *,
    rot: tuple[float, float] | None = None,
    nest: bool = True,
    tracks=None,
):
    """Render each map as a Mollweide PNG with a fixed colour scale. Returns ordered PNG paths.

    ``rot=(lon, lat)`` (degrees) re-centers every frame on the common phase direction so the observed
    patch sits stably at the projection center across the movie. ``tracks`` (if given) overlays
    per-satellite ICRS trajectories (trailing line + current marker + name label) on each frame.
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
        if tracks:
            _overlay_tracks(hp, tracks, i)
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


@contextlib.contextmanager
def _stage_timer(name, timings):
    """Record wall-clock seconds for a named stage into ``timings`` (a list of ``(name, seconds)``)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings.append((name, time.perf_counter() - t0))


def _print_profile(timings, nframes):
    """Print a per-stage timing summary table to stdout."""
    total = sum(dt for _, dt in timings) or 1.0
    print("\n=== smoovie profile ===")
    print(f"{'stage':<18}{'seconds':>10}{'%total':>9}{'ms/frame':>11}")
    for name, dt in timings:
        per_frame = f"{1000.0 * dt / nframes:.1f}" if nframes else "-"
        print(f"{name:<18}{dt:>10.3f}{100.0 * dt / total:>8.1f}%{per_frame:>11}")
    print(f"{'TOTAL':<18}{total:>10.3f}{100.0:>8.1f}%")


def smoovie(
    hdf_dir,
    movie,
    nside: int = 128,
    fps: int = 2,
    cmap: str = "inferno",
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    correct_gains: bool = False,
    overlay_catalog: bool = False,
    catalog_elevation_deg: float = 45.0,
    catalog_cache: str | None = None,
    profile: bool = False,
    nframes: int | None = None,
):
    """Render the HDF sequence in ``hdf_dir`` to an mp4 ``movie``. Returns the movie path.

    Every sub-integration becomes a frame, all imaged into the common ICRS frame and centered on the
    shared phase direction. If ``phase_ra_deg``/``phase_dec_deg`` are unset they default to the local
    zenith RA/Dec at the global mid-time (:func:`common_phase_direction`); supply both to override
    (the multi-TART mosaic hook). Supplying only one raises ``ValueError``.

    ``correct_gains`` divides the visibilities by the per-antenna gain product (TART's own solution)
    before imaging. ``overlay_catalog`` overlays each catalogue satellite above ``catalog_elevation_deg``
    (degrees) as a trailing track + marker + label on every frame; it requires network access to the
    TART catalogue API. ``catalog_cache`` is the zarr path for the cached catalogue
    (``None`` -> ``<movie>.catalog.zarr``). ``profile`` prints a per-stage timing summary; ``nframes``
    caps the frames imaged/rendered (a profiling/preview aid).
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
    timings: list[tuple[str, float]] = []

    with _stage_timer("phase_direction", timings):
        if phase_ra_deg is None:
            phase_ra_deg, phase_dec_deg = common_phase_direction(hdf_paths)

    with _stage_timer("imaging", timings):
        maps, stamps, _ = frame_dirty_maps(hdf_paths, nside, correct_gains=correct_gains, nframes=nframes)

    tracks = None
    if overlay_catalog:
        from kremetart.utils.satellites import satellite_tracks

        cache_path = catalog_cache if catalog_cache is not None else str(movie) + ".catalog.zarr"
        with _stage_timer("catalog", timings):
            tracks = satellite_tracks(hdf_paths, catalog_elevation_deg, cache_path=cache_path, nframes=nframes)

    with tempfile.TemporaryDirectory() as td:
        with _stage_timer("render", timings):
            pngs = render_frames(maps, stamps, nside, cmap, Path(td), rot=(phase_ra_deg, phase_dec_deg), tracks=tracks)
        with _stage_timer("encode", timings):
            encode_movie(pngs, movie, fps)

    if profile:
        _print_profile(timings, len(maps))
    return movie
