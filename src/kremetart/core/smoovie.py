"""
Render a TART HDF sequence into a HEALPix all-sky movie.
Starts by concatenating the HDFs into a single MSv4 zarr, then streams it through the GPU Holoscan app.
The core imaging algorithm is implemented as a GPU Holoscan operator which should completely bypass the CPU.
Every sub-integration is imaged in a fixed equatorial (ICRS) HEALPix grid with a common phase center.
Renders Mollweide frames with a fixed colour scale and encodes them to mp4 with ffmpeg.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import shutil
import subprocess
import tempfile

import healpy as hp
import holoscan as hs
import matplotlib.pyplot as plt
import xarray as xr
from holoscan.conditions import CountCondition

from kremetart.operators.dft_healpix import HealpixDFTOperator
from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator
from kremetart.operators.iwp_kalman import IWPKalmanOperator
from kremetart.utils import unix_to_utc
from kremetart.utils.profiling import print_profile, stage_timer
from kremetart.utils.read_tart_hdf import prepare_msv4_zarr
from kremetart.utils.rephasing import common_phase_direction
from kremetart.utils.satellites import satellite_tracks


class SmooviePipeline(hs.core.Application):
    """Stream a prepared imaging zarr through the GPU HEALPix imager into a ``(TIME, npix)`` zarr."""

    def __init__(
        self,
        prepared_zarr: Path | str,
        output_zarr: Path | str,
        nside: int,
        *args,
        nest: bool = True,
        sigma2: float = 1e-3,
        noise: float = 1e-2,
        **kwargs,
    ):
        self.prepared_zarr = str(prepared_zarr)
        self.output_zarr = str(output_zarr)
        self.nside = nside
        self.nest = nest
        self.sigma2 = sigma2
        self.noise = noise
        super().__init__(*args, **kwargs)

        ds = xr.open_zarr(self.prepared_zarr)
        self.ntime = int(ds.time.size)
        self.out_times = ds.time.values
        self.freqs = ds.frequency.values
        self.npix = hp.nside2npix(nside)

    def compose(self):
        reader = HealpixZarrReaderOperator(
            self,
            CountCondition(self, self.ntime),
            name="reader",
            zarr_path=self.prepared_zarr,
        )
        imager = HealpixDFTOperator(self, self.nside, self.freqs, name="imager", nest=self.nest)
        iwp = IWPKalmanOperator(self, self.npix, name="iwp", sigma2=self.sigma2, noise=self.noise)
        writer = HealpixWriterOperator(
            self,
            self.ntime,
            self.npix,
            name="writer",
            output_dataset=self.output_zarr,
            out_times=self.out_times,
        )
        self.add_flow(
            reader,
            imager,
            {("VISIBILITY", "VISIBILITY"), ("WEIGHT", "WEIGHT"), ("B_ROT", "B_ROT"), ("time", "time")},
        )
        self.add_flow(imager, iwp, {("cube", "cube"), ("time_out", "time_out")})
        self.add_flow(
            iwp,
            writer,
            {("cube", "cube"), ("filtered", "filtered"), ("znorm", "znorm"), ("time_out", "time_out")},
        )


def image_via_app(
    hdf_paths,
    nside: int,
    *,
    output_zarr,
    correct_gains: bool = False,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    nframes: int | None = None,
    nest: bool = True,
    iwp_sigma: float = 1e-3,
    iwp_noise: float = 1e-2,
):
    """Image the HDF sequence through the GPU Holoscan app; return ``(dirty, filtered, znorm, stamps)``.

    Runs the host prepare-step into a temp zarr, streams it through :class:`SmooviePipeline` (imager
    -> per-pixel IWP-Kalman filter -> writer), and writes a DURABLE ``output_zarr`` holding the
    ``(TIME, PIX)`` ``dirty`` / ``filtered`` / ``znorm`` variables, left in place for inspection.
    Loads each variable back to host as a list of ``(npix,)`` maps (one per frame, in order) plus a
    list of UTC stamp strings. This is the single imaging seam :func:`smoovie` drives; the
    host-wiring tests stub it to exercise orchestration without running the GPU.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        output_zarr: durable output zarr path; the caller is responsible for the fail-fast existence
            check before calling this function.
        correct_gains: apply the inverse per-antenna gain solution in the prepare-step.
        phase_ra_deg, phase_dec_deg: common phase direction (deg, ICRS), stored as zarr metadata.
        nframes: optional cap on the number of frames imaged.
        nest: NESTED HEALPix ordering (default True).
        iwp_sigma: IWP driving variance sigma^2.
        iwp_noise: measurement-noise variance R.

    Returns:
        ``(dirty, filtered, znorm, stamps)``.
    """
    output = Path(output_zarr)
    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "prepared.zarr"
        config = Path(td) / "config.yaml"
        config.touch()  # an empty Holoscan config is valid

        prepare_msv4_zarr(
            hdf_paths,
            prepared,
            correct_gains=correct_gains,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            nframes=nframes,
        )

        app = SmooviePipeline(prepared, output, nside, nest=nest, sigma2=iwp_sigma, noise=iwp_noise)
        app.config(str(config))
        app.run()

        ds = xr.open_zarr(str(output), chunks=None)  # eager load
        dirty = np.asarray(ds["dirty"].values)  # (ntime, npix)
        filtered = np.asarray(ds["filtered"].values)
        znorm = np.asarray(ds["znorm"].values)
        times = np.asarray(ds["TIME"].values)

    stamps = [unix_to_utc(t) for t in times]
    return list(dirty), list(filtered), list(znorm), stamps


def _overlay_tracks(ax, tracks, frame_index):
    """Draw each satellite present in ``frame_index``: trailing line, current marker, name label.

    ``tracks`` maps name -> list of ``(frame_index, ra_deg, dec_deg, flux_jy)``. ``ax`` is the active
    healpy Mollweide projection axes (``plt.gca()`` after ``mollview``); its ``projscatter`` /
    ``projplot`` / ``projtext`` methods are called directly rather than the module-level ``hp.proj*``
    wrappers, because each wrapper forces a full ``pylab.draw()`` on every call -- turning an
    N-satellite overlay into ~N full-figure re-rasterizations per frame (the cause of ~15 s/frame
    rendering). The axes methods draw nothing until the single ``savefig`` per frame. Coordinates use
    ``lonlat=True`` (degrees, ``lon == RA``) so the active ``rot`` is applied and the overlay lands in
    the same projected ICRS frame as the imaged pixels.
    """
    for name, points in tracks.items():
        trail = [(ra, dec) for (f, ra, dec, _jy) in points if f <= frame_index]
        current = [(ra, dec) for (f, ra, dec, _jy) in points if f == frame_index]
        if not current:
            continue  # satellite not above the cutoff in this frame
        if len(trail) > 1:
            ax.projplot(
                [ra for ra, _ in trail],
                [dec for _, dec in trail],
                lonlat=True,
                color="cyan",
                linewidth=0.7,
                alpha=0.6,
            )
        ra0, dec0 = current[0]
        ax.projscatter([ra0], [dec0], lonlat=True, color="cyan", marker="x", s=30)


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
    diverging: bool = False,
):
    """Render each map as a Mollweide PNG with a fixed colour scale. Returns ordered PNG paths.

    ``rot=(lon, lat)`` (degrees) re-centers every frame on the common phase direction so the observed
    patch sits stably at the projection center across the movie. ``tracks`` (if given) overlays
    per-satellite ICRS trajectories (trailing line + current marker + name label) on each frame.
    """

    outdir = Path(outdir)
    stacked = np.concatenate([np.asarray(m) for m in maps])
    if diverging:
        # Symmetric scale centred on 0 with a diverging cmap (for the normalised innovation z_k).
        vmax = float(np.percentile(np.abs(stacked), 99.0))
        vmin, cmap = -vmax, "coolwarm"
    else:
        vmin, vmax = (float(v) for v in np.percentile(stacked, [1.0, 99.0]))
    paths = []
    for i, (m, ts) in enumerate(zip(maps, timestamps)):
        hp.mollview(np.asarray(m), nest=nest, title=ts, cmap=cmap, min=vmin, max=vmax, rot=rot)
        hp.graticule()
        if tracks:
            _overlay_tracks(plt.gca(), tracks, i)
        out = outdir / f"frame_{i:04d}.png"
        plt.savefig(out, dpi=100)
        plt.close("all")
        paths.append(out)
    return paths


def _encode_movie(first_png, fps: int, out) -> None:
    """Encode the ``frame_%04d.png`` sequence in ``first_png``'s directory to mp4 ``out``."""
    pattern = str(Path(first_png).parent / "frame_%04d.png")
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
            str(out),
        ],
        check=True,
        capture_output=True,
    )


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
    iwp_sigma: float = 1e-3,
    iwp_noise: float = 1e-2,
    overwrite: bool = False,
    nframes: int | None = None,
):
    """Render the HDF sequence in ``hdf_dir`` to an mp4 ``movie``. Returns the movie path.

    Every sub-integration becomes a frame, all imaged into the common ICRS frame and centered on the
    shared phase direction. If ``phase_ra_deg``/``phase_dec_deg`` are unset they default to the local
    zenith RA/Dec at the global mid-time (:func:`common_phase_direction`); supply both to override
    (the multi-TART mosaic hook). Supplying only one raises ``ValueError``.

    Writes a durable ``<movie>.zarr`` holding per-pixel ``dirty``, ``filtered``, and ``znorm``
    ``(TIME, PIX)`` maps, then renders THREE movies: ``<movie>`` (dirty flux), ``<movie>.filtered.mp4``
    (IWP-Kalman filtered flux), and ``<movie>.znorm.mp4`` (normalised innovation on a diverging colour
    scale). Raises ``FileExistsError`` if ``<movie>.zarr`` already exists unless ``overwrite=True``.

    ``correct_gains`` divides the visibilities by the per-antenna gain product (TART's own solution)
    before imaging. ``overlay_catalog`` overlays each catalogue satellite above ``catalog_elevation_deg``
    (degrees) as a trailing track + marker + label on every frame; it requires network access to the
    TART catalogue API. ``catalog_cache`` is the zarr path for the cached catalogue
    (``None`` -> ``<movie>.catalog.zarr``). ``profile`` prints a per-stage timing summary; ``nframes``
    caps the frames imaged/rendered (a profiling/preview aid). ``iwp_sigma`` sets the IWP driving
    variance σ², ``iwp_noise`` sets the measurement-noise variance R, and ``overwrite`` allows
    replacing an existing ``<movie>.zarr``.
    """

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

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required to encode the movie.")

    output_zarr = Path(str(movie) + ".zarr")
    if output_zarr.exists() and not overwrite:
        raise FileExistsError(f"{output_zarr} already exists; pass overwrite=True to replace it.")

    timings = []
    with stage_timer("imaging", timings):
        dirty, filtered, znorm, stamps = image_via_app(
            hdf_paths,
            nside,
            output_zarr=output_zarr,
            correct_gains=correct_gains,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            nframes=nframes,
            iwp_sigma=iwp_sigma,
            iwp_noise=iwp_noise,
        )

    tracks = None
    if overlay_catalog:
        cache_path = catalog_cache if catalog_cache is not None else str(movie) + ".catalog.zarr"
        tracks = satellite_tracks(hdf_paths, catalog_elevation_deg, cache_path=cache_path, nframes=nframes)

    movie_specs = [
        (dirty, Path(movie), False),
        (filtered, Path(str(movie) + ".filtered.mp4"), False),
        (znorm, Path(str(movie) + ".znorm.mp4"), True),
    ]
    with tempfile.TemporaryDirectory() as td:
        with stage_timer("render", timings):
            rendered = []
            for frames, out, diverging in movie_specs:
                subdir = Path(td) / out.name
                subdir.mkdir()
                pngs = render_frames(
                    frames,
                    stamps,
                    nside,
                    cmap,
                    subdir,
                    rot=(phase_ra_deg, phase_dec_deg),
                    tracks=tracks,
                    diverging=diverging,
                )
                rendered.append((pngs, out))
        with stage_timer("encode", timings):
            for pngs, out in rendered:
                _encode_movie(pngs[0], fps, out)

    if profile:
        print_profile(timings, nframes=len(dirty))

    return movie
