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
        **kwargs,
    ):
        self.prepared_zarr = str(prepared_zarr)
        self.output_zarr = str(output_zarr)
        self.nside = nside
        self.nest = nest
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
        self.add_flow(imager, writer, {("cube", "cube"), ("time_out", "time_out")})


def image_via_app(
    hdf_paths,
    nside: int,
    *,
    correct_gains: bool = False,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    nframes: int | None = None,
    nest: bool = True,
):
    """Image the HDF sequence through the GPU Holoscan app; return ``(maps, stamps)``.

    Runs the host prepare-step into a temp zarr, streams it through :class:`SmooviePipeline`, then
    loads the ``(TIME, npix)`` output zarr back to host: a list of ``(npix,)`` dirty maps (one per
    frame, in order) and a list of UTC stamp strings. This is the single imaging seam :func:`smoovie`
    drives; the wiring tests stub it to exercise the host orchestration without running the GPU.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        correct_gains: apply the inverse per-antenna gain solution in the prepare-step.
        phase_ra_deg, phase_dec_deg: common phase direction (deg, ICRS), stored as zarr metadata.
        nframes: optional cap on the number of frames imaged.
        nest: NESTED HEALPix ordering (default True; matches the rest of the pipeline).

    Returns:
        ``(maps, stamps)``.
    """
    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "prepared.zarr"
        output = Path(td) / "dirty.zarr"
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

        app = SmooviePipeline(prepared, output, nside, nest=nest)
        app.config(str(config))
        app.run()

        ds = xr.open_zarr(str(output), chunks=None)  # eager load before the temp dir is removed
        dirty = np.asarray(ds["dirty"].values)  # (ntime, npix)
        times = np.asarray(ds["TIME"].values)

    maps = list(dirty)
    stamps = [unix_to_utc(t) for t in times]
    return maps, stamps


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
):
    """Render each map as a Mollweide PNG with a fixed colour scale. Returns ordered PNG paths.

    ``rot=(lon, lat)`` (degrees) re-centers every frame on the common phase direction so the observed
    patch sits stably at the projection center across the movie. ``tracks`` (if given) overlays
    per-satellite ICRS trajectories (trailing line + current marker + name label) on each frame.
    """

    outdir = Path(outdir)
    stacked = np.concatenate([np.asarray(m) for m in maps])
    vmin, vmax = np.percentile(stacked, [1.0, 99.0])
    paths = []
    for i, (m, ts) in enumerate(zip(maps, timestamps)):
        hp.mollview(np.asarray(m), nest=nest, title=ts, cmap=cmap, min=float(vmin), max=float(vmax), rot=rot)
        hp.graticule()
        if tracks:
            _overlay_tracks(plt.gca(), tracks, i)
        out = outdir / f"frame_{i:04d}.png"
        plt.savefig(out, dpi=100)
        plt.close("all")
        paths.append(out)
    return paths


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

    timings = []
    with stage_timer("imaging", timings):
        maps, stamps = image_via_app(
            hdf_paths,
            nside,
            correct_gains=correct_gains,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            nframes=nframes,
        )

    tracks = None
    if overlay_catalog:
        cache_path = catalog_cache if catalog_cache is not None else str(movie) + ".catalog.zarr"
        tracks = satellite_tracks(hdf_paths, catalog_elevation_deg, cache_path=cache_path, nframes=nframes)

    with tempfile.TemporaryDirectory() as td:
        with stage_timer("render", timings):
            png_paths = render_frames(
                maps, stamps, nside, cmap, Path(td), rot=(phase_ra_deg, phase_dec_deg), tracks=tracks
            )
        with stage_timer("encode", timings):
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

    if profile:
        print_profile(timings, nframes=len(maps))

    return movie
