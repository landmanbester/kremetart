"""
Image a TART HDF sequence into HEALPix all-sky maps and stream them to a live web viewer.

Host orchestration only: validates the options, resolves the common phase direction, prepares the HDFs
into one snapshot zarr, starts the web viewer (with optional satellite tracks), then hands the snapshots
to a runner (:mod:`kremetart.core.runner`): Holoscan on a CUDA machine, the NumPy loop otherwise. Every
snapshot yields ``dirty``, ``filtered`` and ``znorm`` maps (plus ``tikhonov`` and ``l1`` when enabled), written
to a durable ``(TIME, PIX)`` zarr and, when serving, streamed live; when the run finishes the viewer
freezes for scrub-back inspection.
"""

from __future__ import annotations

import tempfile
import threading
import webbrowser
from pathlib import Path

from kremetart.core.runner import select_runner
from kremetart.core.settings import ImagingSettings
from kremetart.utils.beam import GROUND_PLANE_DIAMETER
from kremetart.utils.geometry import common_phase_direction
from kremetart.utils.healpix_viz import LatestFrameHolder
from kremetart.utils.profiling import print_profile, stage_timer
from kremetart.utils.read_tart_hdf import prepare_msv4_zarr
from kremetart.utils.satellites import satellite_tracks
from kremetart.utils.web_server import FrameServer


def _wait_for_interrupt() -> None:
    """Block until the user interrupts (Ctrl-C), keeping the frozen session served."""
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


def smoovie(
    hdf_dir: Path,
    output: Path,
    nside: int = 128,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    correct_gains: bool = False,
    overlay_catalog: bool = False,
    catalog_elevation_deg: float = 45.0,
    catalog_cache: str | None = None,
    profile: bool = False,
    iwp_sigma: float = 1e-3,
    iwp_noise: float = 1e-2,
    apply_beam: bool = True,
    ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
    l2: float | None = None,
    l1: float | None = None,
    overwrite: bool = False,
    nframes: int | None = None,
    serve: bool = True,
    port: int = 8080,
    open_browser: bool = False,
    runner: str = "auto",
) -> Path:
    """Image the HDF sequence in ``hdf_dir`` into the durable ``output`` zarr; serve a live view.

    Every sub-integration is imaged into a common ICRS HEALPix frame centered on the shared phase
    direction. If ``phase_ra_deg``/``phase_dec_deg`` are unset they default to the local zenith
    RA/Dec at the global mid-time (:func:`common_phase_direction`); supplying only one raises
    ``ValueError``. Writes ``output`` (a ``(TIME, PIX)`` zarr holding ``dirty``/
    ``filtered``/``znorm``, plus ``tikhonov``/``l1`` when enabled); raises ``FileExistsError`` if it exists unless
    ``overwrite=True``.

    Every snapshot gets a dirty map. The deconvolvers are opt-in because they dominate the runtime:
    ``l2`` enables Tikhonov (CG) and ``l1`` reweighted-L1 (FISTA), each strength a fraction of the
    snapshot weight sum (``lambda = strength * Σw``; try ``0.01``, and see
    ``scripts/compare_regularisers.py``). The IWP-Kalman filter smooths the Tikhonov stream when
    ``l2`` is set and the dirty stream otherwise; ``filtered`` and ``znorm`` track that stream.

    With ``serve`` (the default), a FastAPI+uvicorn server starts on ``port``, the view URL is
    printed (and opened if ``open_browser``), frames stream live to an interactive sphere, and when
    the pipeline finishes the session freezes for scrub-back inspection until interrupted (Ctrl-C).
    ``--no-serve`` images headlessly (batch use). ``overlay_catalog`` (serve only)
    overlays each catalogue satellite above ``catalog_elevation_deg`` as 3D tracks; it requires
    network access to the TART catalogue API, cached at ``catalog_cache`` (``None`` ->
    ``<output>.catalog.zarr``). ``profile`` prints the imaging wall-time; ``nframes`` caps the frames
    imaged; ``iwp_sigma``/``iwp_noise`` set the per-pixel filter's σ²/R. ``apply_beam`` (default on)
    folds the Airy primary beam of a ``ground_plane_diameter``-metre aperture into the measurement
    operator so the maps are oriented toward the intrinsic sky.

    ``runner`` picks what drives the stages (:func:`select_runner`): ``auto`` (Holoscan on a CUDA machine,
    the NumPy loop otherwise), ``loop``, ``holoscan`` or ``cyberether``.

    Returns:
        The ``output`` zarr path.
    """
    if hdf_dir is None or output is None:
        raise ValueError("hdf_dir and output are required")
    if (phase_ra_deg is None) != (phase_dec_deg is None):
        raise ValueError("phase_ra_deg and phase_dec_deg must be given together (both or neither)")
    # Before any data access, so a runner that cannot start fails at once.
    choice = select_runner(runner)
    print(f"kremetart: imaging on {choice.description}")
    hdf_dir = Path(hdf_dir)
    output = Path(output)
    hdf_paths = sorted(hdf_dir.glob("*.hdf"))
    if not hdf_paths:
        raise FileNotFoundError(f"no .hdf files found in {hdf_dir}")

    if phase_ra_deg is None:
        phase_ra_deg, phase_dec_deg = common_phase_direction(hdf_paths)

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass overwrite=True to replace it.")

    tracks = None
    if serve and overlay_catalog:
        cache_path = catalog_cache if catalog_cache is not None else str(output) + ".catalog.zarr"
        tracks = satellite_tracks(hdf_paths, catalog_elevation_deg, cache_path=cache_path, nframes=nframes)

    settings = ImagingSettings(
        nside=nside,
        l2=l2,
        l1=l1,
        iwp_sigma=iwp_sigma,
        iwp_noise=iwp_noise,
        apply_beam=apply_beam,
        ground_plane_diameter=ground_plane_diameter,
    )

    holder = None
    server = None
    url = None
    if serve:
        names = settings.stream_names
        holder = LatestFrameHolder(names)
        server = FrameServer(
            holder, nside=nside, nest=settings.nest, names=names, units=settings.stream_units, port=port, tracks=tracks
        )
        url = server.start()
        print(f"kremetart live view: {url}")
        if open_browser:
            webbrowser.open(url)

    timings: list = []
    try:
        with stage_timer("imaging", timings), tempfile.TemporaryDirectory() as td:
            prepared = Path(td) / "prepared.zarr"
            prepare_msv4_zarr(
                hdf_paths,
                prepared,
                correct_gains=correct_gains,
                phase_ra_deg=phase_ra_deg,
                phase_dec_deg=phase_dec_deg,
                nframes=nframes,
            )
            choice.run(prepared, output, settings, holder)
        if serve:
            holder.finish()
            print(f"pipeline finished — serving frozen session at {url} (Ctrl-C to exit)")
            _wait_for_interrupt()
    finally:
        if serve and server is not None:
            # Flush a clean end even if imaging raised, so connected clients freeze
            # (and stop reconnecting) instead of looping against the stopped server.
            if holder is not None:
                holder.finish()
            server.stop()

    if profile:
        print_profile(timings, nframes=nframes)

    return output
