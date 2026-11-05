"""Command-line interface for kremetart: a single command, ``kremetart``."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class RunnerName(str, Enum):
    """The ``--runner`` choices; ``tests/test_cli.py`` keeps them equal to ``kremetart.core.runner.RUNNERS``."""

    auto = "auto"
    loop = "loop"
    holoscan = "holoscan"
    cyberether = "cyberether"
    cyberether_gui = "cyberether-gui"


app = typer.Typer(name="kremetart")


@app.command(no_args_is_help=True)
def kremetart(
    hdf_dir: Annotated[Path, typer.Option(help="Directory of input TART HDF snapshots.")],
    output: Annotated[Path, typer.Option(help="Output HEALPix (TIME, PIX) zarr.")],
    nside: Annotated[int, typer.Option(help="HEALPix nside resolution.")] = 128,
    phase_ra_deg: Annotated[
        float | None,
        typer.Option(help="Common phase-direction RA (deg, ICRS); auto from global mid-time zenith if unset."),
    ] = None,
    phase_dec_deg: Annotated[
        float | None,
        typer.Option(help="Common phase-direction Dec (deg, ICRS); auto from global mid-time zenith if unset."),
    ] = None,
    correct_gains: Annotated[
        bool, typer.Option(help="Apply the inverse per-antenna gain solution before imaging.")
    ] = False,
    overlay_catalog: Annotated[
        bool, typer.Option(help="Overlay TART catalog satellite tracks on the live sphere (requires network).")
    ] = False,
    catalog_elevation_deg: Annotated[
        float, typer.Option(help="Elevation cutoff (deg) for catalog sources to overlay.")
    ] = 45.0,
    catalog_cache: Annotated[
        str | None, typer.Option(help="Catalog cache zarr path; defaults to <output>.catalog.zarr.")
    ] = None,
    profile: Annotated[bool, typer.Option(help="Print a per-stage timing summary.")] = False,
    iwp_sigma: Annotated[
        float, typer.Option(help="IWP driving variance (sigma^2) for the per-pixel Kalman filter.")
    ] = 0.001,
    iwp_noise: Annotated[
        float, typer.Option(help="Measurement-noise variance (R) for the per-pixel Kalman filter.")
    ] = 0.01,
    apply_beam: Annotated[
        bool, typer.Option(help="Fold the Airy primary beam into the measurement operator (image the intrinsic sky).")
    ] = True,
    ground_plane_diameter: Annotated[
        float, typer.Option(help="Airy aperture (ground plane) diameter in metres.")
    ] = 0.125,
    l2: Annotated[
        float | None,
        typer.Option(
            help="Tikhonov (L2) deconvolution strength as a fraction of weight sum (lambda = l2 * sum w). "
            "Off by default because it is slow on the CPU; pass a value such as 0.01 to enable it. "
            "The Kalman filter smooths the Tikhonov maps when enabled, the dirty maps otherwise."
        ),
    ] = None,
    l1: Annotated[
        float | None,
        typer.Option(
            help="Reweighted-L1 (FISTA) deconvolution strength as a fraction of weight sum (lambda = l1 * sum w). "
            "Off by default because it dominates the runtime; pass a value such as 0.01 to enable it."
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Overwrite the output zarr if it already exists.")] = False,
    nframes: Annotated[
        int | None, typer.Option(help="Cap the number of snapshots imaged (profiling/preview aid).")
    ] = None,
    serve: Annotated[
        bool, typer.Option(help="Serve the live web viewer (use --no-serve for headless/batch runs).")
    ] = True,
    port: Annotated[int, typer.Option(help="Port for the live web viewer.")] = 8080,
    open_browser: Annotated[bool, typer.Option(help="Open the live web viewer in a browser on startup.")] = False,
    runner: Annotated[
        RunnerName,
        typer.Option(
            help="What drives the imaging: auto (Holoscan on a CUDA machine, the NumPy loop otherwise), loop, "
            "holoscan, cyberether (needs KREMETART_CYBERETHER or `cyberether` on PATH), or cyberether-gui (the same "
            "flowgraph in a CyberEther window; close it to finish).",
        ),
    ] = RunnerName.auto,
) -> None:
    """Kalman Real-time Evidence Monitoring Extractor for TART.

    Image a TART HDF sequence into HEALPix maps and stream them to a live web viewer.
    """
    # Deferred so that `kremetart --help` stays fast: the command pulls in xarray, astropy and the stages.
    from kremetart.core.kremetart import kremetart as run_kremetart

    run_kremetart(
        hdf_dir=hdf_dir,
        output=output,
        nside=nside,
        phase_ra_deg=phase_ra_deg,
        phase_dec_deg=phase_dec_deg,
        correct_gains=correct_gains,
        overlay_catalog=overlay_catalog,
        catalog_elevation_deg=catalog_elevation_deg,
        catalog_cache=catalog_cache,
        profile=profile,
        iwp_sigma=iwp_sigma,
        iwp_noise=iwp_noise,
        apply_beam=apply_beam,
        ground_plane_diameter=ground_plane_diameter,
        l2=l2,
        l1=l1,
        overwrite=overwrite,
        nframes=nframes,
        serve=serve,
        port=port,
        open_browser=open_browser,
        runner=runner.value,
    )
