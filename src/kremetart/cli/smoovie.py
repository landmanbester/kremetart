from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

File = NewType("File", Path)


@stimela_cab(
    name="smoovie",
    info="Render a TART HDF sequence into a HEALPix all-sky movie.",
)
@stimela_output(
    dtype="File",
    name="movie",
    info="Output mp4 movie",
)
def smoovie(
    hdf_dir: Annotated[
        str | None,
        typer.Option(
            help="Directory of input TART HDF snapshots",
        ),
        StimelaMeta(
            type="Directory",
        ),
    ] = None,
    nside: Annotated[
        int,
        typer.Option(
            help="HEALPix nside resolution.",
        ),
    ] = 128,
    fps: Annotated[
        int,
        typer.Option(
            help="Frames per second.",
        ),
    ] = 2,
    cmap: Annotated[
        str,
        typer.Option(
            help="Matplotlib colormap.",
        ),
    ] = "inferno",
    phase_ra_deg: Annotated[
        float | None,
        typer.Option(
            help="Common phase-direction RA (deg, ICRS); auto from global mid-time zenith if unset.",
        ),
    ] = None,
    phase_dec_deg: Annotated[
        float | None,
        typer.Option(
            help="Common phase-direction Dec (deg, ICRS); auto from global mid-time zenith if unset.",
        ),
    ] = None,
    movie: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Output mp4 movie",
        ),
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Render a TART HDF sequence into a HEALPix all-sky movie.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                smoovie,
                dict(
                    hdf_dir=hdf_dir,
                    nside=nside,
                    fps=fps,
                    cmap=cmap,
                    phase_ra_deg=phase_ra_deg,
                    phase_dec_deg=phase_dec_deg,
                    movie=movie,
                ),
            )

            # Lazy import the core implementation
            from kremetart.core.smoovie import smoovie as smoovie_core  # noqa: E402

            # Call the core function with all parameters
            smoovie_core(
                hdf_dir=hdf_dir,
                nside=nside,
                fps=fps,
                cmap=cmap,
                phase_ra_deg=phase_ra_deg,
                phase_dec_deg=phase_dec_deg,
                movie=movie,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("kremetart")
    if image is None:
        raise RuntimeError("No Container URL in kremetart metadata.")

    run_in_container(
        smoovie,
        dict(
            hdf_dir=hdf_dir,
            nside=nside,
            fps=fps,
            cmap=cmap,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            movie=movie,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
