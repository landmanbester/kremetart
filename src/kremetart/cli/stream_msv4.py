from typing import Annotated, Literal

import typer
from hip_cargo import StimelaMeta, stimela_cab, stimela_output


@stimela_cab(
    name="stream-msv4",
    info="Stream MSv4 data to a Zarr dataset.",
)
@stimela_output(
    dtype="File",
    name="output_dataset",
    info="Output Zarr dataset",
)
def stream_msv4(
    ms: Annotated[
        str | None,
        typer.Option(
            help="Input data source",
        ),
        StimelaMeta(
            type="Directory",
        ),
    ] = None,
    output_dataset: Annotated[
        str | None,
        typer.Option(
            help="Output Zarr dataset",
        ),
        StimelaMeta(
            type="Directory",
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
    Stream MSv4 data to a Zarr dataset.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                stream_msv4,
                dict(
                    ms=ms,
                    output_dataset=output_dataset,
                ),
            )

            # Lazy import the core implementation
            from kremetart.core.stream_msv4 import stream_msv4 as stream_msv4_core  # noqa: E402

            # Call the core function with all parameters
            stream_msv4_core(
                ms=ms,
                output_dataset=output_dataset,
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
        stream_msv4,
        dict(
            ms=ms,
            output_dataset=output_dataset,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
