"""CLI for kremetart."""

import typer

app = typer.Typer(
    name="kremetart",
    help="Kalman Real-time Evidence Monitoring Extractor for TART",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Kalman Real-time Evidence Monitoring Extractor for TART"""
    pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from kremetart.cli.smoovie import smoovie  # noqa: E402

app.command(name="smoovie")(smoovie)

__all__ = ["app"]
