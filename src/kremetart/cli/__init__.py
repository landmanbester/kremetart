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
from kremetart.cli.onboard import onboard  # noqa: E402

app.command(name="onboard")(onboard)

from kremetart.cli.stream_msv4 import stream_msv4  # noqa: E402

app.command(name="stream-msv4")(stream_msv4)

__all__ = ["app"]
