"""The ``kremetart`` CLI: option parsing and hand-off to :func:`kremetart.core.smoovie.smoovie`.

The core module is replaced by a stub in ``sys.modules`` so these tests run without Holoscan/CuPy.
"""

import inspect
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kremetart.cli import app

runner = CliRunner()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {}

    def fake_smoovie(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(str(kwargs["output"]))

    stub = types.ModuleType("kremetart.core.smoovie")
    stub.smoovie = fake_smoovie
    monkeypatch.setitem(sys.modules, "kremetart.core.smoovie", stub)
    return calls


def test_help_lists_smoovie() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0 and "smoovie" in result.output


def test_smoovie_requires_input_and_output(captured: dict) -> None:
    result = runner.invoke(app, ["smoovie"])
    assert result.exit_code != 0
    assert captured == {}


def test_smoovie_forwards_every_option(captured: dict, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "smoovie",
            "--hdf-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            "--nside",
            "32",
            "--l1",
            "0.001",
            "--no-serve",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["hdf_dir"] == tmp_path and captured["output"] == tmp_path / "out.zarr"
    assert captured["nside"] == 32 and captured["l1"] == 0.001 and captured["serve"] is False


def test_cli_options_match_the_core_signature() -> None:
    """Every core parameter is reachable from the CLI, and the CLI adds none of its own."""
    import kremetart.cli as cli

    pytest.importorskip("holoscan")  # importing core.smoovie pulls in the Holoscan stack
    from kremetart.core.smoovie import smoovie as core_smoovie

    cli_params = set(inspect.signature(cli.smoovie).parameters)
    core_params = set(inspect.signature(core_smoovie).parameters)
    assert cli_params == core_params
