"""The ``kremetart`` CLI: option parsing and hand-off to :func:`kremetart.core.kremetart.kremetart`.

The core module is replaced by a stub in ``sys.modules`` and its signature is read from source, so these
tests run without Holoscan/CuPy or a GPU.
"""

import ast
import inspect
import sys
import types
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

import kremetart.cli as cli
from kremetart.cli import app
from kremetart.core.runner import RUNNERS

runner = CliRunner()
CORE_MODULE = Path(cli.__file__).parent / "core" / "kremetart.py"


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {}

    def fake_kremetart(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(str(kwargs["output"]))

    stub = types.ModuleType("kremetart.core.kremetart")
    stub.kremetart = fake_kremetart
    monkeypatch.setitem(sys.modules, "kremetart.core.kremetart", stub)
    return calls


def test_bare_command_prints_help(captured: dict) -> None:
    """``kremetart`` is a single command: with no arguments it shows its options and runs nothing."""
    result = runner.invoke(app, [])
    assert "--hdf-dir" in click.unstyle(result.output)  # CI forces colour, which splits the flag
    assert captured == {}


def test_requires_input_and_output(captured: dict) -> None:
    result = runner.invoke(app, ["--nside", "8"])
    assert result.exit_code != 0
    assert captured == {}


def test_forwards_every_option(captured: dict, tmp_path: Path) -> None:
    """Every option, set to a distinct non-default value, reaches core unchanged."""
    expected = {
        "hdf_dir": tmp_path / "hdf",
        "output": tmp_path / "out.zarr",
        "nside": 32,
        "phase_ra_deg": 12.5,
        "phase_dec_deg": -30.25,
        "correct_gains": True,
        "overlay_catalog": True,
        "catalog_elevation_deg": 30.0,
        "catalog_cache": str(tmp_path / "catalog.zarr"),
        "profile": True,
        "iwp_sigma": 0.002,
        "iwp_noise": 0.03,
        "apply_beam": False,
        "ground_plane_diameter": 0.25,
        "l2": 0.05,
        "l1": 0.001,
        "overwrite": True,
        "nframes": 7,
        "serve": False,
        "port": 9090,
        "open_browser": True,
        "runner": "cyberether",
    }
    args: list[str] = []
    for name, value in expected.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            args += [flag, str(value)]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert captured == expected
    defaults = {
        name: p.default for name, p in inspect.signature(cli.kremetart).parameters.items() if p.default is not p.empty
    }
    assert all(expected[name] != default for name, default in defaults.items()), "use non-default values"


def _core_parameters() -> list[str]:
    """Parameter names of ``core.kremetart.kremetart``, read from source so no GPU stack is imported."""
    tree = ast.parse(CORE_MODULE.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kremetart")
    return [a.arg for a in func.args.posonlyargs + func.args.args + func.args.kwonlyargs]


def test_cli_runner_choices_are_the_core_runners() -> None:
    assert tuple(name.value for name in cli.RunnerName) == RUNNERS


def test_cli_rejects_an_unknown_runner(captured: dict, tmp_path: Path) -> None:
    result = runner.invoke(app, ["--hdf-dir", str(tmp_path), "--output", str(tmp_path / "o"), "--runner", "gnuradio"])
    assert result.exit_code != 0 and captured == {}


def test_cli_options_match_the_core_signature() -> None:
    """Every core parameter is reachable from the CLI, and the CLI adds none of its own."""
    assert list(inspect.signature(cli.kremetart).parameters) == _core_parameters()
