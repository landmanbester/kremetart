"""Structural guards for the hip-cargo cli <-> core <-> cab invariant.

This is the executable companion to ``tests/test_roundtrip.py``. The round-trip test guards the
``cli/*.py`` <-> ``cabs/*.yml`` half; this module guards the package *shape* that the whole pattern
rests on (see ``.claude/rules/architecture.md`` Section 1):

  * ``cli/``, ``core/`` and ``cabs/`` are in strict 1:1:1 correspondence -- one ``<name>.py`` /
    ``<name>.py`` / ``<name>.yml`` per command, sharing the command's name. ``core/`` holds
    *commands only*; shared helper logic belongs in ``utils/`` and Holoscan operators/apps in
    ``operators/``. A ``core/<name>.py`` with no ``cli/<name>.py`` mirror is a bug.
  * ``core.<cmd>``'s signature mirrors ``cli.<cmd>``: its parameters equal the cli wrapper's
    parameters minus the ``StimelaMeta(skip=True)`` flags. A core-only parameter is dead surface
    (unreachable from the CLI or a Stimela cab) and signals that runtime branching should be
    auto-detected internally rather than exposed as an argument.

Both checks are filesystem/AST only -- no package import -- so they run in the lightweight install
and never need the heavy ``[full]`` deps (or a GPU).
"""

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "src" / "kremetart"

# The two parameters every cli wrapper auto-grows via hip-cargo's generate-function, both decorated
# StimelaMeta(skip=True). They live on the cli wrapper but never on the core implementation.
_SKIP_PARAMS = {"backend", "always_pull_images"}


def _command_names(subdir: str, suffix: str) -> set[str]:
    """Stems of ``src/kremetart/<subdir>/*<suffix>`` files, excluding dunder modules."""
    return {p.stem for p in (_PKG / subdir).glob(f"*{suffix}") if not p.name.startswith("__")}


def _param_names(py_path: Path, func_name: str) -> set[str]:
    """Parameter names of the top-level ``def <func_name>`` in ``py_path`` (AST; no import).

    Excludes ``*args`` / ``**kwargs``. Decorators do not affect the parsed signature.
    """
    tree = ast.parse(py_path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            a = node.args
            return {arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    raise AssertionError(f"no top-level function {func_name!r} in {py_path}")


def test_cli_core_cab_are_one_per_command():
    """Every command has exactly cli/<name>.py, core/<name>.py and cabs/<name>.yml -- no orphans.

    Catches helper/implementation modules wrongly placed under ``core/`` (they belong in ``utils/``
    or ``operators/``), and any cli/core/cab that lost its counterpart.
    """
    cli = _command_names("cli", ".py")
    core = _command_names("core", ".py")
    cabs = _command_names("cabs", ".yml")

    assert core == cli, (
        "core/ must hold exactly one module per cli command (see architecture.md Section 1). "
        f"core-only (move to utils/ or operators/): {sorted(core - cli)}; "
        f"cli-only (missing core): {sorted(cli - core)}"
    )
    assert cabs == cli, (
        "every command needs a generated cab (and vice versa). "
        f"cab-only: {sorted(cabs - cli)}; cli-only (missing cab): {sorted(cli - cabs)}"
    )


def test_core_signature_mirrors_cli():
    """``core.<cmd>`` params == ``cli.<cmd>`` params minus the skip flags -- no core-only params."""
    for cmd in sorted(_command_names("cli", ".py") & _command_names("core", ".py")):
        cli_params = _param_names(_PKG / "cli" / f"{cmd}.py", cmd)
        core_params = _param_names(_PKG / "core" / f"{cmd}.py", cmd)
        expected = cli_params - _SKIP_PARAMS
        assert core_params == expected, (
            f"core.{cmd} signature must mirror cli.{cmd} (minus {sorted(_SKIP_PARAMS)}); "
            "see architecture.md Section 1. "
            f"core-only params (drop these or expose them on the cli wrapper): "
            f"{sorted(core_params - expected)}; "
            f"missing from core: {sorted(expected - core_params)}"
        )
