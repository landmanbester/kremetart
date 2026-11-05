"""Render the ``kremetart`` command reference as Markdown tables, from the Typer app's help strings.

The command's docstring gives the introduction, with its ``NAME: text`` paragraphs (the environment
variables) collected into a table; each option's help string, type and default fill the option tables.
"""

import enum
import re
from pathlib import Path

import typer

from kremetart.cli import app

_ENV_VAR = re.compile(r"^(KREMETART_[A-Z_]+): (.*)$", re.DOTALL)
_HIDDEN = {"install_completion", "show_completion"}


def _cell(text: str) -> str:
    """Make ``text`` safe for one Markdown table cell."""
    text = " ".join(text.split()).replace("|", "\\|")
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _option(param: typer.core.TyperOption) -> str:
    """The option's flag(s) and, for a valued option, its type."""
    if param.is_flag:
        return " / ".join(f"`{flag}`" for flag in param.opts + param.secondary_opts)
    choices = getattr(param.type, "choices", None)
    metavar = "{" + ",".join(choices) + "}" if choices else param.type.name.upper()
    return f"`{param.opts[0]} {metavar}`"


def _default(param: typer.core.TyperOption) -> str:
    default = param.default
    if param.is_flag:
        return "on" if default else "off"
    if default is None:
        return ""
    if isinstance(default, enum.Enum):
        default = default.value
    return f"`{default}`"


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]


def render() -> str:
    """The command page as Markdown."""
    command = typer.main.get_command(app)
    paragraphs = [p.strip() for p in (command.help or "").split("\n\n") if p.strip()]
    environment = [m.groups() for p in paragraphs if (m := _ENV_VAR.match(p))]
    intro = [p for p in paragraphs if not _ENV_VAR.match(p)]

    params = [p for p in command.params if p.name not in _HIDDEN]
    required = [p for p in params if p.required]
    optional = [p for p in params if not p.required]

    lines = ["# The `kremetart` command", "", *(line for p in intro for line in (p, "")), "## Usage", ""]
    lines += ["```console", "kremetart " + " ".join(f"{p.opts[0]} <{p.name}>" for p in required) + " [OPTIONS]"]
    lines += ["```", "", "## Required", ""]
    lines += _table(("Option", "Description"), [(_option(p), _cell(p.help or "")) for p in required])
    lines += ["## Options", ""]
    lines += _table(
        ("Option", "Default", "Description"), [(_option(p), _default(p), _cell(p.help or "")) for p in optional]
    )
    if environment:
        lines += ["## Environment", ""]
        lines += _table(("Variable", "Description"), [(f"`{name}`", _cell(text)) for name, text in environment])
    return "\n".join(lines)


def write(path: Path) -> None:
    """Write the page to ``path``."""
    path.write_text(render(), encoding="utf-8")
