"""Render a Typer app's command reference as Markdown tables, from its help strings.

Works for a single-command app and for groups (subcommands, nested groups). Each command's docstring
gives its introduction; its ``NAME: text`` paragraphs (environment variables, matched by ``ENV_PREFIX``)
are collected into a table; each option's and argument's help string, type and default fill the tables.

Typer bundles its own copy of Click (``typer._click``), so sphinx-click rejects Typer commands and
``isinstance`` checks against the ``click`` package fail: this reads the commands directly and tells
parameters and groups apart by duck typing.
"""

import enum
import re
from pathlib import Path
from typing import Any

import typer

HIDDEN = {"install_completion", "show_completion", "help"}


def _is_group(command: Any) -> bool:
    return hasattr(command, "list_commands")


def _cell(text: str) -> str:
    """Make ``text`` safe for one Markdown table cell (MyST reads ``<…>`` as HTML)."""
    text = " ".join(text.split()).replace("|", "\\|")
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _metavar(param: Any) -> str:
    if param.metavar:  # set metavar= on options with a custom parser, or the parser's name shows
        return param.metavar
    choices = getattr(param.type, "choices", None)
    if choices:
        return "{" + ",".join(str(getattr(c, "value", c)) for c in choices) + "}"
    return param.type.name.upper()


def _option(param: Any) -> str:
    """The option's flag(s) and, for a valued option, its type."""
    if param.is_flag:
        return " / ".join(f"`{flag}`" for flag in param.opts + param.secondary_opts)
    return f"`{param.opts[0]} {_metavar(param)}`"


def _default(param: Any) -> str:
    default = param.default
    if getattr(param, "is_flag", False):
        return "on" if default else "off"
    if default is None or callable(default):
        return ""
    if isinstance(default, enum.Enum):
        default = default.value
    if isinstance(default, (list, tuple)):
        if not default:
            return ""
        default = ", ".join(str(d) for d in default)
    return f"`{default}`"


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    if not rows:
        return []
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]


def _paragraphs(command: Any, env_var: re.Pattern[str]) -> tuple[list[str], list[tuple[str, str]]]:
    # Typer strips the docstring's "\b" markers into the help; split on blank lines.
    paragraphs = [p.strip() for p in (command.help or "").split("\n\n") if p.strip()]
    environment = [m.groups() for p in paragraphs if (m := env_var.match(p))]
    return [p for p in paragraphs if not env_var.match(p)], environment


def _command(command: Any, path: str, level: int, env_var: re.Pattern[str]) -> list[str]:
    """One command's section: intro, usage, arguments, required options, options, environment."""
    intro, environment = _paragraphs(command, env_var)
    params = [p for p in command.params if p.name not in HIDDEN and not getattr(p, "hidden", False)]
    arguments = [p for p in params if p.param_type_name == "argument"]
    options = [p for p in params if p.param_type_name == "option"]
    required = [p for p in options if p.required]
    optional = [p for p in options if not p.required]

    hashes = "#" * level
    sub = "#" * (level + 1)
    lines = [f"{hashes} `{path}`", "", *(line for p in intro for line in (p, ""))]
    usage = [path, *(f"<{p.name}>" for p in arguments), *(f"{p.opts[0]} <{p.name}>" for p in required)]
    if optional:
        usage.append("[OPTIONS]")
    lines += [f"{sub} Usage", "", "```console", " ".join(usage), "```", ""]

    if arguments:
        rows = [(f"`{p.name}`", _metavar(p), _cell(getattr(p, "help", None) or "")) for p in arguments]
        lines += [f"{sub} Arguments", "", *_table(("Argument", "Type", "Description"), rows)]
    if required:
        rows = [(_option(p), _cell(p.help or "")) for p in required]
        lines += [f"{sub} Required", "", *_table(("Option", "Description"), rows)]
    if optional:
        rows = [(_option(p), _default(p), _cell(p.help or "")) for p in optional]
        lines += [f"{sub} Options", "", *_table(("Option", "Default", "Description"), rows)]
    if environment:
        rows = [(f"`{name}`", _cell(text)) for name, text in environment]
        lines += [f"{sub} Environment", "", *_table(("Variable", "Description"), rows)]
    return lines


def _group(group: Any, path: str, level: int, env_var: re.Pattern[str]) -> list[str]:
    """A group's intro, a table of its commands, then one section per command (recursing into subgroups)."""
    intro, environment = _paragraphs(group, env_var)
    ctx = group.context_class(group, info_name=path)
    names = [n for n in group.list_commands(ctx) if not group.get_command(ctx, n).hidden]
    hashes = "#" * level
    lines = [f"{hashes} `{path}`", "", *(line for p in intro for line in (p, ""))]
    rows = [(f"`{path} {n}`", _cell(group.get_command(ctx, n).get_short_help_str(limit=200))) for n in names]
    lines += _table(("Command", "Description"), rows)
    if environment:
        rows_env = [(f"`{name}`", _cell(text)) for name, text in environment]
        lines += [f"{'#' * (level + 1)} Environment", "", *_table(("Variable", "Description"), rows_env)]
    for name in names:
        child = group.get_command(ctx, name)
        render_child = _group if _is_group(child) else _command
        lines += render_child(child, f"{path} {name}", level + 1, env_var)
    return lines


def render(app: typer.Typer, prog: str, env_prefix: str = "") -> str:
    """The command page as Markdown, titled with ``prog`` (the installed command name).

    ``env_prefix`` (e.g. ``"KREMETART_"``) marks docstring paragraphs of the form ``PREFIX_NAME: text``
    as environment variables; leave it empty to keep every paragraph as prose.
    """
    env_var = re.compile(rf"^({re.escape(env_prefix)}[A-Z_]+): (.*)$", re.DOTALL) if env_prefix else re.compile("$^")
    command = typer.main.get_command(app)
    render_root = _group if _is_group(command) else _command
    lines = render_root(command, prog, 1, env_var)
    lines[0] = f"# The `{prog}` command" if not _is_group(command) else f"# `{prog}` commands"
    return "\n".join(lines).rstrip() + "\n"


def write(app: typer.Typer, prog: str, path: Path, env_prefix: str = "") -> None:
    """Write the page for ``app`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(app, prog, env_prefix), encoding="utf-8")
