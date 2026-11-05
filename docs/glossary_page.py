"""Render ``GLOSSARY.md`` as the docs glossary: one table of terms and meanings per section.

``GLOSSARY.md`` is written for agents as well as people: each entry is ``**Term**:`` followed by its
definition and an ``_Avoid_:`` line of terms not to use. The page keeps the sections, terms and
definitions and leaves out the ``_Avoid_`` lines.
"""

import re
from pathlib import Path

_ENTRY = re.compile(r"^\*\*(.+?)\*\*:\s*\n(.*)$", re.DOTALL)


def _cell(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.startswith("_Avoid_")]
    return " ".join(" ".join(lines).split()).replace("|", "\\|")


def render(glossary: Path) -> str:
    """The glossary page as Markdown."""
    text = glossary.read_text(encoding="utf-8")
    intro, _, body = text.partition("\n## ")
    lines = ["# Glossary", ""]
    lines += [line for line in intro.splitlines()[1:]]
    for section in ("## " + body).split("\n### ")[1:]:
        heading, _, entries = section.partition("\n")
        entries = entries.split("\n## ")[0]  # stop at the next top-level section
        rows = [m.groups() for block in entries.strip().split("\n\n") if (m := _ENTRY.match(block.strip()))]
        if rows:
            lines += ["", f"## {heading.strip()}", "", "| Term | Meaning |", "|---|---|"]
            lines += [f"| **{term}** | {_cell(meaning)} |" for term, meaning in rows]
    return "\n".join(lines) + "\n"


def write(glossary: Path, path: Path) -> None:
    """Write the page for ``glossary`` to ``path``."""
    path.write_text(render(glossary), encoding="utf-8")
