"""Sphinx configuration for the kremetart documentation.

Every page is generated: the command reference from the Typer app's help strings, the Python reference
from docstrings, and the glossary, decision records and design notes from the Markdown files agents
maintain (``GLOSSARY.md``, ``docs/adr/``, ``docs/design/``). Pages under ``docs/`` only arrange them.
"""

import os
import sys
from importlib.metadata import version as package_version
from pathlib import Path

DOCS = Path(__file__).parent

sys.path.insert(0, str(DOCS))
import command_page  # noqa: E402  (lives next to this file)

command_page.write(DOCS / "reference" / "command.md")

project = "kremetart"
author = "Landman Bester"
release = package_version("kremetart")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_llms_txt",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
# docs/agents is for agents only; docs/tex is the LaTeX design note.
exclude_patterns = ["_build", "agents", "tex", "design/README.md"]

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True, "ignore-module-all": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# The CUDA stack is Linux-only and about 1 GB; the docs build does not install it.
autodoc_mock_imports = ["holoscan", "cupy", "cupyx"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

html_theme = "furo"
html_title = f"kremetart {release}"
html_show_sphinx = False
# Read the Docs sets the canonical URL, so llms.txt links resolve to the published version.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
