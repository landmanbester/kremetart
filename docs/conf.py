"""Sphinx configuration for the kremetart documentation.

Every page is generated: the home page from ``README.md``, the command reference from the Typer app's
help strings, the Python reference from docstrings, and the glossary from ``GLOSSARY.md``. Pages under
``docs/`` only arrange them.
"""

import os
import sys
from importlib.metadata import version as package_version
from pathlib import Path

DOCS = Path(__file__).parent

sys.path.insert(0, str(DOCS))
import command_page  # noqa: E402  (lives next to this file)
import glossary_page  # noqa: E402
import readme_page  # noqa: E402

from kremetart.cli import app  # noqa: E402

readme_page.write(DOCS.parent / "README.md", DOCS / "readme.md", "https://github.com/landmanbester/kremetart")
command_page.write(app, "kremetart", DOCS / "reference" / "command.md", env_prefix="KREMETART_")
glossary_page.write(DOCS.parent / "GLOSSARY.md", DOCS / "glossary.md")

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
    "field_tables",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
# The docs are a quick read for people. The ADRs, design notes and agent notes under docs/ are the
# agents' record and stay out of the site.
exclude_patterns = ["_build", "readme.md", "adr", "agents", "design", "tex"]
templates_path = ["_templates"]

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True, "ignore-module-all": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Show defaults as written (``xp=np``), not as their repr (a module path on the build machine).
autodoc_preserve_defaults = True
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
html_static_path = ["_static"]
html_css_files = ["tables.css"]
# Furo shows API names in red; dark green and purple are calmer.
html_theme_options = {
    "light_css_variables": {"color-api-pre-name": "#2e6b3a", "color-api-name": "#6a3d9a"},
    "dark_css_variables": {"color-api-pre-name": "#8fcf9b", "color-api-name": "#c9a7ef"},
}
# Read the Docs sets the canonical URL, so llms.txt links resolve to the published version.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
