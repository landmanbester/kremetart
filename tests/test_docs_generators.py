"""The docs generators in ``docs/``: the README copy and the command page.

They run inside the Sphinx build; these tests pin the cases kremetart's own sources do not exercise
(reference-style links, command groups, positional arguments), which the shared autodocs copies rely on.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).parents[1] / "docs"))
import command_page  # noqa: E402  (lives in docs/, outside the package)
import readme_page  # noqa: E402

REPO = "https://github.com/owner/name"


def test_readme_links_point_at_the_repository(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "See [the CLI](./src/pkg/cli.py), [docs](https://example.org), [up](#usage) and ![logo](img/logo.png).\n"
        "Also [source][src] and ![diagram][fig].\n\n"
        "[src]: src/pkg/core.py\n"
        "[fig]: <docs/fig.svg>\n"
        "[web]: https://example.org/page\n",
        encoding="utf-8",
    )
    text = readme_page.render(readme, REPO)
    assert f"[the CLI]({REPO}/blob/main/src/pkg/cli.py)" in text
    assert f"![logo]({REPO}/raw/main/img/logo.png)" in text
    assert "[docs](https://example.org)" in text
    assert "[up](#usage)" in text
    assert f"[src]: {REPO}/blob/main/src/pkg/core.py" in text
    assert f"[fig]: <{REPO}/raw/main/docs/fig.svg>" in text
    assert "[web]: https://example.org/page" in text


def _group_app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def main(config: Annotated[Path, typer.Option(help="Config file.")]) -> None:
        """Tools."""

    @app.command()
    def run(
        target: Annotated[str, typer.Argument(help="What to run.")],
        extras: Annotated[list[str] | None, typer.Argument(help="Extra inputs.")] = None,
        level: Annotated[int, typer.Option(help="Level.")] = 1,
    ) -> None:
        """Run a target."""

    @app.command()
    def show() -> None:
        """Show state."""

    return app


def test_command_page_documents_group_options_and_arguments() -> None:
    page = command_page.render(_group_app(), "tool")
    assert "| `tool run` | Run a target. |" in page
    # The group's own (required) option appears before its commands' sections.
    assert "tool --config <config> COMMAND [ARGS]..." in page
    assert page.index("`--config PATH`") < page.index("## `tool run`")
    # Required, optional and repeated arguments read differently in usage.
    assert "tool run [OPTIONS] <target> [<extras>...]" in page
    assert "| `level` |" not in page and "`--level INTEGER` | `1` |" in page
