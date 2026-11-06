"""Copy ``README.md`` for the docs home page, with repo-relative links pointed at GitHub.

On GitHub a README links files by relative path (``./Dockerfile``, ``src/pkg/cli.py``); in the docs those
paths do not exist and MyST warns. Links stay relative only to ``docs/`` pages; every other relative link
or image becomes an absolute URL into the repository.
"""

import re
from pathlib import Path

# [text](target) and ![alt](target); the target ends at whitespace or ")".
_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)")
# Reference definitions, [label]: target (optionally <target>), at the start of a line.
_DEFINITION = re.compile(r"^( {0,3}\[[^\]]+\]:[ \t]*<?)([^>\s]+)", re.MULTILINE)
_ABSOLUTE = re.compile(r"^([a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)
_IMAGE = re.compile(r"\.(png|jpe?g|gif|svg|webp)$", re.IGNORECASE)


def _target(target: str, repo: str, branch: str, image: bool) -> str:
    if _ABSOLUTE.match(target):
        return target
    path = target.removeprefix("./")
    kind = "raw" if image else "blob"
    return f"{repo.rstrip('/')}/{kind}/{branch}/{path}"


def render(readme: Path, repo: str, branch: str = "main") -> str:
    """The README with relative links rewritten to ``repo`` (e.g. ``https://github.com/owner/name``)."""
    text = readme.read_text(encoding="utf-8")
    text = _LINK.sub(lambda m: m[1] + _target(m[2], repo, branch, m[1].startswith("!")), text)
    # A definition may serve an image or a link; judge by the file's extension.
    return _DEFINITION.sub(lambda m: m[1] + _target(m[2], repo, branch, bool(_IMAGE.search(m[2]))), text)


def write(readme: Path, path: Path, repo: str, branch: str = "main") -> None:
    """Write the rewritten README to ``path``."""
    path.write_text(render(readme, repo, branch), encoding="utf-8")
