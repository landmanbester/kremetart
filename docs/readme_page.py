"""Copy ``README.md`` for the docs home page, with repo-relative links pointed at GitHub.

On GitHub a README links files by relative path (``./Dockerfile``, ``src/pkg/cli.py``); in the docs those
paths do not exist and MyST warns. Links stay relative only to ``docs/`` pages; every other relative link
or image becomes an absolute URL into the repository.
"""

import re
from pathlib import Path

# [text](target) and ![alt](target); the target ends at whitespace or ")".
_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)")
_ABSOLUTE = re.compile(r"^([a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)


def _target(target: str, repo: str, branch: str, image: bool) -> str:
    if _ABSOLUTE.match(target):
        return target
    path = target.removeprefix("./")
    kind = "raw" if image else "blob"
    return f"{repo.rstrip('/')}/{kind}/{branch}/{path}"


def render(readme: Path, repo: str, branch: str = "main") -> str:
    """The README with relative links rewritten to ``repo`` (e.g. ``https://github.com/owner/name``)."""
    text = readme.read_text(encoding="utf-8")
    return _LINK.sub(lambda m: m[1] + _target(m[2], repo, branch, m[1].startswith("!")), text)


def write(readme: Path, path: Path, repo: str, branch: str = "main") -> None:
    """Write the rewritten README to ``path``."""
    path.write_text(render(readme, repo, branch), encoding="utf-8")
