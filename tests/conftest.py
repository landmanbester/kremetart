"""pytest session setup: fetch test data from Google Drive via gdown.

The contents of ``tests/data/`` (the nine TART HDF snapshots and the tart2ms Measurement Sets) are
gitignored and not shipped with the repo, so CI and fresh checkouts start without them. This module
downloads a single zip of the whole data directory and extracts it once per session when the data is
missing. (Measurement Sets are directories, hence the zip rather than per-file downloads.)

Set ``KREMETART_OFFLINE=1`` to skip the download (air-gapped CI). Tests carry their own
skip-if-missing guards, so a skipped or failed download never errors the session.
"""

import os
import zipfile
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"

# A single zip of the entire tests/data directory, files at top level (no nesting). Extracted into
# tests/data when missing.
#   https://drive.google.com/file/d/1EqkKt5KYFRLNFsIlbTsGU6zsnWtK-HU4/view
_BUNDLE_ID = "1EqkKt5KYFRLNFsIlbTsGU6zsnWtK-HU4"

# A path that exists only after extraction; its presence means the data is already in place.
_BUNDLE_MARKER = "vis_2026-06-09_08_11_43.476804_nocal.ms"


def pytest_sessionstart(session) -> None:
    """Best-effort populate tests/data once per session."""
    if os.environ.get("KREMETART_OFFLINE") == "1":
        print("[kremetart tests] KREMETART_OFFLINE=1 - skipping test-data download")
        return
    if (_DATA / _BUNDLE_MARKER).exists():
        return  # data already present (local checkout or a prior run)

    import gdown

    _DATA.mkdir(parents=True, exist_ok=True)
    zip_path = _DATA / "_tart_test_data.zip"
    try:
        print("[kremetart tests] downloading test-data bundle from Google Drive ...")
        gdown.download(id=_BUNDLE_ID, output=str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_DATA)
        print(f"[kremetart tests] test data ready in {_DATA}")
    except Exception as exc:  # tests carry their own skip-if-missing guards
        print(f"[kremetart tests] test-data download failed ({exc}); dependent tests will skip")
    finally:
        zip_path.unlink(missing_ok=True)
