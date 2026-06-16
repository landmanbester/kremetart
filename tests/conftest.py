"""pytest session setup: fetch test data from Google Drive via gdown.

The contents of ``tests/data/`` (TART HDF snapshots and the tart2ms Measurement Sets) are
gitignored and not shipped with the repo, so CI and fresh checkouts start without them. This module
downloads whatever is missing once per session. Measurement Sets are *directories*, so they are
distributed as ``.zip`` archives and extracted after download.

Set ``KREMETART_OFFLINE=1`` to skip all downloads (air-gapped CI). Tests that still lack their data
skip themselves via their own existence checks, so a failed or skipped download never errors the
session.
"""

import os
import zipfile
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"

# --- Single files: destination filename (under tests/data) -> Google Drive file id. -------------
# The full movie sequence is nine HDF snapshots (08:11:43 -> 08:19:55). Add the remaining eight ids
# here, or distribute them as a single zip via _ARCHIVES below.
#   https://drive.google.com/file/d/125SzcgEvou-IpZrQh0712wdOMARu6-x7/view
_FILES: dict[str, str] = {
    "vis_2026-06-09_08_11_43.476804.hdf": "125SzcgEvou-IpZrQh0712wdOMARu6-x7",
    # "vis_2026-06-09_08_12_44.971020.hdf": "<id>",
    # "vis_2026-06-09_08_13_46.477614.hdf": "<id>",
    # "vis_2026-06-09_08_14_47.979545.hdf": "<id>",
    # "vis_2026-06-09_08_15_49.472494.hdf": "<id>",
    # "vis_2026-06-09_08_16_50.973156.hdf": "<id>",
    # "vis_2026-06-09_08_17_52.481837.hdf": "<id>",
    # "vis_2026-06-09_08_18_53.972910.hdf": "<id>",
    # "vis_2026-06-09_08_19_55.473550.hdf": "<id>",
}

# --- Zip archives extracted into tests/data: a marker path (relative to tests/data) that exists
# only after extraction -> Google Drive file id of the .zip. Use these for the Measurement Sets
# (directories). A single whole-data zip also works: key it on any file it contains. --------------
_ARCHIVES: dict[str, str] = {
    # "vis_2026-06-09_08_11_43.476804.ms": "<gdrive-zip-id>",       # cal MS (msv4 spec test)
    # "vis_2026-06-09_08_11_43.476804_nocal.ms": "<gdrive-zip-id>",  # nocal MS (rephasing, accuracy)
}


def _gdown(file_id: str, dest: Path) -> None:
    import gdown

    gdown.download(id=file_id, output=str(dest), quiet=False)


def _ensure_files() -> None:
    for name, file_id in _FILES.items():
        dest = _DATA / name
        if dest.exists():
            continue
        print(f"[kremetart tests] downloading {name}")
        _gdown(file_id, dest)


def _ensure_archives() -> None:
    for marker, file_id in _ARCHIVES.items():
        if (_DATA / marker).exists():
            continue
        zip_path = _DATA / f"{Path(marker).name}.zip"
        print(f"[kremetart tests] downloading + extracting {marker}")
        _gdown(file_id, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_DATA)
        zip_path.unlink()


def pytest_sessionstart(session) -> None:
    """Best-effort populate tests/data once per session."""
    if os.environ.get("KREMETART_OFFLINE") == "1":
        print("[kremetart tests] KREMETART_OFFLINE=1 - skipping test-data download")
        return
    _DATA.mkdir(parents=True, exist_ok=True)
    try:
        _ensure_files()
        _ensure_archives()
    except Exception as exc:  # tests carry their own skip-if-missing guards
        print(f"[kremetart tests] test-data download failed ({exc}); dependent tests will skip")
