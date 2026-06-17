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

import pytest

_DATA = Path(__file__).resolve().parent / "data"

# A single zip of the entire tests/data directory, files at top level (no nesting). Extracted into
# tests/data when missing.
#   https://drive.google.com/file/d/1vENbb8wQatHCNwO6LODXgPDwchcHzo7m/view
_BUNDLE_ID = "1vENbb8wQatHCNwO6LODXgPDwchcHzo7m"

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


# --- shared test-data fixtures ---------------------------------------------------------------
# These point tests at the bundled tests/data instead of each test redefining a `_DATA` constant
# and an `_hdfs()` helper. The catalogue fixtures in particular let satellite/overlay tests reuse
# the pre-downloaded tests/data/catalog.zarr so they never query the (slow) TART catalogue API.


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The ``tests/data`` directory (TART HDF snapshots, Measurement Sets, bundled catalogue)."""
    return _DATA


@pytest.fixture(scope="session")
def hdf_paths(data_dir: Path) -> list[Path]:
    """All bundled TART HDF snapshots, sorted; skips the test if none are present."""
    paths = sorted(data_dir.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present in tests/data")
    return paths


@pytest.fixture(scope="session")
def hdf_dir(hdf_paths: list[Path]) -> Path:
    """The ``tests/data`` directory, guaranteed to contain HDFs (else the test skips).

    For ``smoovie(hdf_dir=...)`` calls, which glob the directory themselves.
    """
    return hdf_paths[0].parent


@pytest.fixture(scope="session")
def catalog_cache(data_dir: Path) -> str:
    """Path to the bundled satellite-catalogue cache zarr; skips if absent.

    Pass this as ``cache_path`` (:func:`kremetart.utils.satellites.satellite_tracks`) or
    ``catalog_cache`` (:func:`kremetart.core.smoovie.smoovie`) so tests reuse the pre-downloaded
    catalogue instead of querying the TART API.
    """
    cat = data_dir / "catalog.zarr"
    if not cat.exists():
        pytest.skip("bundled tests/data/catalog.zarr not present")
    return str(cat)


@pytest.fixture(scope="session")
def catalog_elevation(catalog_cache: str) -> float:
    """Elevation cutoff (deg) the bundled catalogue was built at.

    Tests must pass this as the cutoff; any other value is a cache miss that falls through to the API.
    """
    import xarray as xr

    return float(xr.open_zarr(catalog_cache).attrs["elevation_deg"])
