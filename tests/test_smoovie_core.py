"""Tests for the smoovie core (HEALPix movie generation)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("xarray_ms")  # read_hdf_as_msv4 path uses MSv4 machinery

from kremetart.core.smoovie import frame_dirty_maps  # noqa: E402

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_frame_dirty_maps_one_finite_map_per_hdf():
    paths = _hdfs()[:3]
    nside = 32
    maps, stamps, pix = frame_dirty_maps(paths, nside)
    npix = 12 * nside * nside
    assert len(maps) == len(stamps) == 3
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
    assert pix.shape == (npix, 3)
    assert "UTC" in stamps[0]


def test_smoovie_produces_movie(tmp_path):
    pytest.importorskip("matplotlib")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from kremetart.core.smoovie import smoovie

    _hdfs()  # skip if reference data absent
    out = tmp_path / "movie.mp4"
    smoovie(hdf_dir=_DATA, movie=out, nside=32, fps=2)
    assert out.exists() and out.stat().st_size > 0
