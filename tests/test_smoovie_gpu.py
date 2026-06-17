"""GPU-gated tests for the smoovie Holoscan app and its operators.

Skipped unless a CUDA device and the cupy/holoscan/healpy stack are present (CPU CI skips all of
this). If a test segfaults during holoscan import, raise the stack limit first: `ulimit -s 32768`.
Test data comes from the shared ``hdf_paths`` / ``hdf_dir`` fixtures (``conftest.py``).
"""

import numpy as np
import pytest


def _gpu():
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        import healpy  # noqa: F401
        import holoscan  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="requires a CUDA device + cupy/holoscan/healpy")


def test_gpu_operators_import():
    from kremetart.operators.dft_healpix import HealpixDFTOperator
    from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator

    assert HealpixDFTOperator and HealpixZarrReaderOperator and HealpixWriterOperator


def test_image_via_app_end_to_end(hdf_paths):
    from kremetart.operators.smoovie_app import image_via_app

    nside = 8
    npix = 12 * nside * nside
    maps, stamps = image_via_app(hdf_paths[:1], nside, correct_gains=True, nframes=3)
    assert len(maps) == len(stamps) == 3
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
    assert "UTC" in stamps[0]


def test_gpu_app_matches_cpu_frame_dirty_maps(hdf_paths):
    """Behaviour preservation: GPU-app dirty maps equal the CPU frame_dirty_maps baseline."""
    from kremetart.core.smoovie import frame_dirty_maps
    from kremetart.operators.smoovie_app import image_via_app

    paths = hdf_paths[:1]
    nside = 8
    cpu_maps, _, _ = frame_dirty_maps(paths, nside, correct_gains=True, nframes=3)
    gpu_maps, _ = image_via_app(paths, nside, correct_gains=True, nframes=3)

    assert len(gpu_maps) == len(cpu_maps) == 3
    for c, g in zip(cpu_maps, gpu_maps):
        np.testing.assert_allclose(np.asarray(g), np.asarray(c), rtol=1e-4, atol=1e-5)


def test_smoovie_produces_movie_gpu(tmp_path, hdf_dir):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from kremetart.core.smoovie import smoovie

    out = tmp_path / "movie.mp4"
    smoovie(hdf_dir=hdf_dir, movie=out, nside=16, fps=2, nframes=4)
    assert out.exists() and out.stat().st_size > 0
