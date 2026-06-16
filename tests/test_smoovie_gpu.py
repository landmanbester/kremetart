"""GPU-gated tests for the smoovie Holoscan app and its operators.

Skipped unless a CUDA device and the cupy/holoscan/healpy stack are present (CPU CI skips all of
this). If a test segfaults during holoscan import, raise the stack limit first: `ulimit -s 32768`.
"""

from pathlib import Path

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

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_gpu_operators_import():
    from kremetart.operators.dft_healpix import HealpixDFTOperator
    from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator

    assert HealpixDFTOperator and HealpixZarrReaderOperator and HealpixWriterOperator
