"""Tests for the host prepare-step (HDF sequence -> imaging-ready zarr). CPU, no GPU."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("xarray_ms")  # read_hdf_as_msv4 path uses MSv4 machinery
pytest.importorskip("astropy")

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_prepare_msv4_zarr_schema_and_shapes(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie import _partition
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    paths = _hdfs()[:1]
    main = _partition(read_hdf_as_msv4(paths[0])).ds
    n_time = int(main.time.size)
    n_bl = int(main.baseline_id.size)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(paths, out)
    ds = xr.open_zarr(str(out))

    assert set(ds["VISIBILITY"].dims) == {"time", "baseline", "frequency"}
    assert set(ds["WEIGHT"].dims) == {"time", "baseline", "frequency"}
    assert set(ds["B_ROT"].dims) == {"time", "baseline", "xyz"}
    assert ds["VISIBILITY"].shape == (n_time, n_bl, 1)
    assert ds["B_ROT"].shape == (n_time, n_bl, 3)
    assert np.iscomplexobj(ds["VISIBILITY"].values)
    np.testing.assert_allclose(ds.time.values, np.asarray(main.time.values))


def test_prepare_msv4_zarr_brot_matches_equatorial_baselines(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie import _partition
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr
    from kremetart.utils.healpix_dft import equatorial_baselines
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.rephasing import itrs_baselines

    paths = _hdfs()[:1]
    node = _partition(read_hdf_as_msv4(paths[0]))
    times = np.asarray(node.ds.time.values)
    bl = np.asarray(itrs_baselines(node, np))
    expected = equatorial_baselines(bl, times, xp=np)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(paths, out)
    ds = xr.open_zarr(str(out))
    np.testing.assert_allclose(ds["B_ROT"].values, expected, rtol=1e-12, atol=1e-12)


def test_prepare_msv4_zarr_correct_gains_matches_helper(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    paths = _hdfs()[:1]
    node = _partition(read_hdf_as_msv4(paths[0]))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]
    vis_c, wgt_c = _correct_file_gains(node, vis, wgt)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(paths, out, correct_gains=True)
    ds = xr.open_zarr(str(out))
    np.testing.assert_allclose(ds["VISIBILITY"].values, vis_c.astype(np.complex64), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds["WEIGHT"].values, wgt_c.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_prepare_msv4_zarr_nframes_caps(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie_prepare import prepare_msv4_zarr

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(_hdfs(), out, nframes=3)
    ds = xr.open_zarr(str(out))
    assert ds.time.size == 3
