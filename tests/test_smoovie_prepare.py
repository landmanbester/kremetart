"""Tests for the host prepare-step (HDF sequence -> imaging-ready zarr). CPU, no GPU.

Uses the shared ``hdf_paths`` fixture (``conftest.py``) for the bundled TART snapshots.
"""

import numpy as np
import pytest

from kremetart.utils import partition_datatree
from kremetart.utils.calibration import correct_file_gains


def test_prepare_msv4_zarr_schema_and_shapes(tmp_path, hdf_paths):
    import xarray as xr

    from kremetart.utils.read_tart_hdf import prepare_msv4_zarr, read_hdf_as_msv4

    main = partition_datatree(read_hdf_as_msv4(hdf_paths[0])).ds
    n_time = int(main.time.size)
    n_bl = int(main.baseline_id.size)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(hdf_paths[:1], out)
    ds = xr.open_zarr(str(out))

    assert set(ds["VISIBILITY"].dims) == {"time", "baseline", "frequency"}
    assert set(ds["WEIGHT"].dims) == {"time", "baseline", "frequency"}
    assert set(ds["B_ROT"].dims) == {"time", "baseline", "xyz"}
    assert ds["VISIBILITY"].shape == (n_time, n_bl, 1)
    assert ds["B_ROT"].shape == (n_time, n_bl, 3)
    assert np.iscomplexobj(ds["VISIBILITY"].values)
    np.testing.assert_allclose(ds.time.values, np.asarray(main.time.values))


def test_prepare_msv4_zarr_brot_matches_equatorial_baselines(tmp_path, hdf_paths):
    import xarray as xr

    from kremetart.utils.healpix_dft import equatorial_baselines
    from kremetart.utils.read_tart_hdf import prepare_msv4_zarr, read_hdf_as_msv4
    from kremetart.utils.rephasing import itrs_baselines

    node = partition_datatree(read_hdf_as_msv4(hdf_paths[0]))
    times = np.asarray(node.ds.time.values)
    bl = np.asarray(itrs_baselines(node, np))
    expected = equatorial_baselines(bl, times, xp=np)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(hdf_paths[:1], out)
    ds = xr.open_zarr(str(out))
    np.testing.assert_allclose(ds["B_ROT"].values, expected, rtol=1e-12, atol=1e-12)


def test_prepare_msv4_zarr_correct_gains_matches_helper(tmp_path, hdf_paths):
    import xarray as xr

    from kremetart.utils.read_tart_hdf import prepare_msv4_zarr, read_hdf_as_msv4

    node = partition_datatree(read_hdf_as_msv4(hdf_paths[0]))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]
    vis_c, wgt_c = correct_file_gains(node, vis, wgt, xp=np)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(hdf_paths[:1], out, correct_gains=True)
    ds = xr.open_zarr(str(out))
    np.testing.assert_allclose(ds["VISIBILITY"].values, vis_c.astype(np.complex64), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds["WEIGHT"].values, wgt_c.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_prepare_msv4_zarr_nframes_caps(tmp_path, hdf_paths):
    import xarray as xr

    from kremetart.utils.read_tart_hdf import prepare_msv4_zarr

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(hdf_paths, out, nframes=3)
    ds = xr.open_zarr(str(out))
    assert ds.time.size == 3


def test_prepare_msv4_zarr_nframes_crosses_file_boundary(tmp_path, hdf_paths):
    import xarray as xr

    from kremetart.utils.read_tart_hdf import prepare_msv4_zarr, read_hdf_as_msv4

    if len(hdf_paths) < 2:
        pytest.skip("need >=2 HDF files to cross a file boundary")
    n0 = int(partition_datatree(read_hdf_as_msv4(hdf_paths[0])).ds.time.size)
    cap = n0 + 2  # forces reading into the second file, then the early-break + final slice

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(hdf_paths, out, nframes=cap)
    ds = xr.open_zarr(str(out))
    assert ds.time.size == cap
