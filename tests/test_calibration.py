"""Tests for per-antenna gain correction (:func:`kremetart.utils.calibration.correct_file_gains`)."""

import os

import numpy as np
import pytest

from kremetart.utils import partition_datatree
from kremetart.utils.calibration import correct_file_gains
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4


def test_correct_file_gains_real_data(hdf_paths):
    node = partition_datatree(read_hdf_as_msv4(hdf_paths[0]))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]

    vis_c, wgt_c = correct_file_gains(node, vis, wgt)

    assert vis_c.shape == vis.shape
    assert np.all(np.isfinite(vis_c)) and np.all(np.isfinite(wgt_c))
    # The correction must actually change non-trivial gains.
    assert not np.allclose(vis_c, vis)
    # Dead antennas (gain 0) -> zero-weight, zero-vis baselines (no inf/nan).
    gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values
    if np.any(gains == 0):
        assert np.any(wgt_c == 0)


@pytest.mark.skipif(
    os.environ.get("KREMETART_MS_ORACLE") != "1",
    reason="opt-in: cross-checks tart2ms calibration convention (set KREMETART_MS_ORACLE=1)",
)
def test_weighted_corrected_vis_matches_calibrated_ms(ref_hdf, ref_ms):
    """tart2ms writes the *weighted* corrected visibility into DATA: ``V_corr * |g_p g_q|**2``.

    ``correct_file_gains`` returns ``(V_corr, W_corr)`` with ``W_corr = |g_p g_q|**2``, and the
    imaging step forms ``W_corr * V_corr`` -- which equals ``V_raw * conj(g_p) * g_q``, exactly the
    calibrated DATA tart2ms writes. (The MS WEIGHT column is a separate constant nominal value, not
    this gain weight.) The small residual tail is the known ~0.3% ITRF baseline-position convention
    difference -- the same source as the ~cm UVW tolerance in ``test_rephasing.py``.
    """
    import xarray as xr

    node = partition_datatree(read_hdf_as_msv4(ref_hdf))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]
    vis_c, wgt_c = correct_file_gains(node, vis, wgt)

    ref = partition_datatree(xr.open_datatree(str(ref_ms), engine="xarray-ms:msv2"))
    ref_vis = np.asarray(ref.ds.VISIBILITY.values)[..., 0]

    # Compare only weighted baselines (dead antennas are zeroed on our side); the MSv4 reader and
    # tart2ms share baseline ordering (see test_rephasing.py), so this is an element-wise compare.
    mask = wgt > 0
    resid = np.abs((vis_c * wgt_c)[mask] - ref_vis[mask])
    assert np.median(resid) < 0.015
    assert np.percentile(resid, 95) < 0.05
