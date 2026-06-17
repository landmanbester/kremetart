"""L1: kremetart antenna positions match an independent PROJ truth; tart2ms does not."""

import numpy as np
import pytest
import xarray as xr

from kremetart.utils import partition_datatree
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
from tests.accuracy_helpers import (
    antenna_ecef,
    antenna_enu_and_site,
    baseline_index_arrays,
    baselines_from_positions,
    enu_to_ecef_truth,
)


@pytest.fixture(scope="module")
def positions(ref_hdf, ref_ms_nocal):
    ours_part = partition_datatree(read_hdf_as_msv4(ref_hdf))
    enu, lat, lon, alt = antenna_enu_and_site(ours_part)
    truth = enu_to_ecef_truth(enu, lat, lon, alt)
    ours = antenna_ecef(ours_part["antenna_xds"].to_dataset(inherit=False))
    ms_part = partition_datatree(xr.open_datatree(str(ref_ms_nocal), engine="xarray-ms:msv2"))
    tart2ms = antenna_ecef(ms_part["antenna_xds"].to_dataset(inherit=False))
    a1, a2 = baseline_index_arrays(ours_part)
    return dict(truth=truth, ours=ours, tart2ms=tart2ms, a1=a1, a2=a2)


def test_antenna_index_alignment(positions):
    """Ours and tart2ms antennas are in the same index order (per-antenna abs diff < 1 cm)."""
    assert positions["ours"].shape == positions["tart2ms"].shape == (24, 3)
    assert np.abs(positions["ours"] - positions["tart2ms"]).max() < 1e-2


def test_our_positions_match_proj_truth(positions):
    """Our reader's WGS84 transform matches the independent PROJ truth to << 1 mm."""
    assert np.abs(positions["ours"] - positions["truth"]).max() < 1e-3


def test_tart2ms_baselines_are_farther_from_truth(positions):
    """tart2ms baseline lengths deviate from truth far more than ours do."""
    a1, a2 = positions["a1"], positions["a2"]

    def lengths(p):
        return np.linalg.norm(baselines_from_positions(p, a1, a2), axis=1)

    truth_len = lengths(positions["truth"])
    ours_err = np.abs(lengths(positions["ours"]) - truth_len).max()
    tart_err = np.abs(lengths(positions["tart2ms"]) - truth_len).max()
    print(f"\nL1 baseline-length max error: ours={ours_err * 1e3:.4f} mm, tart2ms={tart_err * 1e3:.4f} mm")
    assert ours_err < 1e-3  # ours matches the geodetic standard (sub-mm)
    assert tart_err > 5 * ours_err  # tart2ms is materially farther from truth
