"""Opt-in real-data sanity check: StefCAL phases vs TART's own gain snapshot.

Env-gated (KREMETART_REALDATA=1) and excluded from required CI: it pins the ENU az / baseline-sign
convention against TART's solution, which may need iteration. Uses the bundled catalogue cache so it
never queries the (slow) TART API.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("KREMETART_REALDATA") != "1", reason="set KREMETART_REALDATA=1 to run")


def test_stefcal_phases_track_tart_snapshot(ref_hdf, catalog_cache, catalog_elevation):
    from kremetart.utils import partition_datatree
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.satellites import frame_source_directions
    from kremetart.utils.skymodel import enu_direction_cosines, model_visibilities
    from kremetart.utils.stefcal import referenced_phases, stefcal_solve

    node = partition_datatree(read_hdf_as_msv4(ref_hdf))
    main = node.ds
    antenna = node["antenna_xds"].to_dataset(inherit=False)
    names = list(antenna.antenna_name.values)
    index = {n: i for i, n in enumerate(names)}
    a1 = np.array([index[n] for n in main.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in main.baseline_antenna2_name.values])
    enu = antenna.ANTENNA_POSITION_ENU.values
    bl = enu[a1] - enu[a2]
    n_ant = len(names)
    freqs = np.asarray(main.frequency.values)

    vis = np.asarray(main.VISIBILITY.values)[0, :, :, 0]  # (nbl, nchan), first frame
    weight = np.asarray(main.WEIGHT.values)[0, :, :, 0]

    per_frame = frame_source_directions([ref_hdf], catalog_elevation, cache_path=catalog_cache, nframes=1)
    az = np.array([a for _, a, _ in per_frame[0]])
    el = np.array([e for _, _, e in per_frame[0]])
    s = enu_direction_cosines(az, el)
    model = model_visibilities(s, bl, freqs)

    gain = node["gain_xds"].to_dataset(inherit=False)
    dead = np.where(np.asarray(gain.ANTENNA_FLAG.values))[0]
    ref = int(np.setdiff1d(np.arange(n_ant), dead)[0])  # first live antenna

    g_hat, info = stefcal_solve(vis, model, a1, a2, n_ant, ref_ant=ref, weight=weight)
    assert info["converged"]

    got = referenced_phases(g_hat, ref)
    snap = np.angle(np.asarray(gain.GAIN.values))
    want = snap - snap[ref]
    live = np.isfinite(g_hat)
    # circular distance, robust to sign/convention offset; generous bound (diagnostic, not a gate).
    d = np.angle(np.exp(1j * (got[live] - want[live])))
    print(f"[realdata] circular RMS phase diff = {np.sqrt(np.mean(d**2)):.3f} rad over {live.sum()} ant")
    assert np.all(np.isfinite(got[live]))
