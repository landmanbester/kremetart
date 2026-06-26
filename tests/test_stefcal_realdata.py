"""Opt-in real-data sanity check: StefCAL phases vs TART's own gain snapshot.

Env-gated (KREMETART_REALDATA=1) and excluded from required CI. Uses the bundled catalogue cache so
it never queries the (slow) TART API. The acquisition config that best matches TART's per-file
snapshot is: all catalogue sources, beam-weighted (the Airy power beam down-weights low-elevation
sources better than a hard elevation cut), pooled over the whole file (``t_int = ntime``). The
comparison is gauge-invariant (optimal global phase): the ENU b.s convention is verified consistent
with the imaging path elsewhere, and TART's snapshot is a different (coarser) solution rather than a
per-frame ground truth -- our solution in fact fits this frame's visibilities better -- so the
remaining ~20 deg is model-limited and the threshold here is a loose regression guard, not a target.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("KREMETART_REALDATA") != "1", reason="set KREMETART_REALDATA=1 to run")


def test_stefcal_phases_track_tart_snapshot(ref_hdf, catalog_cache, catalog_elevation):
    from kremetart.utils import partition_datatree
    from kremetart.utils.beam import airy_power_beam
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
    zenith = np.array([0.0, 0.0, 1.0])  # ENU antenna boresight

    vis = np.asarray(main.VISIBILITY.values)[:, :, :, 0]  # (ntime, nbl, nchan)
    weight = np.asarray(main.WEIGHT.values)[:, :, :, 0]
    ntime = vis.shape[0]

    # Per-frame beam-weighted model (sources move between integrations).
    per_frame = frame_source_directions([ref_hdf], catalog_elevation, cache_path=catalog_cache)
    assert len(per_frame) == ntime
    models = []
    for sources in per_frame:
        az = np.array([a for _, a, _ in sources])
        el = np.array([e for _, _, e in sources])
        s = enu_direction_cosines(az, el)
        beam = airy_power_beam(s, zenith, freqs)
        models.append(model_visibilities(s, bl, freqs, beam=beam))
    models = np.stack(models)  # (ntime, nbl, nchan)

    gain = node["gain_xds"].to_dataset(inherit=False)
    dead = np.where(np.asarray(gain.ANTENNA_FLAG.values))[0]
    ref = int(np.setdiff1d(np.arange(n_ant), dead)[0])  # first live antenna

    gains, info = stefcal_solve(vis, models, a1, a2, n_ant, t_int=ntime, ref_ant=ref, weight=weight)
    assert bool(info["converged"][0])
    g_hat = gains[0]

    got = referenced_phases(g_hat, ref)
    snap = np.angle(np.asarray(gain.GAIN.values))
    want = snap - snap[ref]
    live = np.isfinite(g_hat)
    live[ref] = False  # the pinned reference is identically 0 in both, exclude from the stat

    # Gauge-invariant comparison: remove the best global phase before measuring the RMS.
    d = (got - want)[live]
    phi = np.angle(np.sum(np.exp(1j * d)))
    rms = np.sqrt(np.mean(np.angle(np.exp(1j * (d - phi))) ** 2))
    print(f"[realdata] gauge-invariant RMS phase diff vs TART = {np.degrees(rms):.1f} deg over {live.sum()} ant")
    assert np.all(np.isfinite(got[np.isfinite(g_hat)]))
    assert rms < np.radians(30.0)  # loose regression guard (measured ~19 deg)
