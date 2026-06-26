"""Shared helper for the StefCAL gain-calibration scripts (host/CPU; no GPU/Holoscan).

Solves a single TART HDF file's acquisition StefCAL gains -- one solution pooled over the whole file
(``t_int = nframes``), with the model visibilities beam-weighted by the Airy primary beam -- using
the ``kremetart.utils`` calibration utilities. The catalogue comes from a cached ``catalog.zarr`` so
the solve runs offline. This is the seed of the future calibration operator/driver; for now it lives
under ``scripts/`` as a validation helper.
"""

from __future__ import annotations

import numpy as np

from kremetart.utils import partition_datatree
from kremetart.utils.beam import airy_power_beam
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
from kremetart.utils.satellites import frame_source_directions
from kremetart.utils.skymodel import enu_direction_cosines, model_visibilities
from kremetart.utils.stefcal import stefcal_solve

_ZENITH = np.array([0.0, 0.0, 1.0])  # antenna boresight in the ENU frame


def solve_file_gains(hdf_path, cache_path, *, elevation_deg: float = 45.0, use_beam: bool = True):
    """Solve one TART file's StefCAL gains, pooled over the whole file.

    Args:
        hdf_path: path to a TART ``vis_*.hdf`` file.
        cache_path: path to a cached ``catalog.zarr`` (offline catalogue source).
        elevation_deg: catalogue elevation cutoff the cache was built at.
        use_beam: weight the model visibilities by the Airy primary beam (recommended).

    Returns:
        ``(gains, names, ref_ant)``: ``gains`` is ``(n_ant,)`` complex with ``gains[ref_ant] == 1``
        and dead antennas ``NaN``; ``names`` the antenna names; ``ref_ant`` the reference antenna.
    """
    node = partition_datatree(read_hdf_as_msv4(hdf_path))
    main = node.ds
    ant = node["antenna_xds"].to_dataset(inherit=False)
    names = list(ant.antenna_name.values)
    n_ant = len(names)
    index = {n: i for i, n in enumerate(names)}
    a1 = np.array([index[n] for n in main.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in main.baseline_antenna2_name.values])
    bl_enu = ant.ANTENNA_POSITION_ENU.values[a1] - ant.ANTENNA_POSITION_ENU.values[a2]
    freqs = np.asarray(main.frequency.values)
    vis = np.asarray(main.VISIBILITY.values)[:, :, :, 0]  # (ntime, nbl, nchan)
    wgt = np.asarray(main.WEIGHT.values)[:, :, :, 0]
    dead = np.asarray(node["gain_xds"].to_dataset(inherit=False).ANTENNA_FLAG.values)
    ref = int(np.where(~dead)[0][0])  # first live antenna

    per_frame = frame_source_directions([hdf_path], elevation_deg, cache_path=str(cache_path))
    models = []
    for sources in per_frame:
        az = np.array([a for _, a, _ in sources])
        el = np.array([e for _, _, e in sources])
        s = enu_direction_cosines(az, el)
        beam = airy_power_beam(s, _ZENITH, freqs) if use_beam else None
        models.append(model_visibilities(s, bl_enu, freqs, beam=beam))
    models = np.stack(models)  # (ntime, nbl, nchan)

    gains = stefcal_solve(vis, models, a1, a2, n_ant, t_int=vis.shape[0], ref_ant=ref, weight=wgt)[0][0]
    return gains, names, ref
