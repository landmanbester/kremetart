"""Prepare a TART HDF sequence into one imaging-ready snapshot zarr, the input every runner reads."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from kremetart.utils import partition_datatree
from kremetart.utils.calibration import correct_file_gains
from kremetart.utils.geometry import equatorial_baselines, radec_to_unit_vectors


def prepare_msv4_zarr(
    hdf_paths: Iterable[Path | str],
    out_zarr: Path | str,
    *,
    correct_gains: bool = False,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    nframes: int | None = None,
):
    """Write the HDF sequence to one imaging-ready zarr; return the output path.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (frames are emitted in this order).
        out_zarr: output zarr path (overwritten if present).
        correct_gains: divide vis/weights by the per-antenna gain product before writing.
        phase_ra_deg: common phase-direction RA (deg, ICRS); stored in attrs (``NaN`` if unset).
        phase_dec_deg: common phase-direction Dec (deg, ICRS); stored in attrs (``NaN`` if unset).
        nframes: optional cap on the total number of frames written (profiling/preview aid).

    Returns:
        The ``out_zarr`` path.

    Raises:
        FileNotFoundError: if no frames are produced.
    """

    out_zarr = Path(out_zarr)
    vis_all: list[np.ndarray] = []
    wgt_all: list[np.ndarray] = []
    brot_all: list[np.ndarray] = []
    bore_all: list[np.ndarray] = []
    time_all: list[np.ndarray] = []
    freqs = None
    info = None

    for path in hdf_paths:
        if nframes is not None and sum(v.shape[0] for v in vis_all) >= nframes:
            break
        node = partition_datatree(xr.open_datatree(path, engine="tart"))
        main = node.ds
        times = np.asarray(main.time.values)
        antenna = node["antenna_xds"].to_dataset(inherit=False)
        pos = antenna.ANTENNA_POSITION.values  # (n_ant, 3) ITRS
        names = list(antenna.antenna_name.values)
        index = {name: i for i, name in enumerate(names)}
        a1 = np.array([index[n] for n in node.ds.baseline_antenna1_name.values])
        a2 = np.array([index[n] for n in node.ds.baseline_antenna2_name.values])
        bl = np.asarray(np.asarray(pos[a1] - pos[a2]))  # (nbl, 3) host
        vis = np.asarray(main.VISIBILITY.values)[..., 0]  # (n_time, nbl, nchan)
        wgt = np.asarray(main.WEIGHT.values)[..., 0]
        if freqs is None:
            freqs = np.asarray(main.frequency.values)
            info = main.attrs["observation_info"]
        if correct_gains:
            vis, wgt = correct_file_gains(node, vis, wgt, xp=np)
        b_rot = equatorial_baselines(bl, times, xp=np)  # (n_time, nbl, 3)
        # Each snapshot's phase centre is its instantaneous zenith -> the Airy beam boresight.
        field = node["field_and_source_base_xds"].to_dataset(inherit=False)
        ra, dec = field.FIELD_PHASE_CENTER_DIRECTION.sel(field_name=main.field_name.values).values.T
        boresight = radec_to_unit_vectors(ra, dec)  # (n_time, 3) ICRS zenith unit vectors
        vis_all.append(np.asarray(vis))
        wgt_all.append(np.asarray(wgt))
        brot_all.append(np.asarray(b_rot))
        bore_all.append(np.asarray(boresight))
        time_all.append(times)

    if not vis_all:
        raise FileNotFoundError("no HDF frames to prepare")

    vis_c = np.concatenate(vis_all, axis=0)
    wgt_c = np.concatenate(wgt_all, axis=0)
    brot = np.concatenate(brot_all, axis=0)
    bore = np.concatenate(bore_all, axis=0)
    tt = np.concatenate(time_all, axis=0)
    if nframes is not None:
        vis_c, wgt_c, brot, bore, tt = (
            vis_c[:nframes],
            wgt_c[:nframes],
            brot[:nframes],
            bore[:nframes],
            tt[:nframes],
        )

    ds = xr.Dataset(
        data_vars={
            "VISIBILITY": (("time", "baseline", "frequency"), vis_c.astype(np.complex64)),
            "WEIGHT": (("time", "baseline", "frequency"), wgt_c.astype(np.float32)),
            "B_ROT": (("time", "baseline", "xyz"), brot.astype(np.float64)),
            "BORESIGHT": (("time", "xyz"), bore.astype(np.float64)),
        },
        coords={
            "time": ("time", tt.astype(np.float64)),
            "frequency": ("frequency", np.asarray(freqs, dtype=np.float64)),
            "xyz": ("xyz", np.array(["x", "y", "z"])),
        },
        attrs={
            "phase_ra_deg": float("nan") if phase_ra_deg is None else float(phase_ra_deg),
            "phase_dec_deg": float("nan") if phase_dec_deg is None else float(phase_dec_deg),
            "site_latitude_deg": float(info["site_latitude_deg"]),
            "site_longitude_deg": float(info["site_longitude_deg"]),
            "site_altitude_m": float(info["site_altitude_m"]),
        },
    )
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    ds.to_zarr(out_zarr, mode="w")
    return out_zarr
