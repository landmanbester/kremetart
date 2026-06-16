"""Host prepare-step: a TART HDF sequence -> one imaging-ready zarr for the GPU smoovie app.

Does every host/astropy/gain task once, up front, so the streaming Holoscan imager (the GPU
``HealpixDFTOperator``) is pure cupy. For the whole HDF sequence it reads each file to MSv4,
optionally applies the inverse per-antenna gains (:func:`kremetart.core.smoovie._correct_file_gains`
-> :func:`kremetart.utils.gains.apply_inverse_gains`), and precomputes the per-frame
equatorial-rotated baselines ``b_rot(t)`` (:func:`kremetart.utils.healpix_dft.equatorial_baselines`).
The result is a single ``xarray.Dataset`` written to zarr with corrected ``VISIBILITY``/``WEIGHT``,
``B_ROT``, the ``time``/``frequency`` coordinates, and the common phase-direction + site metadata.

The zarr is nside-independent (``B_ROT`` and the corrected visibilities do not depend on the pixel
grid) and reusable across runs. This module is host-only (numpy/astropy/xarray) and must stay
importable on a CPU machine -- it never imports cupy or holoscan.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


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
        hdf_paths: ordered iterable of TART HDF paths (same order as ``frame_dirty_maps``).
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
    import shutil

    import xarray as xr

    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.utils.healpix_dft import equatorial_baselines
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.rephasing import itrs_baselines

    out_zarr = Path(out_zarr)
    vis_all: list[np.ndarray] = []
    wgt_all: list[np.ndarray] = []
    brot_all: list[np.ndarray] = []
    time_all: list[np.ndarray] = []
    freqs = None
    info = None

    for path in hdf_paths:
        if nframes is not None and sum(v.shape[0] for v in vis_all) >= nframes:
            break
        node = _partition(read_hdf_as_msv4(path))
        main = node.ds
        times = np.asarray(main.time.values)
        bl = np.asarray(itrs_baselines(node, np))  # (nbl, 3) host
        vis = np.asarray(main.VISIBILITY.values)[..., 0]  # (n_time, nbl, nchan)
        wgt = np.asarray(main.WEIGHT.values)[..., 0]
        if freqs is None:
            freqs = np.asarray(main.frequency.values)
            info = main.attrs["observation_info"]
        if correct_gains:
            vis, wgt = _correct_file_gains(node, vis, wgt, xp=np)
        b_rot = equatorial_baselines(bl, times, xp=np)  # (n_time, nbl, 3)
        vis_all.append(np.asarray(vis))
        wgt_all.append(np.asarray(wgt))
        brot_all.append(np.asarray(b_rot))
        time_all.append(times)

    if not vis_all:
        raise FileNotFoundError("no HDF frames to prepare")

    vis_c = np.concatenate(vis_all, axis=0)
    wgt_c = np.concatenate(wgt_all, axis=0)
    brot = np.concatenate(brot_all, axis=0)
    tt = np.concatenate(time_all, axis=0)
    if nframes is not None:
        vis_c, wgt_c, brot, tt = vis_c[:nframes], wgt_c[:nframes], brot[:nframes], tt[:nframes]

    ds = xr.Dataset(
        data_vars={
            "VISIBILITY": (("time", "baseline", "frequency"), vis_c.astype(np.complex64)),
            "WEIGHT": (("time", "baseline", "frequency"), wgt_c.astype(np.float32)),
            "B_ROT": (("time", "baseline", "xyz"), brot.astype(np.float64)),
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
