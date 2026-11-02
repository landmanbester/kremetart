from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
from tart_tools import api_handler

from kremetart.utils import partition_datatree
from kremetart.utils.calibration import correct_file_gains
from kremetart.utils.geometry import equatorial_baselines, radec_to_unit_vectors


def read_hdf_as_xr(path, filter_elevation=45):
    # open hdf as xarray dataset
    ds = xr.open_dataset(path, engine="netcdf4")

    # get the config
    # eg for the Mauritius array, this is the config dict:
    # {'name': 'Bel Air - Mauritius',
    # 'frequency': 1575420000.0,
    # 'L0_frequency': 1571328000.0,
    # 'baseband_frequency': 4092000.0,
    # 'sampling_frequency': 16368000.0,
    # 'bandwidth': 2500000.0,
    # 'lat': -20.2587508,
    # 'lon': 57.7591989,
    # 'alt': 20.0,
    # 'num_antenna': 24,
    # 'orientation': 0.0,
    # 'axes': ['East', 'North', 'Up']}
    config = json.loads(ds.config.values[0])

    # attributes
    attrs = {}
    for key, value in config.items():
        attrs[key] = value
    attrs["phase_elaz_deg"] = ds.phase_elaz.values

    # coordinates
    n_baselines = ds.baselines.shape[0]
    n_antenna = ds.antenna_positions.shape[0]
    n_times = ds.timestamp.shape[0]
    coords = {
        "time": ("time", ds.timestamp.values),
        "frequency": ("frequency", np.array([config["frequency"]])),
        "baseline_id": ("baseline_id", np.arange(n_baselines)),
        "antenna_id": ("antenna_id", np.arange(n_antenna)),
        "antenna1": ("antenna1", ds.baselines.values[:, 0]),
        "antenna2": ("antenna2", ds.baselines.values[:, 1]),
        "polarisation": ("polarisation", np.array(["LL"])),
        "enu": ("enu", np.array(config["axes"])),
    }

    # flag faulty antenna
    faulty_idx = np.where(ds.gains.values == 0)[0]
    flag = np.zeros(n_baselines, dtype=bool)
    for idx in faulty_idx:
        flag |= (ds.baselines.values[:, 0] == idx) | (ds.baselines.values[:, 1] == idx)
    flag = np.broadcast_to(flag[None, :], (n_times, n_baselines))

    # data variables
    shape = (ds.timestamp.shape[0], ds.baselines.shape[0], 1)  # add polarisation dim
    data_vars = {
        "antenna_positions": (("antenna_id", "enu"), ds.antenna_positions.values),
        "vis": (("time", "baseline_id", "polarisation"), ds.vis.values[:, :, None]),
        "weight": (("time", "baseline_id", "polarisation"), np.ones(shape, dtype=np.float32)),
        "flag": (("time", "baseline_id", "polarisation"), flag[:, :, None]),
    }

    # query the API to get the satelite positions
    lat = config["lat"]
    lon = config["lon"]
    api = api_handler.APIhandler("")
    nretry = 5
    catalog = []
    for t in ds.timestamp.values:
        cat_url = (
            api.catalog_url(
                lon,
                lat,
                datestr=t,
            )
            + f"&elevation={filter_elevation}"
        )
        retry = 0
        while retry < nretry:
            try:
                cat = api.get_url(cat_url)
                break
            except Exception as e:
                print(f"Error fetching catalog (attempt {retry + 1}/{nretry}): {e}")
                retry += 1
        else:
            raise RuntimeError(f"Failed to fetch catalog after {nretry} attempts.")
        catalog.append(cat)

    # append source data vars
    data_vars["source_name"] = (("time",), [cat[0]["name"] for cat in catalog])
    data_vars["source_elevation_deg"] = (("time",), [cat[0]["el"] for cat in catalog])
    data_vars["source_azimuth_deg"] = (("time",), [cat[0]["az"] for cat in catalog])
    data_vars["source_flux_jy"] = (("time",), [cat[0]["jy"] for cat in catalog])
    data_vars["source_height_m"] = (("time",), [cat[0]["r"] for cat in catalog])

    # create xarray dataset
    xds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    return xds


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
