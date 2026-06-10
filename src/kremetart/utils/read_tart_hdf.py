import json

import numpy as np
import xarray as xr
from tart_tools import api_handler


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
