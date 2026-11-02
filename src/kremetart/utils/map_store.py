"""The durable map store: a ``(TIME, PIX)`` zarr with one float32 variable per map name.

Host-only (NumPy, xarray, dask). Every runner writes through these two functions, so the output format
is defined once: :func:`create_map_store` writes a lazy scaffold sized for the whole run, then
:func:`write_maps` fills one snapshot's row, located by matching its time against the ``TIME``
coordinate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr


def create_map_store(path: Path | str, times: np.ndarray, npix: int, names: Iterable[str]) -> None:
    """Write an empty ``(TIME, PIX)`` scaffold (no data allocated), replacing anything at ``path``.

    Args:
        path: output zarr path.
        times: every snapshot time the run will write, in order.
        npix: pixels per map.
        names: the map variables to create.
    """
    ntime = len(times)
    data_vars = {name: (("TIME", "PIX"), da.empty((ntime, npix), chunks=(1, npix), dtype=np.float32)) for name in names}
    ds = xr.Dataset(data_vars=data_vars, coords={"TIME": (("TIME",), np.asarray(times)), "PIX": np.arange(npix)})
    ds.to_zarr(path, mode="w", compute=True)


def write_maps(path: Path | str, time: float, maps: Mapping[str, np.ndarray]) -> None:
    """Write one snapshot's maps (each ``(npix,)`` on the host) into the row whose time is ``time``.

    Raises:
        KeyError: if ``time`` is not one of the scaffold's times.
    """
    data_vars = {name: (("TIME", "PIX"), np.asarray(m, dtype=np.float32).reshape(1, -1)) for name, m in maps.items()}
    npix = next(iter(data_vars.values()))[1].shape[1]
    ds = xr.Dataset(data_vars=data_vars, coords={"TIME": (("TIME",), np.array([time])), "PIX": np.arange(npix)})
    ds.to_zarr(path, region="auto")
