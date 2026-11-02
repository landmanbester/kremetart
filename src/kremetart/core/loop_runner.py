"""The loop runner: image a prepared snapshot zarr in-process, one snapshot at a time.

Builds the :class:`SnapshotImager` chain on the requested array module (NumPy by default, so it runs
on any machine) and drives it in a plain Python loop. Each snapshot's maps go to the durable map store
and, when a holder is given, to the live viewer.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import xarray as xr

from kremetart.core.settings import STREAM_NAME, ImagingSettings
from kremetart.imaging.snapshot import Array, Snapshot
from kremetart.imaging.snapshot_imager import SnapshotImager, SnapshotMaps
from kremetart.utils.healpix_viz import LatestFrameHolder, publish_maps
from kremetart.utils.map_store import create_map_store, write_maps


def run_loop(
    prepared: Path,
    output: Path,
    settings: ImagingSettings,
    holder: LatestFrameHolder | None = None,
    *,
    xp: ModuleType = np,
) -> None:
    """Image every snapshot in ``prepared`` into the map store at ``output`` (the runner interface).

    Args:
        prepared: the prepared snapshot zarr.
        output: the map store to create.
        settings: the imaging settings.
        holder: when given, also publish each snapshot's maps to the live viewer.
        xp: array module the stages run on (``numpy``, or ``cupy`` on a GPU).
    """
    ds = xr.open_zarr(prepared)
    times = ds.time.values
    vis, weight, b_rot, boresight = (ds[v].values for v in ("VISIBILITY", "WEIGHT", "B_ROT", "BORESIGHT"))
    freqs = xp.asarray(ds.frequency.values)

    imager = SnapshotImager.build(
        settings.nside,
        l2=settings.l2,
        l1=settings.l1,
        sigma2=settings.iwp_sigma,
        noise=settings.iwp_noise,
        nest=settings.nest,
        apply_beam=settings.apply_beam,
        ground_plane_diameter=settings.ground_plane_diameter,
        xp=xp,
    )
    create_map_store(output, times, settings.npix, settings.stored_names)

    for k, t in enumerate(times):
        snapshot = Snapshot(
            vis=xp.asarray(vis[k]),
            weight=xp.asarray(weight[k]),
            b_rot=xp.asarray(b_rot[k]),
            boresight=xp.asarray(boresight[k]),
            time=float(t),
            freqs=freqs,
        )
        maps = _stored_maps(imager.image(snapshot))
        write_maps(output, float(t), maps)
        if holder is not None:
            publish_maps(holder, k, float(t), {STREAM_NAME[name]: m for name, m in maps.items()})


def _stored_maps(maps: SnapshotMaps) -> dict[str, np.ndarray]:
    """The snapshot's maps on the host, keyed by map-store name (``l1`` only when produced)."""
    named = {
        "dirty": maps.dirty,
        "tikhonov": maps.tikhonov,
        "l1": maps.l1,
        "filtered": maps.smooth,
        "znorm": maps.normalised_innovation,
    }
    return {name: _to_host(m) for name, m in named.items() if m is not None}


def _to_host(a: Array) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)
