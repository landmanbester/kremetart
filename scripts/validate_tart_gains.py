"""Numerical check that StefCAL phase-only gains reproduce TART's calibrated image (host/CPU).

Images one frame of one TART HDF three ways -- no calibration, TART's stored gains, and our StefCAL
phase-only gains -- and reports the pixel correlation of each against the TART-calibrated image over
above-horizon pixels. A high ``corr(ours, TART)`` (vs ~0 for no-cal) means our gains recover the
TART-quality image. Uses the dirty imager stage on the CPU, so no GPU/Holoscan is needed.

Example:

    python scripts/validate_tart_gains.py tests/data_stefcal/vis_2026-06-09_08_11_43.476804.hdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from stefcal_calibrate import solve_file_gains

from kremetart.imaging.dirty_imager import DirtyImager
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.snapshot import Snapshot
from kremetart.utils import partition_datatree
from kremetart.utils.gains import apply_inverse_gains
from kremetart.utils.geometry import equatorial_baselines, zenith_icrs_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf", help="a TART vis_*.hdf file (with TART's original gains still present)")
    parser.add_argument("--catalog-cache", default=None, help="catalog.zarr (default: <hdf parent>/catalog.zarr)")
    parser.add_argument("--nside", type=int, default=64, help="HEALPix nside for the test image")
    parser.add_argument("--frame", type=int, default=None, help="frame index to image (default: middle frame)")
    parser.add_argument("--no-beam", action="store_true", help="disable the Airy beam in the StefCAL model")
    args = parser.parse_args()

    hdf = Path(args.hdf)
    cache = args.catalog_cache or str(hdf.parent / "catalog.zarr")

    node = partition_datatree(xr.open_datatree(hdf, engine="tart"))
    main_ds = node.ds
    ant = node["antenna_xds"].to_dataset(inherit=False)
    names = list(ant.antenna_name.values)
    n_ant = len(names)
    index = {n: i for i, n in enumerate(names)}
    a1 = np.array([index[n] for n in main_ds.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in main_ds.baseline_antenna2_name.values])
    bl_itrs = ant.ANTENNA_POSITION.values[a1] - ant.ANTENNA_POSITION.values[a2]
    freqs = np.asarray(main_ds.frequency.values)
    times = np.asarray(main_ds.time.values)
    vis = np.asarray(main_ds.VISIBILITY.values)[:, :, :, 0]
    wgt = np.asarray(main_ds.WEIGHT.values)[:, :, :, 0]
    obs = main_ds.attrs["observation_info"]
    lat, lon, alt = obs["site_latitude_deg"], obs["site_longitude_deg"], obs["site_altitude_m"]

    g_tart = np.asarray(node["gain_xds"].to_dataset(inherit=False).isel(gain_time=0).GAIN.values)
    gains_ours, _names, _ref = solve_file_gains(hdf, cache, use_beam=not args.no_beam)
    g_ours = np.where(np.isfinite(gains_ours), np.exp(1j * np.angle(gains_ours)), 0.0)  # phase-only

    grid = SkyGrid.build(args.nside, nest=True, apply_beam=True, xp=np)
    imager = DirtyImager(grid)
    pix = grid.pix_vec
    bore = zenith_icrs_vectors(times, lat, lon, alt)
    frame = args.frame if args.frame is not None else vis.shape[0] // 2
    b_rot = equatorial_baselines(bl_itrs, times[frame : frame + 1], xp=np)[0]

    def image(gains):
        vis_c, wgt_c = apply_inverse_gains(vis[frame : frame + 1], wgt[frame : frame + 1], gains, a1, a2)
        snapshot = Snapshot(
            vis=vis_c[0], weight=wgt_c[0], b_rot=b_rot, boresight=bore[frame], time=float(times[frame]), freqs=freqs
        )
        return imager.image(snapshot)

    img_nocal = image(np.ones(n_ant, dtype=complex))
    img_tart = image(g_tart)
    img_ours = image(g_ours)
    mask = pix @ bore[frame] > 0.05  # above-horizon pixels only

    def corr(a, b):
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    print(f"{hdf.name} frame {frame}: above-horizon pixels = {int(mask.sum())}, nside = {args.nside}")
    print(f"  corr(no-cal, TART) = {corr(img_nocal, img_tart):+.3f}")
    print(f"  corr(ours,   TART) = {corr(img_ours, img_tart):+.3f}   <- high => our gains recover the TART image")


if __name__ == "__main__":
    main()
