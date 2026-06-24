"""Numerical check that StefCAL phase-only gains reproduce TART's calibrated image (host/CPU).

Images one frame of one TART HDF three ways -- no calibration, TART's stored gains, and our StefCAL
phase-only gains -- and reports the pixel correlation of each against the TART-calibrated image over
above-horizon pixels. A high ``corr(ours, TART)`` (vs ~0 for no-cal) means our gains recover the
TART-quality image. Uses the gridless HEALPix imager on the CPU, so no GPU/Holoscan is needed.

Example:

    python scripts/validate_tart_gains.py tests/data_stefcal/vis_2026-06-09_08_11_43.476804.hdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stefcal_calibrate import solve_file_gains

from kremetart.utils import partition_datatree
from kremetart.utils.beam import airy_power_beam
from kremetart.utils.gains import apply_inverse_gains
from kremetart.utils.healpix_dft import image_frame, make_pixel_grid, zenith_icrs_vectors
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4


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

    node = partition_datatree(read_hdf_as_msv4(hdf))
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

    g_tart = np.asarray(node["gain_xds"].to_dataset(inherit=False).GAIN.values)
    gains_ours, _names, _ref = solve_file_gains(hdf, cache, use_beam=not args.no_beam)
    g_ours = np.where(np.isfinite(gains_ours), np.exp(1j * np.angle(gains_ours)), 0.0)  # phase-only

    pix = make_pixel_grid(args.nside, nest=True)
    bore = zenith_icrs_vectors(times, lat, lon, alt)
    frame = args.frame if args.frame is not None else vis.shape[0] // 2

    def image(gains):
        vis_c, wgt_c = apply_inverse_gains(vis[frame : frame + 1], wgt[frame : frame + 1], gains, a1, a2)
        beam = airy_power_beam(pix, bore[frame], freqs)
        return image_frame(vis_c, wgt_c, times[frame : frame + 1], bl_itrs, pix, freqs, beam=beam, xp=np)

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
