"""Overwrite a directory of TART HDFs' gain solution with our StefCAL phase-only solution.

For each ``vis_*.hdf`` in the directory, solve the acquisition StefCAL gains (beam-weighted, pooled
over the whole file) and write **phase-only** gains back into the HDF -- unit amplitude for live
antennas, 0 for the dead one -- so ``kremetart smoovie --correct-gains`` images with our calibration
instead of TART's. The amplitudes are discarded because the unit-flux acquisition model leaves them
unreliable; only the phases are trusted (and the global phase cancels in ``g_p conj(g_q)``).

Run this only on a *copy* of the data (e.g. ``tests/data_stefcal``), never the pristine
``tests/data``. Example:

    python scripts/overwrite_tart_gains.py tests/data_stefcal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from stefcal_calibrate import solve_file_gains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf_dir", help="directory of TART vis_*.hdf files to overwrite in place")
    parser.add_argument("--catalog-cache", default=None, help="catalog.zarr (default: <hdf_dir>/catalog.zarr)")
    parser.add_argument("--elevation-deg", type=float, default=45.0, help="catalogue elevation cutoff of the cache")
    parser.add_argument("--no-beam", action="store_true", help="disable the Airy primary-beam weighting")
    args = parser.parse_args()

    hdf_dir = Path(args.hdf_dir)
    cache = args.catalog_cache or str(hdf_dir / "catalog.zarr")
    files = sorted(hdf_dir.glob("vis_*.hdf"))
    if not files:
        raise SystemExit(f"no vis_*.hdf files in {hdf_dir}")

    for path in files:
        gains, _names, _ref = solve_file_gains(path, cache, elevation_deg=args.elevation_deg, use_beam=not args.no_beam)
        live = np.isfinite(gains)
        amp = np.where(live, 1.0, 0.0).astype(np.float32)  # phase-only: unit amplitude, dead -> 0
        phase = np.where(live, np.angle(gains), 0.0).astype(np.float32)
        with h5py.File(path, "r+") as h:  # plain HDF5 -> h5py, not the netCDF writer
            h["gains"][...] = amp
            h["phases"][...] = phase
        print(f"{path.name}: wrote phase-only gains for {int(live.sum())}/{live.size} live antennas (dead -> 0)")


if __name__ == "__main__":
    main()
