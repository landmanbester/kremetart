"""Compare Tikhonov vs reweighted-L1 deconvolution on one TART frame (host/CPU).

Images one frame of one TART HDF three ways -- raw dirty, Tikhonov (CG on H + λI), and reweighted-L1
(FISTA on the same H) -- at matched strength ``λ = eta·Σw``, and reports a point-source concentration
metric over above-horizon pixels: the fraction of total flux in the brightest pixels (higher => more
point-like => cleaner). Uses the gridless HEALPix Hessian on the CPU, so no GPU/Holoscan is needed.

Example:

    python scripts/compare_regularisers.py tests/data_stefcal/vis_2026-06-09_08_11_43.476804.hdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kremetart.opt.cg import cg
from kremetart.opt.fista import fista_quadratic
from kremetart.utils import partition_datatree
from kremetart.utils.beam import airy_power_beam
from kremetart.utils.healpix_dft import (
    equatorial_baselines,
    hessian_healpix,
    image_frame,
    make_pixel_grid,
    zenith_icrs_vectors,
)
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4


def _topk_fraction(x: np.ndarray, mask: np.ndarray, k: int) -> float:
    """Fraction of total (non-negative) flux held by the brightest ``k`` above-horizon pixels."""
    vals = np.clip(x[mask], 0.0, None)
    total = float(vals.sum())
    if total == 0.0:
        return 0.0
    return float(np.sort(vals)[-k:].sum()) / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hdf", help="a TART vis_*.hdf file")
    parser.add_argument("--nside", type=int, default=64, help="HEALPix nside for the test image")
    parser.add_argument("--frame", type=int, default=None, help="frame index (default: middle frame)")
    parser.add_argument("--eta", type=float, default=1e-2, help="regulariser strength as a fraction of Σw")
    parser.add_argument("--max-reweight", type=int, default=3, help="reweighting rounds for the L1 solve")
    parser.add_argument("--no-beam", action="store_true", help="disable the Airy primary beam")
    parser.add_argument("--topk", type=int, default=10, help="k for the brightest-k flux-concentration metric")
    args = parser.parse_args()

    hdf = Path(args.hdf)
    node = partition_datatree(read_hdf_as_msv4(hdf))
    main_ds = node.ds
    ant = node["antenna_xds"].to_dataset(inherit=False)
    names = list(ant.antenna_name.values)
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

    pix = make_pixel_grid(args.nside, nest=True)
    bore = zenith_icrs_vectors(times, lat, lon, alt)
    frame = args.frame if args.frame is not None else vis.shape[0] // 2

    use_beam = not args.no_beam
    beam = airy_power_beam(pix, bore[frame], freqs) if use_beam else None
    sl = slice(frame, frame + 1)
    dirty = image_frame(vis[sl], wgt[sl], times[sl], bl_itrs, pix, freqs, beam=beam, xp=np)  # (npix,) normalised

    rows = equatorial_baselines(bl_itrs, times[sl], xp=np)[0]  # (nbl, 3) for this frame
    w = wgt[frame]  # (nbl, nchan)
    wsum = float(w.sum())
    hmv, hdiag = hessian_healpix(rows, pix, freqs, w, beam=beam, xp=np)
    b = dirty * wsum
    lam = args.eta * wsum

    x_tik = cg(lambda x: hmv(x) + lam * x, b, maxiter=100, tol=1e-5, xp=np)
    x_l1, info = fista_quadratic(
        hmv, b, lam=lam, positive=True, L0=float(hdiag.max()), max_reweight=args.max_reweight, xp=np
    )

    mask = pix @ bore[frame] > 0.05  # above-horizon pixels only
    k = args.topk
    print(f"{hdf.name} frame {frame}: nside={args.nside}, eta={args.eta}, Σw={wsum:.3g}, k={k}")
    print(f"  top-{k} flux fraction  dirty   = {_topk_fraction(np.abs(dirty), mask, k):.3f}")
    print(f"  top-{k} flux fraction  tikhonov= {_topk_fraction(x_tik, mask, k):.3f}")
    print(f"  top-{k} flux fraction  l1      = {_topk_fraction(x_l1, mask, k):.3f}   <- higher => cleaner")
    print(f"  l1 solve: reweights={info['reweights']}, iterations={info['iterations']}, converged={info['converged']}")


if __name__ == "__main__":
    main()
