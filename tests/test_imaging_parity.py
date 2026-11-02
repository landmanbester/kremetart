"""Parity of the snapshot imager with reference outputs recorded from the pre-stage Holoscan pipeline.

Opt-in: set ``KREMETART_GOLDEN`` to a directory holding ``prepared.zarr`` (the snapshots), plus one or
more reference runs, each a ``<name>.zarr`` with a ``meta_<suffix>.json``/``meta.json`` describing its
settings. Records are made with the capture scripts kept alongside them (the git revision is stored in
the meta file). ``KREMETART_GOLDEN_NFRAMES`` caps the snapshots compared (default: 10 on NumPy, 60 on CuPy, because
FISTA is slow; set it to 540 for a full run); state carries across snapshots, so a prefix is a valid
comparison.

Tolerances (relative L2 per snapshot): references are stored as float32 (relative rounding ~6e-8) and
were computed with CuPy, so the CuPy run must reproduce every map to 1e-6. On NumPy the dirty map is a
direct computation and must agree to 1e-5, but the solvers stop at ``tol=1e-5``, and a different
reduction order changes where they stop: the Tikhonov map already differs by ~3e-4 on the first
snapshot, where both runs start cold, and stays bounded thereafter (measured over 40 snapshots: median
2.7e-5, max 6.6e-4; smooth map max 3.7e-4). The normalised innovation divides by a small predicted
standard deviation and amplifies that (max 1.3e-2). NumPy tolerances follow those measurements.
A failure is investigated, not met by loosening a tolerance.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

GOLDEN = os.environ.get("KREMETART_GOLDEN")
pytestmark = pytest.mark.skipif(not GOLDEN, reason="set KREMETART_GOLDEN=<dir> to run the parity check")

# (reference zarr, meta file, l1 solver overrides)
RUNS = [
    ("golden.zarr", "meta.json", {}),
    ("golden_l1_noreweight.zarr", "meta_l1_noreweight.json", {"max_reweight": 0}),
]
PRODUCTS = {
    "dirty": "dirty",
    "tikhonov": "tikhonov",
    "l1": "l1",
    "filtered": "smooth",
    "znorm": "normalised_innovation",
}


NUMPY_RTOL = {"dirty": 1e-5, "tikhonov": 1e-3, "l1": 1e-3, "filtered": 1e-3, "znorm": 3e-2}
CUPY_RTOL = dict.fromkeys(PRODUCTS, 1e-6)


def _backends():
    out = [pytest.param(np, NUMPY_RTOL, id="numpy")]
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        out.append(pytest.param(cp, CUPY_RTOL, id="cupy"))
    except Exception:
        pass
    return out


def _rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    scale = np.linalg.norm(ref)
    err = np.linalg.norm(got - ref)
    return float(err / scale) if scale > 0 else float(err)


@pytest.mark.parametrize("xp, rtol", _backends())
@pytest.mark.parametrize("ref_name, meta_name, l1_overrides", RUNS, ids=[r[0] for r in RUNS])
def test_snapshot_imager_matches_reference(xp, rtol, ref_name, meta_name, l1_overrides):
    import xarray as xr

    from kremetart.imaging.dirty_imager import DirtyImager
    from kremetart.imaging.iwp_filter import IWPFilter
    from kremetart.imaging.l1_solver import L1Solver
    from kremetart.imaging.sky_grid import SkyGrid
    from kremetart.imaging.snapshot import Snapshot
    from kremetart.imaging.snapshot_imager import SnapshotImager
    from kremetart.imaging.tikhonov_solver import TikhonovSolver

    golden = Path(GOLDEN)
    if not (golden / ref_name).exists():
        pytest.skip(f"{ref_name} not recorded in {golden}")
    meta = json.loads((golden / meta_name).read_text())
    ref = xr.open_zarr(golden / ref_name)
    prep = xr.open_zarr(golden / "prepared.zarr")

    default_n = 10 if xp is np else 60
    nframes = min(int(os.environ.get("KREMETART_GOLDEN_NFRAMES", default_n)), prep.time.size)

    grid = SkyGrid.build(meta["nside"], nest=meta["nest"], apply_beam=meta["apply_beam"], xp=xp)
    imager = SnapshotImager(
        DirtyImager(grid),
        TikhonovSolver(grid, eta=meta["l2"]),
        IWPFilter(grid.npix, sigma2=meta["sigma2"], noise=meta["noise"], xp=xp),
        None if meta.get("l1") is None else L1Solver(grid, eta=meta["l1"], **l1_overrides),
    )
    freqs = xp.asarray(prep.frequency.values)
    vis, wgt, b_rot, bore, times = (prep[v].values for v in ("VISIBILITY", "WEIGHT", "B_ROT", "BORESIGHT", "time"))

    worst = {name: (0.0, -1) for name in PRODUCTS}
    for k in range(nframes):
        snap = Snapshot(
            xp.asarray(vis[k]), xp.asarray(wgt[k]), xp.asarray(b_rot[k]), xp.asarray(bore[k]), float(times[k]), freqs
        )
        maps = imager.image(snap)
        for ref_var, attr in PRODUCTS.items():
            got = getattr(maps, attr)
            if got is None:
                continue
            got = got.get() if hasattr(got, "get") else np.asarray(got)
            err = _rel_l2(got, ref[ref_var].values[k].astype(np.float64))
            if err > worst[ref_var][0]:
                worst[ref_var] = (err, k)

    report = {name: f"{err:.2e} (snapshot {k})" for name, (err, k) in worst.items()}
    print(f"worst relative L2 per product ({ref_name}, {xp.__name__}): {report}")
    failed = {name: report[name] for name, (err, _) in worst.items() if err > rtol[name]}
    assert not failed, f"relative L2 per snapshot exceeds {rtol}: {failed}"
