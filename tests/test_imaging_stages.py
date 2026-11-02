"""The imaging stages, exercised through their interfaces on synthetic snapshots (CPU; GPU agreement opt-in)."""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

from kremetart.imaging.dirty_imager import DirtyImager
from kremetart.imaging.healpix import dft_forward, hessian_healpix
from kremetart.imaging.iwp_filter import IWPFilter
from kremetart.imaging.l1_solver import L1Solver
from kremetart.imaging.sky_grid import SkyGrid
from kremetart.imaging.snapshot import Snapshot
from kremetart.imaging.snapshot_imager import SnapshotImager
from kremetart.imaging.tikhonov_solver import TikhonovSolver

NSIDE = 8
FREQS = np.array([1.57542e9])
ZENITH = np.array([0.0, 0.0, 1.0])


def _source_pixel(grid: SkyGrid) -> int:
    """A pixel near (not at) the boresight, so the beam does not zero it."""
    cos_zenith = grid.pix_vec @ ZENITH
    return int(np.argmin(np.abs(cos_zenith - 0.95)))


def _snapshot(grid: SkyGrid, time: float, *, flux: float = 1.0, seed: int = 0, flagged: bool = False) -> Snapshot:
    """A point source of ``flux`` observed by 40 random baselines, beam applied as the grid says."""
    rng = np.random.default_rng(seed)
    b_rot = rng.uniform(-3.0, 3.0, size=(40, 3))
    sky = np.zeros(grid.npix)
    sky[_source_pixel(grid)] = flux
    vis = dft_forward(sky, b_rot, grid.pix_vec, FREQS, beam=grid.beam(ZENITH, FREQS))
    weight = np.zeros(vis.shape) if flagged else np.ones(vis.shape)
    return Snapshot(vis=vis, weight=weight, b_rot=b_rot, boresight=ZENITH, time=time, freqs=FREQS)


@pytest.fixture
def grid() -> SkyGrid:
    return SkyGrid.build(NSIDE)


# --- dirty imager -------------------------------------------------------------------------------


def test_dirty_imager_peaks_at_the_source(grid: SkyGrid) -> None:
    dirty = DirtyImager(grid).image(_snapshot(grid, 0.0))
    assert dirty.shape == (grid.npix,)
    assert int(np.argmax(dirty)) == _source_pixel(grid)


def test_dirty_imager_returns_zeros_for_a_snapshot_without_data(grid: SkyGrid) -> None:
    dirty = DirtyImager(grid).image(_snapshot(grid, 0.0, flagged=True))
    assert np.all(dirty == 0.0)


# --- solvers ------------------------------------------------------------------------------------


def test_tikhonov_solver_solves_the_regularised_normal_equations(grid: SkyGrid) -> None:
    snap = _snapshot(grid, 0.0)
    dirty = DirtyImager(grid).image(snap)
    eta = 0.05
    x = TikhonovSolver(grid, eta=eta, tol=1e-10, maxiter=500).solve(snap, dirty)

    wsum = snap.total_weight()
    hmv, _ = hessian_healpix(snap.b_rot, grid.pix_vec, FREQS, snap.weight, beam=grid.beam(ZENITH, FREQS))
    residual = hmv(x) + eta * wsum * x - dirty * wsum
    assert np.linalg.norm(residual) <= 1e-8 * np.linalg.norm(dirty * wsum)


@pytest.mark.parametrize("make_solver", [lambda g: TikhonovSolver(g, eta=0.05), lambda g: L1Solver(g, eta=0.05)])
def test_solver_returns_zeros_for_a_snapshot_without_data(
    grid: SkyGrid, make_solver: Callable[[SkyGrid], TikhonovSolver | L1Solver]
) -> None:
    snap = _snapshot(grid, 0.0, flagged=True)
    x = make_solver(grid).solve(snap, DirtyImager(grid).image(snap))
    assert x.shape == (grid.npix,) and np.all(x == 0.0)


@pytest.mark.parametrize(
    "make_solver",
    [lambda g: TikhonovSolver(g, eta=0.05, maxiter=5), lambda g: L1Solver(g, eta=0.05, max_iter=5, max_reweight=0)],
    ids=["tikhonov", "l1"],
)
def test_snapshot_without_data_leaves_the_warm_start_untouched(
    grid: SkyGrid, make_solver: Callable[[SkyGrid], TikhonovSolver | L1Solver]
) -> None:
    """[s1, gap, s2] gives the same s2 map as [s1, s2]: the gap neither seeds nor resets the next solve."""
    imager = DirtyImager(grid)
    s1, gap, s2 = _snapshot(grid, 0.0, seed=1), _snapshot(grid, 1.0, flagged=True), _snapshot(grid, 2.0, seed=2)

    with_gap = make_solver(grid)
    for snap in (s1, gap):
        with_gap.solve(snap, imager.image(snap))
    without_gap = make_solver(grid)
    without_gap.solve(s1, imager.image(s1))

    np.testing.assert_array_equal(with_gap.solve(s2, imager.image(s2)), without_gap.solve(s2, imager.image(s2)))


def test_warm_start_changes_a_budget_limited_solve(grid: SkyGrid) -> None:
    """With a tight iteration budget, starting from the previous solution lands somewhere else than cold."""
    imager = DirtyImager(grid)
    s1, s2 = _snapshot(grid, 0.0, seed=1), _snapshot(grid, 1.0, seed=2)
    warm, cold = (
        TikhonovSolver(grid, eta=0.05, maxiter=2),
        TikhonovSolver(grid, eta=0.05, maxiter=2, use_warm_start=False),
    )
    for solver in (warm, cold):
        solver.solve(s1, imager.image(s1))
    assert not np.allclose(warm.solve(s2, imager.image(s2)), cold.solve(s2, imager.image(s2)))


def test_l1_solver_is_non_negative_and_reports_diagnostics(grid: SkyGrid) -> None:
    snap = _snapshot(grid, 0.0)
    solver = L1Solver(grid, eta=0.01, max_reweight=0)
    x = solver.solve(snap, DirtyImager(grid).image(snap))
    assert np.all(x >= 0.0) and x.max() > 0.0
    assert solver.last_info is not None and solver.last_info["reweights"] == 0


# --- IWP filter ---------------------------------------------------------------------------------


def test_iwp_filter_first_map_is_taken_almost_as_is() -> None:
    """A diffuse prior means the first map dominates the posterior."""
    y = np.linspace(-1.0, 1.0, 10)
    smooth, _ = IWPFilter(10, sigma2=1e-3, noise=1e-2, xp=np).update(y, 0.0)
    np.testing.assert_allclose(smooth, y, atol=1e-6)


def test_iwp_filter_coasts_over_a_map_without_data() -> None:
    filt = IWPFilter(4, sigma2=1e-3, noise=1e-2, xp=np)
    filt.update(np.ones(4), 0.0)
    filt.update(np.ones(4) * 1.1, 1.0)
    smooth, innovation = filt.update(np.zeros(4), 2.0)  # all-zero map = no data
    assert np.all(innovation == 0.0)
    assert np.all(smooth > 1.0)  # the trend carries on; zeros were not ingested


# --- snapshot imager ----------------------------------------------------------------------------


@pytest.mark.parametrize("l1", [None, 0.01])
def test_snapshot_imager_produces_every_map(l1: float | None) -> None:
    imager = SnapshotImager.build(NSIDE, l1=l1)
    grid = imager.dirty_imager.grid
    for k in range(3):
        maps = imager.image(_snapshot(grid, float(k), seed=k))
        assert maps.time == float(k)
        for name in ("dirty", "tikhonov", "smooth", "normalised_innovation"):
            got = getattr(maps, name)
            assert got.shape == (grid.npix,) and np.all(np.isfinite(got))
        assert (maps.l1 is None) == (l1 is None)


# --- NumPy / CuPy agreement ---------------------------------------------------------------------


def _cupy() -> ModuleType:
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    return cp


def test_snapshot_imager_agrees_on_numpy_and_cupy() -> None:
    """The same chain on both backends, with tight solver tolerances so the backends converge together."""
    cp = _cupy()

    def chain(xp: ModuleType) -> SnapshotImager:
        g = SkyGrid.build(NSIDE, xp=xp)
        return SnapshotImager(
            DirtyImager(g),
            TikhonovSolver(g, eta=0.05, tol=1e-12, maxiter=1000),
            IWPFilter(g.npix, sigma2=1e-3, noise=1e-2, xp=xp),
        )

    host_grid = SkyGrid.build(NSIDE)
    on_cpu, on_gpu = chain(np), chain(cp)
    for k in range(3):
        snap = _snapshot(host_grid, float(k), seed=k)
        dev = Snapshot(
            *(cp.asarray(a) for a in (snap.vis, snap.weight, snap.b_rot, snap.boresight)),
            snap.time,
            cp.asarray(snap.freqs),
        )
        a, b = on_cpu.image(snap), on_gpu.image(dev)
        for name in ("dirty", "tikhonov", "smooth", "normalised_innovation"):
            got, ref = getattr(b, name).get(), getattr(a, name)
            # Relative L2 per map: element-wise rtol is meaningless at pixels that are ~0.
            assert np.linalg.norm(got - ref) <= 1e-8 * np.linalg.norm(ref), name
