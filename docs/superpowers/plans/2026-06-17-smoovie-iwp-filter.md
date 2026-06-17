# Per-pixel IWP–Kalman Filter for smoovie — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-pixel q=1 integrated-Wiener-process (IWP) Kalman whitening filter to the `smoovie` Holoscan pipeline as one new streaming operator, persisting a durable `<movie>.zarr` and rendering filtered-flux and normalised-innovation movies alongside the dirty one.

**Architecture:** Mirror the existing `HealpixDFTOperator` ↔ `utils/healpix_dft.py` split: xp-generic numerical kernels in `utils/iwp.py` (numpy on CI, cupy in the operator), wrapped by a thin GPU operator `operators/iwp_kalman.py` that holds `(X, P, t_prev)` as attributes. Insert it as `reader → imager → iwp → writer`; the writer grows two output variables; `core/smoovie.py` writes a durable zarr (fail-fast on overwrite) and renders three movies.

**Tech Stack:** Python 3.10+, NumPy, CuPy, Holoscan, xarray/dask/zarr, healpy, matplotlib, Typer + hip-cargo, pytest.

**Spec:** `docs/superpowers/specs/2026-06-17-smoovie-iwp-filter-design.md`

**Dev workflow (run after every code change):** `uv run ruff format . && uv run ruff check . --fix`

---

### Task 1: IWP transition + Kalman recursion kernels (`utils/iwp.py`)

xp-generic, numpy-testable on CI. No GPU, no holoscan.

**Files:**
- Create: `src/kremetart/utils/iwp.py`
- Test: `tests/test_iwp.py`

- [ ] **Step 1: Write the failing tests (closed forms + Joseph PSD)**

Create `tests/test_iwp.py`:

```python
"""CPU tests for the per-pixel IWP-Kalman recursion (utils.iwp), xp=numpy.

The GPU operator (operators/iwp_kalman.py) wraps these same functions with xp=cupy; here we pin
the q=1 closed forms (eq. AQ of the design note), the Joseph-form covariance staying symmetric
PSD over many steps, and the whitening property (Task 2).
"""

import numpy as np

from kremetart.utils.iwp import iwp_transition, kalman_predict, kalman_update


def test_iwp_transition_closed_form():
    A, Q = iwp_transition(2.0, 3.0)
    # A = [[1, dt],[0, 1]]
    np.testing.assert_allclose(A, [[1.0, 2.0], [0.0, 1.0]])
    # Q = sigma2 * [[dt^3/3, dt^2/2],[dt^2/2, dt]] with sigma2=3, dt=2 -> [[8,6],[6,6]]
    np.testing.assert_allclose(Q, [[8.0, 6.0], [6.0, 6.0]])


def test_kalman_update_keeps_covariance_symmetric_psd():
    rng = np.random.default_rng(0)
    npix = 5
    X = rng.standard_normal((npix, 2))
    P = np.broadcast_to(np.eye(2) * 10.0, (npix, 2, 2)).copy()
    A, Q = iwp_transition(1.5, 0.5)
    for _ in range(50):
        X, P = kalman_predict(X, P, A, Q)
        y = X[:, 0] + rng.standard_normal(npix) * 0.1
        X, P, e, S = kalman_update(X, P, y, 0.01)
    assert np.all(S > 0)
    for p in range(npix):
        np.testing.assert_allclose(P[p], P[p].T, atol=1e-9)         # symmetric
        assert np.all(np.linalg.eigvalsh(P[p]) >= -1e-9)            # PSD
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_iwp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kremetart.utils.iwp'`.

- [ ] **Step 3: Write `src/kremetart/utils/iwp.py`**

```python
"""Integrated-Wiener-process (IWP) state-space model and the per-pixel Kalman recursion.

The quiescent prior of the kremetart design note (sec:iwp, sec:kf): each pixel's light curve is
a q=1 integrated Wiener process observed in noise, and the Kalman filter whitens it. All
functions are ``xp``-injectable -- pass ``xp=numpy`` (CPU tests) or ``xp=cupy`` (the GPU operator)
-- and vectorise over the leading pixel axis. See
docs/superpowers/specs/2026-06-17-smoovie-iwp-filter-design.md.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def iwp_transition(dt: float, sigma2: float, *, xp: ModuleType = np):
    """Exact q=1 IWP discrete transition matrices for an interval ``dt`` (eq. AQ).

    Args:
        dt: inter-frame interval Delta (seconds), read from the timestamp stream every frame
            (no constant-step assumption -- the design note's hard requirement).
        sigma2: scalar driving variance sigma^2.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(A, Q)``: the ``(2, 2)`` transition matrix ``A(dt)`` and process-noise covariance
        ``Q(dt)``.
    """
    dt = float(dt)
    A = xp.asarray([[1.0, dt], [0.0, 1.0]])
    Q = sigma2 * xp.asarray([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    return A, Q


def kalman_predict(X, P, A, Q, *, xp: ModuleType = np):
    """IWP predict step, vectorised over pixels.

    Args:
        X: ``(npix, 2)`` posterior means x_{k-1|k-1}.
        P: ``(npix, 2, 2)`` posterior covariances P_{k-1|k-1}.
        A: ``(2, 2)`` transition matrix.
        Q: ``(2, 2)`` process-noise covariance.
        xp: array module.

    Returns:
        ``(X_pred, P_pred)``: predicted means ``(npix, 2)`` and covariances ``(npix, 2, 2)``.
    """
    X_pred = X @ A.T
    P_pred = xp.einsum("ij,pjk,lk->pil", A, P, A) + Q
    return X_pred, P_pred


def kalman_update(X_pred, P_pred, y, R, *, xp: ModuleType = np):
    """IWP update step with scalar observation y = H x + v, H = (1, 0), Joseph form.

    Args:
        X_pred: ``(npix, 2)`` predicted means.
        P_pred: ``(npix, 2, 2)`` predicted covariances.
        y: ``(npix,)`` observations (dirty-map pixel values).
        R: scalar measurement-noise variance.
        xp: array module.

    Returns:
        ``(X_kk, P_kk, e, S)``: posterior means ``(npix, 2)``, posterior covariances
        ``(npix, 2, 2)``, innovations ``(npix,)`` and innovation variances ``(npix,)``.
    """
    npix = X_pred.shape[0]
    e = y - X_pred[:, 0]                       # innovation (npix,)
    S = P_pred[:, 0, 0] + R                     # innovation variance (npix,)
    K = P_pred[:, :, 0] / S[:, None]            # gain (npix, 2): P_pred @ H^T is column 0
    X_kk = X_pred + K * e[:, None]
    # Joseph form: (I - K H) P_pred (I - K H)^T + K R K^T, with H = (1, 0).
    eye = xp.broadcast_to(xp.eye(2), (npix, 2, 2))
    KH = K[:, :, None] * xp.asarray([1.0, 0.0])[None, None, :]   # (npix, 2, 2)
    ImKH = eye - KH
    P_kk = xp.einsum("pij,pjk,plk->pil", ImKH, P_pred, ImKH) + R * (K[:, :, None] * K[:, None, :])
    return X_kk, P_kk, e, S
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_iwp.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format + lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/kremetart/utils/iwp.py tests/test_iwp.py
git commit -m "feat: add xp-generic IWP transition and Kalman recursion kernels"
```

---

### Task 2: Whitening / NIS correctness test (synthetic IWP)

Validates the kernels from Task 1 on data drawn from the model: innovations are white and the
time-averaged NIS sits near 1 (sec:nis). Pure numpy; no new implementation.

**Files:**
- Test: `tests/test_iwp.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_iwp.py`:

```python
def test_innovations_whiten_on_synthetic_iwp():
    """Data from the q=1 IWP + Gaussian-noise model -> normalised innovations ~ N(0,1):
    mean ~ 0 and mean NIS (E[z^2]) ~ 1 after a warm-up."""
    rng = np.random.default_rng(42)
    npix, n, dt = 200, 400, 1.0
    sigma2, R = 0.05, 0.2
    A, Q = iwp_transition(dt, sigma2)

    # Simulate true IWP states and noisy flux observations per pixel.
    Lq = np.linalg.cholesky(Q)
    x = np.zeros((npix, 2))
    Y = np.zeros((npix, n))
    for k in range(n):
        x = x @ A.T + rng.standard_normal((npix, 2)) @ Lq.T
        Y[:, k] = x[:, 0] + rng.standard_normal(npix) * np.sqrt(R)

    # Run the filter from a diffuse prior; collect normalised innovations after warm-up.
    X = np.zeros((npix, 2))
    P = np.broadcast_to(np.eye(2) * 1e6, (npix, 2, 2)).copy()
    z = []
    for k in range(n):
        X, P = kalman_predict(X, P, A, Q)
        X, P, e, S = kalman_update(X, P, Y[:, k], R)
        z.append(e / np.sqrt(S))
    z = np.asarray(z[50:])                      # drop warm-up samples

    assert abs(float(z.mean())) < 0.05
    assert 0.9 < float((z**2).mean()) < 1.1     # mean NIS ~ chi^2_1 mean = 1
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_iwp.py::test_innovations_whiten_on_synthetic_iwp -v`
Expected: PASS (the kernels already exist from Task 1).

- [ ] **Step 3: Commit**

```bash
git add tests/test_iwp.py
git commit -m "test: IWP innovations whiten with mean NIS ~ 1 on synthetic data"
```

---

### Task 3: GPU operator `IWPKalmanOperator` (`operators/iwp_kalman.py`)

Thin GPU operator holding `(X, P, t_prev)` as attributes; wraps Task 1's kernels with `xp=cp`.
`cupy`/`holoscan` at module top (inherently GPU-only, like `dft_healpix.py`). Functional
behaviour is validated by the e2e test (Task 7); here we add an import guard.

**Files:**
- Create: `src/kremetart/operators/iwp_kalman.py`
- Test: `tests/test_smoovie.py:63-67` (extend `test_gpu_operators_import`)

- [ ] **Step 1: Extend the failing import test**

In `tests/test_smoovie.py`, replace `test_gpu_operators_import` (currently lines 63-67):

```python
def test_gpu_operators_import():
    from kremetart.operators.dft_healpix import HealpixDFTOperator
    from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator
    from kremetart.operators.iwp_kalman import IWPKalmanOperator

    assert HealpixDFTOperator and HealpixZarrReaderOperator and HealpixWriterOperator
    assert IWPKalmanOperator
```

- [ ] **Step 2: Run test to verify it fails (on a GPU box)**

Run: `uv run pytest tests/test_smoovie.py::test_gpu_operators_import -v`
Expected: on a GPU box, FAIL — `ModuleNotFoundError: kremetart.operators.iwp_kalman`. On a CPU-only CI runner this test SKIPS (module-level `pytestmark` gate) — that is expected; the operator runs only on GPU.

- [ ] **Step 3: Write `src/kremetart/operators/iwp_kalman.py`**

```python
"""Holoscan operator: per-pixel q=1 IWP-Kalman whitening filter (GPU-resident, xp=cupy).

Holds the Kalman state (means ``X``, covariances ``P``, previous timestamp ``t_prev``) as
attributes; consumes the imager's per-frame dirty map (the observation) and timestamp, runs the
exact IWP predict+update (kremetart.utils.iwp) with the per-frame Delta from the timestamp
stream, and emits the dirty map (passthrough), the filtered flux x_{k|k}[0] and the normalised
innovation z_k. See docs/superpowers/specs/2026-06-17-smoovie-iwp-filter-design.md.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.utils.iwp import iwp_transition, kalman_predict, kalman_update

_DIFFUSE = 1e6  # diffuse-prior variance for the frame-0 warm-up


class IWPKalmanOperator(Operator):
    """Per-pixel q=1 IWP-Kalman filter.

    Args:
        fragment: Holoscan fragment.
        npix: number of HEALPix pixels (independent filters).
        sigma2: IWP driving variance sigma^2.
        noise: scalar measurement-noise variance R.
    """

    def __init__(self, fragment, npix, *args, sigma2, noise, **kwargs):
        self.npix = npix
        self.sigma2 = float(sigma2)
        self.noise = float(noise)
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Diffuse prior: zero mean, large covariance. Frame 0 runs update-only (no Delta yet).
        self.X = cp.zeros((self.npix, 2))
        self.P = cp.broadcast_to(cp.eye(2) * _DIFFUSE, (self.npix, 2, 2)).copy()
        self.t_prev = None

    def setup(self, spec: OperatorSpec):
        spec.input("cube")
        spec.input("time_out")
        spec.output("cube")
        spec.output("filtered")
        spec.output("znorm")
        spec.output("time_out")

    def compute(self, op_input, op_output, context):
        cube = cp.asarray(op_input.receive("cube"))      # (1, npix)
        time_out = cp.asarray(op_input.receive("time_out"))  # (1,)
        y = cube[0]                                       # (npix,)
        t = float(time_out[0])

        if self.t_prev is not None:
            A, Q = iwp_transition(t - self.t_prev, self.sigma2, xp=cp)
            self.X, self.P = kalman_predict(self.X, self.P, A, Q, xp=cp)

        self.X, self.P, e, S = kalman_update(self.X, self.P, y, self.noise, xp=cp)
        self.t_prev = t

        filtered = self.X[:, 0]
        znorm = e / cp.sqrt(S)

        op_output.emit(hs.as_tensor(cube), "cube")                 # passthrough dirty map
        op_output.emit(hs.as_tensor(filtered[None, :]), "filtered")
        op_output.emit(hs.as_tensor(znorm[None, :]), "znorm")
        op_output.emit(hs.as_tensor(time_out), "time_out")
```

- [ ] **Step 4: Run test to verify it passes (GPU box)**

Run: `uv run pytest tests/test_smoovie.py::test_gpu_operators_import -v`
Expected: PASS on GPU; SKIP on CPU CI.

- [ ] **Step 5: Format + lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/kremetart/operators/iwp_kalman.py tests/test_smoovie.py
git commit -m "feat: add per-pixel IWP-Kalman Holoscan operator"
```

---

### Task 4: Writer emits `dirty` / `filtered` / `znorm` (`operators/io.py`)

`HealpixWriterOperator` gains two more `(TIME, PIX)` variables and two more inputs. GPU-gated;
validated by the e2e test (Task 7).

**Files:**
- Modify: `src/kremetart/operators/io.py` — `HealpixWriterOperator.start` (246-255), `.setup` (257-259), `.compute` (261-268)

- [ ] **Step 1: Replace `HealpixWriterOperator.start`**

Replace the body of `start` (lines 246-255):

```python
    def start(self):
        # dask scaffold: da.empty allocates no data, only the zarr structure to write regions into.
        data_vars = {
            name: (("TIME", "PIX"), da.empty((self.ntime, self.npix), chunks=(1, self.npix), dtype=np.float32))
            for name in ("dirty", "filtered", "znorm")
        }
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"TIME": (("TIME",), self.out_times), "PIX": (("PIX",), self.pix)},
        )
        ds.to_zarr(self.output_dataset, mode="w", compute=True)
```

- [ ] **Step 2: Replace `HealpixWriterOperator.setup`**

Replace `setup` (lines 257-259):

```python
    def setup(self, spec: OperatorSpec):
        spec.input("cube")
        spec.input("filtered")
        spec.input("znorm")
        spec.input("time_out")
```

- [ ] **Step 3: Replace `HealpixWriterOperator.compute`**

Replace `compute` (lines 261-268), dropping the stale trailing comment:

```python
    def compute(self, op_input, op_output, context):
        cube = cp.asnumpy(cp.asarray(op_input.receive("cube")))          # (1, npix)
        filtered = cp.asnumpy(cp.asarray(op_input.receive("filtered")))  # (1, npix)
        znorm = cp.asnumpy(cp.asarray(op_input.receive("znorm")))        # (1, npix)
        time_out = cp.asnumpy(cp.asarray(op_input.receive("time_out")))  # (1,)
        dso = xr.Dataset(
            data_vars={
                "dirty": (("TIME", "PIX"), cube.astype(np.float32)),
                "filtered": (("TIME", "PIX"), filtered.astype(np.float32)),
                "znorm": (("TIME", "PIX"), znorm.astype(np.float32)),
            },
            coords={"TIME": (("TIME",), time_out), "PIX": (("PIX",), self.pix)},
        )
        dso.to_zarr(self.output_dataset, region="auto")
```

- [ ] **Step 4: Format + lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/kremetart/operators/io.py
git commit -m "feat: HealpixWriterOperator writes dirty/filtered/znorm variables"
```

---

### Task 5: Wire the filter into `core/smoovie.py` (pipeline, durable zarr, 3 movies)

The host orchestration. After this task the GPU pipeline produces `<movie>.zarr` with three
variables and three mp4s; the core `smoovie()` signature gains `iwp_sigma`, `iwp_noise`,
`overwrite` (the cli wrapper follows in Task 6). The CPU host-wiring tests are updated to the new
`image_via_app` return shape and `render_frames` signature.

**Files:**
- Modify: `src/kremetart/core/smoovie.py` — top import (28), `render_frames` (171-202), `SmooviePipeline` (37-82), `image_via_app` (85-138), `smoovie` (205-295)
- Modify: `tests/test_smoovie.py` — `sm_stub` (48-57), `fake_render` signatures (148, 167, 184, 205), `test_smoovie_nframes_flows_to_imaging` (215-227)

- [ ] **Step 1: Import the operator at module top**

In `src/kremetart/core/smoovie.py`, replace line 28:

```python
from kremetart.operators.dft_healpix import HealpixDFTOperator
```

with:

```python
from kremetart.operators.dft_healpix import HealpixDFTOperator
from kremetart.operators.iwp_kalman import IWPKalmanOperator
```

- [ ] **Step 2: Add `diverging` mode to `render_frames`**

Replace the signature line (171-181) — add the `diverging` keyword:

```python
def render_frames(
    maps,
    timestamps,
    nside: int,
    cmap: str,
    outdir,
    *,
    rot: tuple[float, float] | None = None,
    nest: bool = True,
    tracks=None,
    diverging: bool = False,
):
```

Then replace the vmin/vmax computation + the `hp.mollview` call (currently 189-194):

```python
    outdir = Path(outdir)
    stacked = np.concatenate([np.asarray(m) for m in maps])
    if diverging:
        # Symmetric scale centred on 0 with a diverging cmap (for the normalised innovation z_k).
        vmax = float(np.percentile(np.abs(stacked), 99.0))
        vmin, cmap = -vmax, "coolwarm"
    else:
        vmin, vmax = (float(v) for v in np.percentile(stacked, [1.0, 99.0]))
    paths = []
    for i, (m, ts) in enumerate(zip(maps, timestamps)):
        hp.mollview(np.asarray(m), nest=nest, title=ts, cmap=cmap, min=vmin, max=vmax, rot=rot)
```

(The docstring above it is unchanged.)

- [ ] **Step 3: Update `SmooviePipeline` for the IWP knobs + wiring**

Replace `SmooviePipeline.__init__` (40-59) so it accepts `sigma2` / `noise`:

```python
    def __init__(
        self,
        prepared_zarr: Path | str,
        output_zarr: Path | str,
        nside: int,
        *args,
        nest: bool = True,
        sigma2: float = 1e-3,
        noise: float = 1e-2,
        **kwargs,
    ):
        self.prepared_zarr = str(prepared_zarr)
        self.output_zarr = str(output_zarr)
        self.nside = nside
        self.nest = nest
        self.sigma2 = sigma2
        self.noise = noise
        super().__init__(*args, **kwargs)

        ds = xr.open_zarr(self.prepared_zarr)
        self.ntime = int(ds.time.size)
        self.out_times = ds.time.values
        self.freqs = ds.frequency.values
        self.npix = hp.nside2npix(nside)
```

Replace `compose` (61-82) to insert the operator as `imager → iwp → writer`:

```python
    def compose(self):
        reader = HealpixZarrReaderOperator(
            self,
            CountCondition(self, self.ntime),
            name="reader",
            zarr_path=self.prepared_zarr,
        )
        imager = HealpixDFTOperator(self, self.nside, self.freqs, name="imager", nest=self.nest)
        iwp = IWPKalmanOperator(self, self.npix, name="iwp", sigma2=self.sigma2, noise=self.noise)
        writer = HealpixWriterOperator(
            self,
            self.ntime,
            self.npix,
            name="writer",
            output_dataset=self.output_zarr,
            out_times=self.out_times,
        )
        self.add_flow(
            reader,
            imager,
            {("VISIBILITY", "VISIBILITY"), ("WEIGHT", "WEIGHT"), ("B_ROT", "B_ROT"), ("time", "time")},
        )
        self.add_flow(imager, iwp, {("cube", "cube"), ("time_out", "time_out")})
        self.add_flow(
            iwp,
            writer,
            {("cube", "cube"), ("filtered", "filtered"), ("znorm", "znorm"), ("time_out", "time_out")},
        )
```

- [ ] **Step 4: Rewrite `image_via_app` (durable zarr, 4-tuple return)**

Replace the whole `image_via_app` function (85-138):

```python
def image_via_app(
    hdf_paths,
    nside: int,
    *,
    output_zarr,
    overwrite: bool = False,
    correct_gains: bool = False,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    nframes: int | None = None,
    nest: bool = True,
    iwp_sigma: float = 1e-3,
    iwp_noise: float = 1e-2,
):
    """Image the HDF sequence through the GPU Holoscan app; return ``(dirty, filtered, znorm, stamps)``.

    Runs the host prepare-step into a temp zarr, streams it through :class:`SmooviePipeline` (imager
    -> per-pixel IWP-Kalman filter -> writer), and writes a DURABLE ``output_zarr`` holding the
    ``(TIME, PIX)`` ``dirty`` / ``filtered`` / ``znorm`` variables, left in place for inspection.
    Loads each variable back to host as a list of ``(npix,)`` maps (one per frame, in order) plus a
    list of UTC stamp strings. This is the single imaging seam :func:`smoovie` drives; the
    host-wiring tests stub it to exercise orchestration without running the GPU.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        output_zarr: durable output zarr path (overwritten only if it does not already exist; the
            caller is responsible for the fail-fast existence check).
        overwrite: reserved for symmetry with the caller; the writer always writes ``mode="w"``.
        correct_gains: apply the inverse per-antenna gain solution in the prepare-step.
        phase_ra_deg, phase_dec_deg: common phase direction (deg, ICRS), stored as zarr metadata.
        nframes: optional cap on the number of frames imaged.
        nest: NESTED HEALPix ordering (default True).
        iwp_sigma: IWP driving variance sigma^2.
        iwp_noise: measurement-noise variance R.

    Returns:
        ``(dirty, filtered, znorm, stamps)``.
    """
    output = Path(output_zarr)
    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "prepared.zarr"
        config = Path(td) / "config.yaml"
        config.touch()  # an empty Holoscan config is valid

        prepare_msv4_zarr(
            hdf_paths,
            prepared,
            correct_gains=correct_gains,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            nframes=nframes,
        )

        app = SmooviePipeline(prepared, output, nside, nest=nest, sigma2=iwp_sigma, noise=iwp_noise)
        app.config(str(config))
        app.run()

        ds = xr.open_zarr(str(output), chunks=None)  # eager load
        dirty = np.asarray(ds["dirty"].values)        # (ntime, npix)
        filtered = np.asarray(ds["filtered"].values)
        znorm = np.asarray(ds["znorm"].values)
        times = np.asarray(ds["TIME"].values)

    stamps = [unix_to_utc(t) for t in times]
    return list(dirty), list(filtered), list(znorm), stamps
```

- [ ] **Step 5: Add the `_encode_movie` helper (DRY the ffmpeg call)**

Insert this module-level helper just above `def smoovie(` (before line 205):

```python
def _encode_movie(first_png, fps: int, out) -> None:
    """Encode the ``frame_%04d.png`` sequence in ``first_png``'s directory to mp4 ``out``."""
    pattern = str(Path(first_png).parent / "frame_%04d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
```

- [ ] **Step 6: Update `smoovie()` — params, fail-fast, 3-movie render**

Replace the `smoovie` signature (205-219) to add the three params (insert before `nframes`):

```python
def smoovie(
    hdf_dir,
    movie,
    nside: int = 128,
    fps: int = 2,
    cmap: str = "inferno",
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    correct_gains: bool = False,
    overlay_catalog: bool = False,
    catalog_elevation_deg: float = 45.0,
    catalog_cache: str | None = None,
    profile: bool = False,
    iwp_sigma: float = 1e-3,
    iwp_noise: float = 1e-2,
    overwrite: bool = False,
    nframes: int | None = None,
):
```

Replace the imaging `stage_timer` block (251-260) with the fail-fast guard + new call:

```python
    output_zarr = Path(str(movie) + ".zarr")
    if output_zarr.exists() and not overwrite:
        raise FileExistsError(f"{output_zarr} already exists; pass overwrite=True to replace it.")

    timings = []
    with stage_timer("imaging", timings):
        dirty, filtered, znorm, stamps = image_via_app(
            hdf_paths,
            nside,
            output_zarr=output_zarr,
            overwrite=overwrite,
            correct_gains=correct_gains,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            nframes=nframes,
            iwp_sigma=iwp_sigma,
            iwp_noise=iwp_noise,
        )
```

(Delete the pre-existing `timings = []` line at 251 if it now duplicates the one above — keep exactly one.)

Replace the render/encode block (267-290) with the three-movie loop:

```python
    movie_specs = [
        (dirty, Path(movie), False),
        (filtered, Path(str(movie) + ".filtered.mp4"), False),
        (znorm, Path(str(movie) + ".znorm.mp4"), True),
    ]
    with tempfile.TemporaryDirectory() as td:
        with stage_timer("render", timings):
            rendered = []
            for frames, out, diverging in movie_specs:
                subdir = Path(td) / out.name
                subdir.mkdir()
                pngs = render_frames(
                    frames, stamps, nside, cmap, subdir, rot=(phase_ra_deg, phase_dec_deg),
                    tracks=tracks, diverging=diverging,
                )
                rendered.append((pngs, out))
        with stage_timer("encode", timings):
            for pngs, out in rendered:
                _encode_movie(pngs[0], fps, out)
```

Finally, replace the profile line (292-293) so the frame count comes from `dirty`:

```python
    if profile:
        print_profile(timings, nframes=len(dirty))
```

- [ ] **Step 7: Update the CPU host-wiring test stubs**

In `tests/test_smoovie.py`, replace the `sm_stub` fixture body line (54) so the stub returns the
4-tuple and tolerates the new kwargs:

```python
    monkeypatch.setattr(
        sm, "image_via_app", lambda paths, nside, **k: ([np.zeros(12)], [np.zeros(12)], [np.zeros(12)], ["t UTC"])
    )
```

Add `diverging=False` to every local `fake_render` signature (the four functions at lines 148,
167, 184, 205), e.g.:

```python
    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None, diverging=False):
```

And in `test_smoovie_nframes_flows_to_imaging`, replace `fake_image` (218-220) so it returns the
4-tuple:

```python
    def fake_image(paths, nside, **k):
        captured.update(k)
        return ([np.zeros(12)], [np.zeros(12)], [np.zeros(12)], ["t UTC"])
```

- [ ] **Step 8: Run the CPU host-wiring tests**

Run: `uv run pytest tests/test_smoovie.py -v -k "smoovie and not produces_movie"`
Expected: on CPU CI these SKIP (GPU gate). On a GPU box the host-wiring tests PASS (they exercise
`smoovie()` orchestration with imaging/encode stubbed). If running CPU-only, rely on Task 7's
e2e + the format/lint gate here.

- [ ] **Step 9: Format + lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors.

- [ ] **Step 10: Commit**

```bash
git add src/kremetart/core/smoovie.py tests/test_smoovie.py
git commit -m "feat: wire per-pixel IWP filter into smoovie with durable zarr and 3 movies"
```

---

### Task 6: Expose the knobs on the CLI wrapper + regenerate the cab

`cli/smoovie.py` gains the same three parameters (the mirror rule: `core.smoovie` params ==
`cli.smoovie` params minus `{backend, always_pull_images}`). The cab regenerates via the
pre-commit hook. **Round-trip safety:** write the float defaults exactly as hip-cargo's
`generate-function` emits them — plain decimals `0.001` / `0.01`, not `1e-3` / `1e-2` — or the
round-trip byte-comparison fails.

**Files:**
- Modify: `src/kremetart/cli/smoovie.py` — insert params after the `profile` block (88) and before `nframes` (89); add to the preflight dict (131-145), the `smoovie_core(...)` call (152-166), and the `run_in_container` dict (182-196).
- Auto-regenerated: `src/kremetart/cabs/smoovie.yml`

- [ ] **Step 1: Insert the three parameters into the signature**

In `src/kremetart/cli/smoovie.py`, insert immediately after the `profile` parameter block (the
line `    ] = False,` at 88) and before `    nframes:` (89):

```python
    iwp_sigma: Annotated[
        float,
        typer.Option(
            help="IWP driving variance (sigma^2) for the per-pixel Kalman filter.",
        ),
    ] = 0.001,
    iwp_noise: Annotated[
        float,
        typer.Option(
            help="Measurement-noise variance (R) for the per-pixel Kalman filter.",
        ),
    ] = 0.01,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite the output <movie>.zarr if it already exists.",
        ),
    ] = False,
```

- [ ] **Step 2: Thread the params through all three call sites**

In each of the three `dict(...)` / call blocks (preflight `dict(`, the `smoovie_core(` call, and
the `run_in_container` `dict(`), add these three lines immediately before the `nframes=nframes,`
line in that block:

```python
            iwp_sigma=iwp_sigma,
            iwp_noise=iwp_noise,
            overwrite=overwrite,
```

(There are three such blocks; update all three. Indentation: the preflight and run_in_container
dicts are nested one extra level — match the surrounding `nframes=nframes,` indentation in each
block exactly.)

- [ ] **Step 3: Format + lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors.

- [ ] **Step 4: Regenerate the cab via pre-commit (do NOT run generate-cabs by hand)**

Stage and commit; the pre-commit hook runs `generate-cabs` and may rewrite `cabs/smoovie.yml`.
If pre-commit reports a modification, re-add and re-commit so the regenerated cab is included:

```bash
git add src/kremetart/cli/smoovie.py
git commit -m "feat: expose iwp_sigma/iwp_noise/overwrite on the smoovie CLI"
# if pre-commit modified cabs/smoovie.yml:
git add -u && git commit -m "feat: expose iwp_sigma/iwp_noise/overwrite on the smoovie CLI"
```

> Heads-up (architecture.md §4): activate the project venv so `kremetart` is importable before
> committing, or the regenerated cab loses its `image:` field.

- [ ] **Step 5: Verify round-trip + structure**

Run: `uv run pytest tests/test_roundtrip.py tests/test_structure.py -v`
Expected: PASS. If `test_roundtrip_smoovie` fails on a default line, make the cli literal match
the regenerated file byte-for-byte (the generator output is the source of truth for formatting),
then re-commit and re-run.

---

### Task 7: GPU end-to-end + overwrite tests

**Files:**
- Modify: `tests/test_smoovie.py` — `test_image_via_app_end_to_end` (70-81), `test_smoovie_produces_movie` (261-279); add `test_smoovie_overwrite_fail_fast`.

- [ ] **Step 1: Update `test_image_via_app_end_to_end` for the durable zarr + 4-tuple**

Replace it (70-81):

```python
def test_image_via_app_end_to_end(hdf_paths, tmp_path):
    """The real Holoscan app images + filters one frame per sub-integration into finite maps."""
    from kremetart.core.smoovie import image_via_app

    nside = 8
    npix = 12 * nside * nside
    out = tmp_path / "imaging.zarr"
    dirty, filtered, znorm, stamps = image_via_app(
        hdf_paths[:1], nside, output_zarr=out, correct_gains=True, nframes=3
    )
    assert len(dirty) == len(filtered) == len(znorm) == len(stamps) == 3
    for m in (*dirty, *filtered, *znorm):
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
    assert out.exists()              # durable: left in place for inspection
    assert "UTC" in stamps[0]
```

- [ ] **Step 2: Assert the extra movies + zarr in the full e2e**

In `test_smoovie_produces_movie`, replace the final assertion (279):

```python
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / "movie.mp4.filtered.mp4").exists()
    assert (tmp_path / "movie.mp4.znorm.mp4").exists()
    assert (tmp_path / "movie.mp4.zarr").exists()
```

- [ ] **Step 3: Add the overwrite fail-fast test**

Append to `tests/test_smoovie.py` (host-wiring style; uses `sm_stub`, so no GPU pipeline runs
beyond the gate):

```python
def test_smoovie_overwrite_fail_fast(tmp_path, hdf_dir, sm_stub, monkeypatch):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm_stub, "render_frames", lambda *a, **k: [Path("frame_0000.png")])
    movie = tmp_path / "m.mp4"
    (tmp_path / "m.mp4.zarr").mkdir()  # pre-existing output

    with pytest.raises(FileExistsError):
        sm_stub.smoovie(hdf_dir=hdf_dir, movie=movie, nside=1)

    # With overwrite set, the guard passes and orchestration proceeds (imaging/encode stubbed).
    sm_stub.smoovie(hdf_dir=hdf_dir, movie=movie, nside=1, overwrite=True)
```

- [ ] **Step 4: Run the GPU suite (on a GPU box)**

Run: `uv run pytest tests/test_smoovie.py -v`
Expected: on a GPU box, all PASS (raise the stack limit first if the holoscan import segfaults:
`ulimit -s 32768`). On CPU CI the whole module SKIPS — expected.

- [ ] **Step 5: Run the full CPU-safe suite**

Run: `uv run pytest tests/test_iwp.py tests/test_roundtrip.py tests/test_structure.py -v`
Expected: PASS (these need no GPU).

- [ ] **Step 6: Commit**

```bash
git add tests/test_smoovie.py
git commit -m "test: GPU e2e for filtered/znorm outputs and overwrite fail-fast"
```

---

## Self-Review

**Spec coverage** (each spec section → task):
- §2 per-pixel / q=1 / Joseph / outputs → Tasks 1–3.
- §3 math (AQ, predict/update, diffuse-prior frame-0 warm-up, irregular Δ) → Task 1 kernels + Task 3 operator (Δ from timestamp stream; `t_prev=None` skips frame-0 predict).
- §4 module split (`utils/iwp.py` ↔ `operators/iwp_kalman.py`) → Tasks 1, 3.
- §5 app/writer/renderer (durable zarr, fail-fast, 3 movies, diverging mode) → Tasks 4, 5.
- §6 CLI/cab knobs + mirror rule → Tasks 5 (core) + 6 (cli/cab).
- §7 tests (CPU whitening/NIS, GPU e2e, overwrite) → Tasks 2, 7.
- §8 out-of-scope items → not implemented (correct).

**Type/name consistency:** `iwp_transition`/`kalman_predict`/`kalman_update` signatures and the
`(X_kk, P_kk, e, S)` / `(dirty, filtered, znorm, stamps)` tuples are used identically across
Tasks 1, 3, 5, 7. Operator I/O ports (`cube`/`filtered`/`znorm`/`time_out`) match the writer
inputs (Task 4) and the `add_flow` edges (Task 5). CLI params (`iwp_sigma`/`iwp_noise`/`overwrite`)
match core (Task 5) so `test_structure.py` passes.

**Placeholder scan:** none — every code step shows complete code; the only deliberate
verification loop is the round-trip default-literal check (Task 6 Step 5).
