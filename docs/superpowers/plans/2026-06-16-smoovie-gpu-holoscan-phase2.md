# Smoovie GPU Holoscan (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `smoovie`'s imaging into a Holoscan streaming GPU application (mirroring `core/stream_msv4.py`): a host prepare-step turns the HDF sequence into one imaging-ready zarr (gains applied, per-frame rotated baselines `b_rot(t)` precomputed), a pure-cupy `HealpixDFTOperator` streams it onto a fixed equatorial HEALPix grid, a writer persists the `(TIME, npix)` dirty-map zarr, and rendering/ffmpeg run on the host after `app.run()`.

**Architecture:** The DFT math is split into a host frame-rotation (`equatorial_baselines`, already exists) and a device-only `image_frame_prerotated`; both the CPU `image_frame` and the GPU operator call the latter. A new host module `core/smoovie_prepare.py` writes the prepared zarr. New `operators/io.py` reader/writer classes stream that zarr (the existing `XarrayZarrReaderOperator`/`ResultWriterOperator` are left untouched — they serve `stream_msv4`). A GPU-only app module `core/smoovie_app.py` wires reader → imager → writer and exposes `image_via_app(...)`, which `smoovie()` calls when a GPU is present and otherwise falls back to the existing CPU `frame_dirty_maps`. This is Phase 2 of `docs/superpowers/specs/2026-06-16-smoovie-performance-and-gpu-design.md` §4. The detection (per-pixel IWP/Kalman) operators are explicitly out of scope — they attach to this backbone next.

**Tech Stack:** Python 3.10+, NumPy, xarray + zarr 3.x, astropy (host `C(t)`), CuPy 14.x + Holoscan 4.x (GPU operators/app), healpy, matplotlib + ffmpeg (host render), pytest.

---

## Background the engineer needs

- **Run tests:** `uv run pytest <path> -v`. **After every code change:** `uv run ruff format . && uv run ruff check . --fix` (non-negotiable; pre-commit + CI enforce it and generated code uses the same config). Commit only when a task says to. Branch is `actually_make_movie` — commit there; do **not** switch branches or touch `main`.
- **GPU reality:** the dev box has CuPy 14.1.1 (1 CUDA device) and Holoscan 4.3.0; **CI is CPU-only (no CuPy)**. Therefore: anything that imports `cupy`/`holoscan` at module top is GPU-only and its tests must be gated (skip when no GPU). The pure DFT logic (`image_frame_prerotated`), the prepare-step (numpy/astropy/xarray), and the render/encode/catalogue code stay CPU-importable and CPU-tested. **Do not add `import cupy`/`import holoscan` to `core/smoovie.py`, `core/smoovie_prepare.py`, or `utils/healpix_dft.py`** — those must import on a CPU machine.
- **Holoscan stack-size warning:** importing `holoscan` prints a `RuntimeWarning` about stack size and *may* segfault under the default stack. If a GPU test segfaults, run it under `ulimit -s 32768` (note this in the test docstring, do not hard-code it).
- **CLI/cab is unchanged in Phase 2.** No new Typer parameters are added, so `cli/smoovie.py`, `cabs/smoovie.yml`, and `tests/test_roundtrip.py` are **not** touched. The GPU/CPU choice is a core-only `use_gpu` argument (auto-detect by default), deliberately *not* surfaced on the CLI, so the round-trip stays byte-stable. Do not regenerate the cab.
- **Existing reader/writer (`operators/io.py`) are shared with `stream_msv4` and must keep working** — add *new* operator classes rather than mutating `XarrayZarrReaderOperator`/`ResultWriterOperator`.
- **`_partition(dt)`** returns the sole partition node under an MSv4 DataTree; `node.ds` is the main visibility dataset. `_correct_file_gains(node, vis, wgt, *, xp=np)` (in `core/smoovie.py`) maps baselines→antenna gains and applies `apply_inverse_gains`. `itrs_baselines(node, xp)` (in `utils/rephasing.py`) returns `(nbl, 3)` ITRS baselines. `equatorial_baselines(itrs_bl, times, *, backend="astropy", xp=np)` (in `utils/healpix_dft.py`) returns `(n_time, nbl, 3)` rotated baselines. `_utc(unix_seconds)` (in `core/smoovie.py`) formats a UTC stamp string. Reuse these — do not reimplement.
- **Frame ordering** must match `frame_dirty_maps`: iterate `for path in hdf_paths` then sub-integration `time` within each file, in order. The catalogue cache (Phase 1) already aligns to this by `time`.
- **`config.yaml`:** an empty Holoscan config is valid (`tests/data/config.yaml` is empty). The app writes/uses a throwaway empty config in a temp dir; no repo file dependency.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/kremetart/utils/healpix_dft.py` | Add device-pure `image_frame_prerotated`; reimplement `image_frame` on top of it (signature/behaviour unchanged). | Modify (Task 1) |
| `src/kremetart/core/smoovie_prepare.py` | Host prepare-step: HDF sequence → one imaging-ready zarr (corrected `VISIBILITY`/`WEIGHT`, `B_ROT`, `time`, `frequency`, phase/site attrs). | Create (Task 2) |
| `src/kremetart/operators/dft_healpix.py` | `HealpixDFTOperator` consumes streamed `B_ROT` and calls `image_frame_prerotated` — pure cupy, no astropy. | Modify (Task 3) |
| `src/kremetart/operators/io.py` | New `HealpixZarrReaderOperator` + `HealpixWriterOperator` (the prepared-zarr stream + `(TIME, npix)` dirty-map writer). | Modify (Task 3) |
| `src/kremetart/core/smoovie_app.py` | GPU-only `SmooviePipeline(hs.core.Application)` + `image_via_app(...)` host helper. | Create (Task 4) |
| `src/kremetart/core/smoovie.py` | `smoovie()` gains `use_gpu` (auto), routing imaging through `image_via_app` or `frame_dirty_maps`; `_gpu_imaging_available()` helper. | Modify (Task 4) |
| `tests/test_healpix_dft.py` | `image_frame_prerotated` equals `image_frame` (CPU). | Modify (Task 1) |
| `tests/test_smoovie_prepare.py` | Prepare-step schema/shape/values/nframes (CPU). | Create (Task 2) |
| `tests/test_smoovie_gpu.py` | GPU-gated: app end-to-end + equivalence to CPU `frame_dirty_maps`. | Create (Task 5) |
| `tests/test_smoovie_core.py` | Force CPU path in existing orchestration unit tests; add GPU-gated end-to-end. | Modify (Tasks 4, 5) |

---

## Task 1: split the DFT into a device-pure `image_frame_prerotated`

**Files:**
- Modify: `src/kremetart/utils/healpix_dft.py:152-178` (the `image_frame` function)
- Test: `tests/test_healpix_dft.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_healpix_dft.py`:

```python
def test_image_frame_prerotated_matches_image_frame():
    """The device-pure core equals the full image_frame (which now wraps it)."""
    pytest.importorskip("astropy")
    from kremetart.utils.healpix_dft import equatorial_baselines, image_frame, image_frame_prerotated

    rng = np.random.default_rng(7)
    nside = 8
    pix = make_pixel_grid(nside, xp=np)
    itrs_bl = rng.standard_normal((15, 3)) * 3.0
    times = np.array([1.6e9, 1.6e9 + 60.0])
    freqs = np.array([1.575e9])
    nbl = itrs_bl.shape[0]
    vis = rng.standard_normal((2, nbl, 1)) + 1j * rng.standard_normal((2, nbl, 1))
    wgt = np.ones((2, nbl, 1))

    ref = image_frame(vis, wgt, times, itrs_bl, pix, freqs, xp=np)
    b_rot = equatorial_baselines(itrs_bl, times, xp=np)
    got = image_frame_prerotated(vis, wgt, b_rot, pix, freqs, xp=np)

    assert got.shape == (pix.shape[0],)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_healpix_dft.py::test_image_frame_prerotated_matches_image_frame -v`
Expected: FAIL — `ImportError: cannot import name 'image_frame_prerotated'`.

- [ ] **Step 3: Add `image_frame_prerotated` and rewrite `image_frame` on top of it**

In `src/kremetart/utils/healpix_dft.py`, replace the whole `image_frame` function (currently lines 152-178) with the two functions below:

```python
def image_frame_prerotated(vis, weights, b_rot, pix_vec, freqs, *, xp: ModuleType = np):
    """Per-frame dirty image from already-rotated baselines (device-pure; no host astropy).

    The frame rotation ``C(t)`` has already been folded into ``b_rot``; this is the pure-``xp``
    core shared by the CPU :func:`image_frame` and the GPU
    :class:`kremetart.operators.dft_healpix.HealpixDFTOperator`. It flattens ``(time, baseline)``
    into the row axis and adjoint-DFTs onto the fixed grid.

    Args:
        vis: ``(n_time, nbl, nchan)`` complex residual visibilities (scalar pol).
        weights: ``(n_time, nbl, nchan)`` gain-corrected weights.
        b_rot: ``(n_time, nbl, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        xp: Array module.

    Returns:
        ``(npix,)`` real dirty image.
    """
    b_rot = xp.asarray(b_rot)
    vis = xp.asarray(vis)
    weights = xp.asarray(weights)
    n_time, nbl, _ = b_rot.shape
    nchan = vis.shape[-1]
    rows = b_rot.reshape(n_time * nbl, 3)
    vis_rows = vis.reshape(n_time * nbl, nchan)
    wgt_rows = weights.reshape(n_time * nbl, nchan)
    return dirty_map(vis_rows, wgt_rows, rows, pix_vec, freqs, xp=xp)


def image_frame(
    vis, weights, times, itrs_baselines, pix_vec, freqs, *, ctime_backend: str = "astropy", xp: ModuleType = np
):
    """Per-frame dirty image from unstopped residual visibilities.

    Rotates the ITRS baselines by ``C(t)`` on the host (:func:`equatorial_baselines`) and delegates
    the DFT to the device-pure :func:`image_frame_prerotated`. Signature and result are unchanged
    from the original single-function implementation, so existing callers/tests are unaffected.

    Args:
        vis: ``(n_time, nbl, nchan)`` complex residual visibilities (scalar pol).
        weights: ``(n_time, nbl, nchan)`` gain-corrected weights.
        times: ``(n_time,)`` unix-second timestamps.
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        ctime_backend: passed to :func:`equatorial_baselines`.
        xp: Array module.

    Returns:
        ``(npix,)`` real dirty image.
    """
    b_rot = equatorial_baselines(itrs_baselines, times, backend=ctime_backend, xp=xp)  # (n_time, nbl, 3)
    return image_frame_prerotated(vis, weights, b_rot, pix_vec, freqs, xp=xp)
```

- [ ] **Step 4: Run the new test and the full healpix suite**

Run: `uv run pytest tests/test_healpix_dft.py -v`
Expected: PASS — the new test plus all pre-existing tests (`test_image_frame_recovers_source_through_ctime`, etc.) still green (proves `image_frame` behaviour is preserved).

- [ ] **Step 5: Format and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
git commit -m "refactor: split healpix image_frame into device-pure image_frame_prerotated"
```

---

## Task 2: host prepare-step — HDF sequence → imaging-ready zarr

**Files:**
- Create: `src/kremetart/core/smoovie_prepare.py`
- Test: `tests/test_smoovie_prepare.py`

The prepared zarr is a plain `xr.Dataset` (not a DataTree) with dims `(time, baseline, frequency)` for the data and `(time, baseline, xyz)` for `B_ROT`. It is **nside-independent** (neither `B_ROT` nor the corrected visibilities depend on the pixel grid), so one prepared zarr serves any `nside`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_smoovie_prepare.py`:

```python
"""Tests for the host prepare-step (HDF sequence -> imaging-ready zarr). CPU, no GPU."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("xarray_ms")  # read_hdf_as_msv4 path uses MSv4 machinery
pytest.importorskip("astropy")

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_prepare_msv4_zarr_schema_and_shapes(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie import _partition
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    paths = _hdfs()[:1]
    main = _partition(read_hdf_as_msv4(paths[0])).ds
    n_time = int(main.time.size)
    n_bl = int(main.baseline_id.size)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(paths, out)
    ds = xr.open_zarr(str(out))

    assert set(ds["VISIBILITY"].dims) == {"time", "baseline", "frequency"}
    assert set(ds["WEIGHT"].dims) == {"time", "baseline", "frequency"}
    assert set(ds["B_ROT"].dims) == {"time", "baseline", "xyz"}
    assert ds["VISIBILITY"].shape == (n_time, n_bl, 1)
    assert ds["B_ROT"].shape == (n_time, n_bl, 3)
    assert np.iscomplexobj(ds["VISIBILITY"].values)
    np.testing.assert_allclose(ds.time.values, np.asarray(main.time.values))


def test_prepare_msv4_zarr_brot_matches_equatorial_baselines(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie import _partition
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr
    from kremetart.utils.healpix_dft import equatorial_baselines
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.rephasing import itrs_baselines

    paths = _hdfs()[:1]
    node = _partition(read_hdf_as_msv4(paths[0]))
    times = np.asarray(node.ds.time.values)
    bl = np.asarray(itrs_baselines(node, np))
    expected = equatorial_baselines(bl, times, xp=np)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(paths, out)
    ds = xr.open_zarr(str(out))
    np.testing.assert_allclose(ds["B_ROT"].values, expected, rtol=1e-12, atol=1e-12)


def test_prepare_msv4_zarr_correct_gains_matches_helper(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    paths = _hdfs()[:1]
    node = _partition(read_hdf_as_msv4(paths[0]))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]
    vis_c, wgt_c = _correct_file_gains(node, vis, wgt)

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(paths, out, correct_gains=True)
    ds = xr.open_zarr(str(out))
    np.testing.assert_allclose(ds["VISIBILITY"].values, vis_c.astype(np.complex64), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ds["WEIGHT"].values, wgt_c.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_prepare_msv4_zarr_nframes_caps(tmp_path):
    import xarray as xr

    from kremetart.core.smoovie_prepare import prepare_msv4_zarr

    out = tmp_path / "prepared.zarr"
    prepare_msv4_zarr(_hdfs(), out, nframes=3)
    ds = xr.open_zarr(str(out))
    assert ds.time.size == 3
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run pytest tests/test_smoovie_prepare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kremetart.core.smoovie_prepare'`.

- [ ] **Step 3: Implement the prepare-step**

Create `src/kremetart/core/smoovie_prepare.py`:

```python
"""Host prepare-step: a TART HDF sequence -> one imaging-ready zarr for the GPU smoovie app.

Does every host/astropy/gain task once, up front, so the streaming Holoscan imager (the GPU
``HealpixDFTOperator``) is pure cupy. For the whole HDF sequence it reads each file to MSv4,
optionally applies the inverse per-antenna gains (:func:`kremetart.core.smoovie._correct_file_gains`
-> :func:`kremetart.utils.gains.apply_inverse_gains`), and precomputes the per-frame
equatorial-rotated baselines ``b_rot(t)`` (:func:`kremetart.utils.healpix_dft.equatorial_baselines`).
The result is a single ``xarray.Dataset`` written to zarr with corrected ``VISIBILITY``/``WEIGHT``,
``B_ROT``, the ``time``/``frequency`` coordinates, and the common phase-direction + site metadata.

The zarr is nside-independent (``B_ROT`` and the corrected visibilities do not depend on the pixel
grid) and reusable across runs. This module is host-only (numpy/astropy/xarray) and must stay
importable on a CPU machine -- it never imports cupy or holoscan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def prepare_msv4_zarr(
    hdf_paths,
    out_zarr,
    *,
    correct_gains: bool = False,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    nframes: int | None = None,
):
    """Write the HDF sequence to one imaging-ready zarr; return the output path.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (same order as ``frame_dirty_maps``).
        out_zarr: output zarr path (overwritten if present).
        correct_gains: divide vis/weights by the per-antenna gain product before writing.
        phase_ra_deg: common phase-direction RA (deg, ICRS); stored in attrs (``NaN`` if unset).
        phase_dec_deg: common phase-direction Dec (deg, ICRS); stored in attrs (``NaN`` if unset).
        nframes: optional cap on the total number of frames written (profiling/preview aid).

    Returns:
        The ``out_zarr`` path.

    Raises:
        FileNotFoundError: if no frames are produced.
    """
    import shutil

    import xarray as xr

    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.utils.healpix_dft import equatorial_baselines
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.rephasing import itrs_baselines

    out_zarr = Path(out_zarr)
    vis_all: list[np.ndarray] = []
    wgt_all: list[np.ndarray] = []
    brot_all: list[np.ndarray] = []
    time_all: list[np.ndarray] = []
    freqs = None
    info = None

    for path in hdf_paths:
        if nframes is not None and sum(v.shape[0] for v in vis_all) >= nframes:
            break
        node = _partition(read_hdf_as_msv4(path))
        main = node.ds
        times = np.asarray(main.time.values)
        bl = np.asarray(itrs_baselines(node, np))  # (nbl, 3) host
        vis = np.asarray(main.VISIBILITY.values)[..., 0]  # (n_time, nbl, nchan)
        wgt = np.asarray(main.WEIGHT.values)[..., 0]
        if freqs is None:
            freqs = np.asarray(main.frequency.values)
            info = main.attrs["observation_info"]
        if correct_gains:
            vis, wgt = _correct_file_gains(node, vis, wgt, xp=np)
        b_rot = equatorial_baselines(bl, times, xp=np)  # (n_time, nbl, 3)
        vis_all.append(np.asarray(vis))
        wgt_all.append(np.asarray(wgt))
        brot_all.append(np.asarray(b_rot))
        time_all.append(times)

    if not vis_all:
        raise FileNotFoundError("no HDF frames to prepare")

    vis_c = np.concatenate(vis_all, axis=0)
    wgt_c = np.concatenate(wgt_all, axis=0)
    brot = np.concatenate(brot_all, axis=0)
    tt = np.concatenate(time_all, axis=0)
    if nframes is not None:
        vis_c, wgt_c, brot, tt = vis_c[:nframes], wgt_c[:nframes], brot[:nframes], tt[:nframes]

    ds = xr.Dataset(
        data_vars={
            "VISIBILITY": (("time", "baseline", "frequency"), vis_c.astype(np.complex64)),
            "WEIGHT": (("time", "baseline", "frequency"), wgt_c.astype(np.float32)),
            "B_ROT": (("time", "baseline", "xyz"), brot.astype(np.float64)),
        },
        coords={
            "time": ("time", tt.astype(np.float64)),
            "frequency": ("frequency", np.asarray(freqs, dtype=np.float64)),
            "xyz": ("xyz", np.array(["x", "y", "z"])),
        },
        attrs={
            "phase_ra_deg": float("nan") if phase_ra_deg is None else float(phase_ra_deg),
            "phase_dec_deg": float("nan") if phase_dec_deg is None else float(phase_dec_deg),
            "site_latitude_deg": float(info["site_latitude_deg"]),
            "site_longitude_deg": float(info["site_longitude_deg"]),
            "site_altitude_m": float(info["site_altitude_m"]),
        },
    )
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    ds.to_zarr(out_zarr, mode="w")
    return out_zarr
```

- [ ] **Step 4: Run the prepare tests**

Run: `uv run pytest tests/test_smoovie_prepare.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Format and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/core/smoovie_prepare.py tests/test_smoovie_prepare.py
git commit -m "feat: add host prepare-step writing an imaging-ready smoovie zarr"
```

---

## Task 3: GPU operators — pure-cupy `HealpixDFTOperator` + healpix reader/writer

**Files:**
- Modify: `src/kremetart/operators/dft_healpix.py` (full rewrite — consume streamed `B_ROT`)
- Modify: `src/kremetart/operators/io.py` (append two new operator classes)
- Test: `tests/test_smoovie_gpu.py` (create)

These three operators import `cupy`/`holoscan` at module top, so they are **GPU-only**. Their test is gated and skipped on CPU CI. The behavioural verification (equals the CPU path) is Task 5; this task verifies they import and the classes exist on a GPU box.

- [ ] **Step 1: Create the GPU-gated test file with an import smoke test**

Create `tests/test_smoovie_gpu.py`:

```python
"""GPU-gated tests for the smoovie Holoscan app and its operators.

Skipped unless a CUDA device and the cupy/holoscan/healpy stack are present (CPU CI skips all of
this). If a test segfaults during holoscan import, raise the stack limit first: `ulimit -s 32768`.
"""

from pathlib import Path

import numpy as np
import pytest


def _gpu():
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        import healpy  # noqa: F401
        import holoscan  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="requires a CUDA device + cupy/holoscan/healpy")

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_gpu_operators_import():
    from kremetart.operators.dft_healpix import HealpixDFTOperator
    from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator

    assert HealpixDFTOperator and HealpixZarrReaderOperator and HealpixWriterOperator
```

- [ ] **Step 2: Run it (passes-or-skips, depending on the box)**

Run: `uv run pytest tests/test_smoovie_gpu.py::test_gpu_operators_import -v`
Expected: on the dev GPU box, FAIL — `ImportError: cannot import name 'HealpixZarrReaderOperator'`. On CPU CI it would SKIP. (We are on the GPU box, so expect FAIL until Step 4.)

- [ ] **Step 3: Rewrite `HealpixDFTOperator` to consume streamed `B_ROT`**

Replace the entire contents of `src/kremetart/operators/dft_healpix.py` with:

```python
"""Holoscan operator: full-sky HEALPix dirty-map imager (GPU-resident, xp=cupy).

Pure cupy: consumes the per-frame equatorial-rotated baselines ``b_rot(t)`` precomputed by the host
prepare-step (:mod:`kremetart.core.smoovie_prepare`), so there is no astropy round-trip inside
``compute()``. Thin wrapper around :func:`kremetart.utils.healpix_dft.image_frame_prerotated`; images
onto a fixed full-sky equatorial HEALPix grid built once on the device.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.utils.healpix_dft import image_frame_prerotated, make_pixel_grid


class HealpixDFTOperator(Operator):
    """Adjoint (dirty-map) HEALPix DFT imager, pure cupy.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        freqs: ``(nchan,)`` frequencies in Hz.
        nest: NESTED HEALPix ordering (default True; index locality for the streaming detector).
    """

    def __init__(self, fragment, nside, freqs, *args, nest=True, **kwargs):
        self.nside = nside
        self.freqs = cp.asarray(freqs)
        self.nest = nest
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Build the fixed equatorial pixel grid once, on device.
        self.pix_vec = make_pixel_grid(self.nside, nest=self.nest, xp=cp)

    def setup(self, spec: OperatorSpec):
        spec.input("VISIBILITY")
        spec.input("WEIGHT")
        spec.input("B_ROT")
        spec.input("time")
        spec.output("cube")
        spec.output("time_out")

    def compute(self, op_input, op_output, context):
        vis = cp.asarray(op_input.receive("VISIBILITY"))  # (1, nbl, nchan)
        wgt = cp.asarray(op_input.receive("WEIGHT"))  # (1, nbl, nchan)
        b_rot = cp.asarray(op_input.receive("B_ROT"))  # (1, nbl, 3)
        times = cp.asarray(op_input.receive("time"))  # (1,)

        dmap = image_frame_prerotated(vis, wgt, b_rot, self.pix_vec, self.freqs, xp=cp)  # (npix,)

        # Output layout: (ntime_out=1, npix) -- one dirty-map row per frame.
        op_output.emit(hs.as_tensor(dmap[None, :]), "cube")
        op_output.emit(hs.as_tensor(cp.mean(times, keepdims=True)), "time_out")
```

- [ ] **Step 4: Append the healpix reader and writer to `operators/io.py`**

The module already imports `cp`, `da`, `hs`, `np`, `xr`, `Operator`, `OperatorSpec`, `NDArray` (lines 1-7); no new imports are needed. Append these two classes to the **end** of `src/kremetart/operators/io.py`:

```python
class HealpixZarrReaderOperator(Operator):
    """Stream a prepared imaging zarr (VISIBILITY, WEIGHT, B_ROT, time) one frame at a time to the GPU.

    Reads the host prepare-step output (:func:`kremetart.core.smoovie_prepare.prepare_msv4_zarr`).
    Unlike :class:`XarrayZarrReaderOperator` it carries the precomputed ``B_ROT`` and drops UVW/FLAG
    (the HEALPix imager does not need them).
    """

    def __init__(self, fragment, *args, zarr_path: str, **kwargs):
        self.zarr_path = zarr_path
        self.current_index = 0
        self.ntime = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("VISIBILITY")
        spec.output("WEIGHT")
        spec.output("B_ROT")
        spec.output("time")

    def start(self):
        self.dataset = xr.open_zarr(self.zarr_path)
        self.ntime = self.dataset["time"].size
        self.current_index = 0

    def compute(self, op_input, op_output, context):
        if self.current_index >= self.ntime:
            return
        s = self.dataset.isel({"time": slice(self.current_index, self.current_index + 1)})
        op_output.emit(hs.as_tensor(cp.asarray(s["VISIBILITY"].values)), "VISIBILITY")
        op_output.emit(hs.as_tensor(cp.asarray(s["WEIGHT"].values)), "WEIGHT")
        op_output.emit(hs.as_tensor(cp.asarray(s["B_ROT"].values)), "B_ROT")
        op_output.emit(hs.as_tensor(cp.asarray(s["time"].values)), "time")
        self.current_index += 1

    def stop(self):
        pass


class HealpixWriterOperator(Operator):
    """Write per-frame HEALPix dirty maps to a ``(TIME, PIX)`` zarr (dask scaffold + region="auto").

    Mirrors :class:`ResultWriterOperator`'s scaffold-then-region-write pattern, but for a flat
    ``(TIME, npix)`` HEALPix cube rather than a ``(STOKES, FREQ, TIME, Y, X)`` image cube.
    """

    def __init__(
        self,
        fragment,
        ntime,
        npix,
        *args,
        output_dataset: str = None,
        out_times: NDArray = None,
        **kwargs,
    ):
        self.output_dataset = output_dataset
        self.ntime = ntime
        self.npix = npix
        self.out_times = out_times if out_times is not None else np.arange(ntime)
        self.pix = np.arange(npix)
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # dask scaffold: da.empty allocates no data, only the zarr structure to write regions into.
        data_vars = {
            "dirty": (("TIME", "PIX"), da.empty((self.ntime, self.npix), chunks=(1, self.npix), dtype=np.float32)),
        }
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"TIME": (("TIME",), self.out_times), "PIX": (("PIX",), self.pix)},
        )
        ds.to_zarr(self.output_dataset, mode="w", compute=True)

    def setup(self, spec: OperatorSpec):
        spec.input("cube")
        spec.input("time_out")

    def compute(self, op_input, op_output, context):
        cube = cp.asnumpy(cp.asarray(op_input.receive("cube")))  # (1, npix)
        time_out = cp.asnumpy(cp.asarray(op_input.receive("time_out")))  # (1,)
        dso = xr.Dataset(
            data_vars={"dirty": (("TIME", "PIX"), cube.astype(np.float32))},
            coords={"TIME": (("TIME",), time_out), "PIX": (("PIX",), self.pix)},
        )
        dso.to_zarr(self.output_dataset, region="auto")

    def stop(self):
        pass
```

- [ ] **Step 5: Run the import smoke test**

Run: `uv run pytest tests/test_smoovie_gpu.py::test_gpu_operators_import -v`
Expected (GPU box): PASS. (CPU CI: SKIP.)

- [ ] **Step 6: Format and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/operators/dft_healpix.py src/kremetart/operators/io.py tests/test_smoovie_gpu.py
git commit -m "feat: GPU healpix imager consumes streamed b_rot; add healpix reader/writer ops"
```

---

## Task 4: the GPU app + route `smoovie()` through it (CPU fallback preserved)

**Files:**
- Create: `src/kremetart/core/smoovie_app.py`
- Modify: `src/kremetart/core/smoovie.py` (add `_gpu_imaging_available`, add `use_gpu` param, route the imaging stage)
- Modify: `tests/test_smoovie_core.py` (force the CPU path in the orchestration unit tests)

`smoovie()` keeps its current CPU behaviour as a fallback. The new `use_gpu` argument (auto-detect by default) is **core-only** — it is not added to `cli/smoovie.py`, so the cab and round-trip test are untouched. The existing monkeypatch-based unit tests stub `frame_dirty_maps`; on the dev GPU box the auto-detect would otherwise route past that stub into a real GPU run, so those tests must pin `use_gpu=False`.

- [ ] **Step 1: Write the GPU app module**

Create `src/kremetart/core/smoovie_app.py`:

```python
"""GPU Holoscan smoovie imaging app: a prepared zarr -> a streamed HEALPix dirty-map zarr.

Mirrors :mod:`kremetart.core.stream_msv4`. All host work (gain correction, the per-frame ``b_rot(t)``
rotation, the catalogue) is done by the prepare-step (:mod:`kremetart.core.smoovie_prepare`) and the
catalogue cache; this module is the pure-GPU imaging backbone: reader -> HealpixDFTOperator ->
writer. Movie rendering/encoding happens on the host *after* ``app.run()`` (see :func:`image_via_app`).

``holoscan`` and ``cupy`` import at module top, so importing this module requires a GPU. The CPU
``smoovie`` path in :mod:`kremetart.core.smoovie` never imports it -- it is imported lazily, only when
:func:`kremetart.core.smoovie._gpu_imaging_available` is true.
"""

from pathlib import Path

import holoscan as hs
import numpy as np
import xarray as xr
from holoscan.conditions import CountCondition

from kremetart.operators.dft_healpix import HealpixDFTOperator
from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator


class SmooviePipeline(hs.core.Application):
    """Stream a prepared imaging zarr through the GPU HEALPix imager into a ``(TIME, npix)`` zarr."""

    def __init__(self, prepared_zarr, output_zarr, nside, *args, nest=True, **kwargs):
        self.prepared_zarr = str(prepared_zarr)
        self.output_zarr = str(output_zarr)
        self.nside = nside
        self.nest = nest
        super().__init__(*args, **kwargs)

        import healpy as hp

        ds = xr.open_zarr(self.prepared_zarr)
        self.ntime = int(ds.time.size)
        self.out_times = ds.time.values
        self.freqs = ds.frequency.values
        self.npix = hp.nside2npix(nside)

    def compose(self):
        reader = HealpixZarrReaderOperator(
            self,
            CountCondition(self, self.ntime),
            name="reader",
            zarr_path=self.prepared_zarr,
        )
        imager = HealpixDFTOperator(self, self.nside, self.freqs, name="imager", nest=self.nest)
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
        self.add_flow(imager, writer, {("cube", "cube"), ("time_out", "time_out")})


def image_via_app(
    hdf_paths,
    nside,
    *,
    correct_gains=False,
    phase_ra_deg=None,
    phase_dec_deg=None,
    nframes=None,
    nest=True,
):
    """Image the HDF sequence through the GPU app; return ``(maps, stamps)``.

    Drop-in for the imaging half of :func:`kremetart.core.smoovie.frame_dirty_maps`: returns a list
    of ``(npix,)`` dirty maps (one per frame, in order) and a list of UTC stamp strings. Runs the
    host prepare-step into a temp zarr, streams it through :class:`SmooviePipeline`, then loads the
    ``(TIME, npix)`` output zarr back to host.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        correct_gains: apply inverse per-antenna gains in the prepare-step.
        phase_ra_deg, phase_dec_deg: common phase direction (deg, ICRS), stored as zarr metadata.
        nframes: optional cap on frames.
        nest: NESTED HEALPix ordering (default True; matches the CPU path).

    Returns:
        ``(maps, stamps)``.
    """
    import tempfile

    from kremetart.core.smoovie import _utc
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr

    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "prepared.zarr"
        output = Path(td) / "dirty.zarr"
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

        app = SmooviePipeline(prepared, output, nside, nest=nest)
        app.config(str(config))
        app.run()

        ds = xr.open_zarr(str(output))
        dirty = np.asarray(ds["dirty"].values)  # (ntime, npix)
        times = np.asarray(ds["TIME"].values)

    maps = [dirty[i] for i in range(dirty.shape[0])]
    stamps = [_utc(t) for t in times]
    return maps, stamps
```

- [ ] **Step 2: Add the `_gpu_imaging_available` helper to `core/smoovie.py`**

In `src/kremetart/core/smoovie.py`, insert this function immediately **after** the `_utc` function (currently lines 25-27, before `common_phase_direction`):

```python
def _gpu_imaging_available() -> bool:
    """True if a CUDA device plus the GPU imaging stack (cupy/holoscan/healpy) is importable.

    Drives ``smoovie``'s auto-routing: when true the imaging runs through the Holoscan GPU app
    (:func:`kremetart.core.smoovie_app.image_via_app`); otherwise it falls back to the CPU
    :func:`frame_dirty_maps`. Any import error or absent device -> CPU path, so CPU-only CI and
    machines without a GPU keep working.
    """
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        import healpy  # noqa: F401
        import holoscan  # noqa: F401

        return True
    except Exception:
        return False
```

- [ ] **Step 3: Add the `use_gpu` parameter and route the imaging stage**

In `src/kremetart/core/smoovie.py`, change the `smoovie(...)` signature to add `use_gpu` (insert it after `nframes: int | None = None,`):

```python
    nframes: int | None = None,
    use_gpu: bool | None = None,
):
```

Then replace the imaging stage block (currently):

```python
    print("Making dirty maps")
    with _stage_timer("imaging", timings):
        maps, stamps, _ = frame_dirty_maps(hdf_paths, nside, correct_gains=correct_gains, nframes=nframes)
```

with:

```python
    print("Making dirty maps")
    with _stage_timer("imaging", timings):
        use = _gpu_imaging_available() if use_gpu is None else use_gpu
        if use:
            from kremetart.core.smoovie_app import image_via_app

            maps, stamps = image_via_app(
                hdf_paths,
                nside,
                correct_gains=correct_gains,
                phase_ra_deg=phase_ra_deg,
                phase_dec_deg=phase_dec_deg,
                nframes=nframes,
            )
        else:
            maps, stamps, _ = frame_dirty_maps(hdf_paths, nside, correct_gains=correct_gains, nframes=nframes)
```

Also extend the `smoovie` docstring: add a line documenting `use_gpu` (after the `nframes` sentence):

```
    ``use_gpu`` selects the imaging backend: ``None`` (default) auto-detects a CUDA device + the
    Holoscan stack and uses the GPU app when present, else the CPU ``frame_dirty_maps``; pass
    ``True``/``False`` to force one. It is a core-only knob (not a CLI flag).
```

- [ ] **Step 4: Pin the CPU path in the orchestration unit tests**

These tests stub `frame_dirty_maps`; pin them to the CPU branch so the auto-detect never routes into a real GPU run on the dev box. In `tests/test_smoovie_core.py`:

Edit `test_smoovie_produces_movie` — change:
```python
    smoovie(hdf_dir=_DATA, movie=out, nside=32, fps=2)
```
to:
```python
    smoovie(hdf_dir=_DATA, movie=out, nside=32, fps=2, use_gpu=False)
```

Edit `test_smoovie_honors_explicit_phase_direction` — change:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, phase_ra_deg=12.0, phase_dec_deg=-20.0)
```
to:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, phase_ra_deg=12.0, phase_dec_deg=-20.0, use_gpu=False)
```

Edit `test_smoovie_overlay_passes_tracks` — change:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, overlay_catalog=True, catalog_elevation_deg=30.0)
```
to:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, overlay_catalog=True, catalog_elevation_deg=30.0, use_gpu=False)
```

Edit `test_smoovie_profile_prints` — change:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, profile=True)
```
to:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, profile=True, use_gpu=False)
```

Edit `test_smoovie_nframes_flows_to_imaging` — change:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, nframes=7)
```
to:
```python
    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, nframes=7, use_gpu=False)
```

Edit `test_smoovie_default_catalog_cache_path` — change:
```python
    sm.smoovie(hdf_dir=_DATA, movie=movie, nside=1, overlay_catalog=True, nframes=3)
```
to:
```python
    sm.smoovie(hdf_dir=_DATA, movie=movie, nside=1, overlay_catalog=True, nframes=3, use_gpu=False)
```

The two bare-call tests (`test_smoovie_auto_phase_direction_used` and `test_smoovie_no_overlay_passes_none_tracks`) both have the identical line `    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1)`. Replace **both** occurrences with `    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, use_gpu=False)` (use the Edit tool's `replace_all=True`, or edit each by hand).

- [ ] **Step 5: Run the CPU orchestration tests (no GPU path exercised)**

Run: `uv run pytest tests/test_smoovie_core.py -v`
Expected: PASS — every test green, including `test_smoovie_produces_movie` (now pinned to the CPU backend). This proves the CPU fallback and the existing behaviour are intact.

- [ ] **Step 6: Format and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/core/smoovie_app.py src/kremetart/core/smoovie.py tests/test_smoovie_core.py
git commit -m "feat: route smoovie imaging through the GPU Holoscan app with CPU fallback"
```

---

## Task 5: GPU end-to-end + equivalence to the CPU baseline

**Files:**
- Modify: `tests/test_smoovie_gpu.py` (append three GPU-gated tests)

These tests run only on the dev GPU box (skipped on CPU CI by the module-level `pytestmark`). The equivalence test is the behaviour-preservation guard the spec (§5, §6) requires: the GPU app's dirty maps must match the trusted CPU `frame_dirty_maps`. Tolerances are set by the shared single-precision (`complex64`) visibilities and the `float32` zarr dirty-map column — the frame rotation and DFT kernel are `float64` on both paths.

- [ ] **Step 1: Append the GPU end-to-end + equivalence tests**

Append to `tests/test_smoovie_gpu.py`:

```python
def test_image_via_app_end_to_end(tmp_path):
    from kremetart.core.smoovie_app import image_via_app

    nside = 8
    npix = 12 * nside * nside
    maps, stamps = image_via_app(_hdfs()[:1], nside, correct_gains=True, nframes=3)
    assert len(maps) == len(stamps) == 3
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
    assert "UTC" in stamps[0]


def test_gpu_app_matches_cpu_frame_dirty_maps():
    """Behaviour preservation: GPU-app dirty maps equal the CPU frame_dirty_maps baseline."""
    from kremetart.core.smoovie import frame_dirty_maps
    from kremetart.core.smoovie_app import image_via_app

    paths = _hdfs()[:1]
    nside = 8
    cpu_maps, _, _ = frame_dirty_maps(paths, nside, correct_gains=True, nframes=3)
    gpu_maps, _ = image_via_app(paths, nside, correct_gains=True, nframes=3)

    assert len(gpu_maps) == len(cpu_maps) == 3
    for c, g in zip(cpu_maps, gpu_maps):
        np.testing.assert_allclose(np.asarray(g), np.asarray(c), rtol=1e-4, atol=1e-5)


def test_smoovie_produces_movie_gpu(tmp_path):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from kremetart.core.smoovie import smoovie

    out = tmp_path / "movie.mp4"
    smoovie(hdf_dir=_DATA, movie=out, nside=16, fps=2, nframes=4, use_gpu=True)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run the GPU test file**

Run: `uv run pytest tests/test_smoovie_gpu.py -v`
Expected (GPU box): PASS (4 tests). If the equivalence test fails by a hair, first confirm it is a precision gap (max abs diff ~1e-4..1e-5, structurally identical maps) and not a real bug — a wrong axis/transpose/sign would be `O(map-scale)`, not `O(1e-4)`. If holoscan segfaults on import, re-run under `ulimit -s 32768`.

- [ ] **Step 3: Run the full CPU suite (regression check)**

Run: `KREMETART_OFFLINE=1 uv run pytest tests/ -v --ignore=tests/test_smoovie_gpu.py` (or just `uv run pytest tests/`)
Expected: PASS/skip — in particular `tests/test_roundtrip.py::test_roundtrip_smoovie` still passes (the CLI/cab were not touched), `tests/test_healpix_dft.py` and `tests/test_smoovie_core.py` green, `tests/test_smoovie_prepare.py` green.

- [ ] **Step 4: Format and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add tests/test_smoovie_gpu.py
git commit -m "test: GPU smoovie end-to-end and equivalence to the CPU baseline"
```

- [ ] **Step 5: Finish the development branch**

Announce: "I'm using the finishing-a-development-branch skill to complete this work."
**REQUIRED SUB-SKILL:** Use superpowers:finishing-a-development-branch — verify the suite, then present merge/PR/cleanup options.

---

## Self-Review (performed against the spec §4)

- **§4.1 data flow** — `prepare → reader → HealpixDFTOperator → writer → healpix zarr → host render/ffmpeg`: Task 2 (prepare), Task 3 (operators), Task 4 (app wiring + post-`run()` render reuse), Task 5 (end-to-end). ✓
- **§4.2 prepare step** — `core/smoovie_prepare.py`, gains via `apply_inverse_gains` (through `_correct_file_gains`), `b_rot(t)` via `equatorial_baselines`, writes corrected `VISIBILITY`/`WEIGHT`/`B_ROT`/`time`/`frequency` + phase metadata: Task 2. ✓
- **§4.2 reader** — streams `(VISIBILITY, WEIGHT, B_ROT, time)`, drops UVW/FLAG: Task 3 `HealpixZarrReaderOperator`. (Deviation from the spec's "adapt `XarrayZarrReaderOperator`": a **new** class is added instead of mutating the shared one, so `stream_msv4` keeps working — flagged in Background.) ✓
- **§4.2 `HealpixDFTOperator` pure GPU** — consumes streamed `b_rot`, calls `image_frame_prerotated` with `xp=cp`, no astropy in `compute()`: Task 3. ✓
- **§4.2 writer** — healpix variant writing `(TIME, npix)` zarr (dask scaffold + `region="auto"`): Task 3 `HealpixWriterOperator`. ✓
- **§4.2 post-`app.run()`** — `render_frames`/`_overlay_tracks`/`encode_movie` unchanged; `image_via_app` returns `(maps, stamps)` so `smoovie()`'s render/encode tail is untouched: Task 4. ✓
- **§4.2 `smoovie()` Phase-2 shape + empty `config.yaml`** — `prepare → app.run() → render → ffmpeg`, empty temp config via `app.config(...)`: Task 4 `image_via_app`. ✓
- **§4.3 `healpix_dft` refactor** — `image_frame` decomposed into host `equatorial_baselines` + device-pure `image_frame_prerotated`; `image_frame` signature/behaviour preserved: Task 1. ✓
- **§5 risks** — GPU-only operators are gated (Tasks 3/5 `pytestmark`); `image_frame_prerotated` stays `xp`-injectable and CPU-tested (Task 1); empty config handled (Task 4); behaviour-preservation equivalence test (Task 5). The rendering risk is already resolved (Phase-1 follow-up: overlay uses axes methods), so no render rework is needed here. ✓
- **§6 Phase-2 testing** — `image_frame_prerotated == image_frame` (Task 1), prepare produces valid zarr with corrected `VISIBILITY` + right-shape `b_rot` (Task 2), `HealpixDFTOperator` emits the expected shape (Task 3 import + Task 5 end-to-end), GPU≈CPU + playable mp4 (Task 5). ✓
- **Out of scope (§7)** — no per-pixel IWP/detection operators, no multi-frame batching, no GPU-native `C(t)`, no overlay/gain semantics change. The per-pixel IWP evolution is the explicit next step after this backbone runs. ✓
- **Type consistency** — `image_frame_prerotated(vis, weights, b_rot, pix_vec, freqs, *, xp)` is called identically in Task 1 (`image_frame`) and Task 3 (operator). `prepare_msv4_zarr(...)` kwargs match between Task 2 (def), Task 4 (`image_via_app` call), and the tests. Reader emits exactly the four ports the imager's `setup` declares; imager emits the two ports the writer declares; app `add_flow` pairs match all port names. ✓
- **No CLI/cab churn** — `use_gpu` is core-only; `cli/smoovie.py`/`cabs/smoovie.yml`/`test_roundtrip.py` untouched, so the round-trip stays byte-stable (verified in Task 5 Step 3). ✓

## Notes / deviations from the spec (call out at execution time)

1. **New reader/writer classes instead of mutating the shared ones.** The spec said "adapt `XarrayZarrReaderOperator`"; mutating it would break `stream_msv4` (it consumes UVW/FLAG/FREQ). New `HealpixZarrReaderOperator`/`HealpixWriterOperator` keep both pipelines working at the cost of mild duplication — the right trade. If the team later retires the `dft_lm` path, the two readers can be merged.
2. **`smoovie()` keeps a CPU fallback (auto-detected).** The spec's Phase-2 shape is GPU-first; this plan makes the GPU app the default *when a GPU is present* and otherwise falls back to `frame_dirty_maps`, so CPU-only CI keeps a working end-to-end. This is additive, not a divergence from the GPU direction.
3. **`use_gpu` is not a CLI flag.** Keeps the cab/round-trip frozen for Phase 2. If a user-facing switch is wanted later, add it to `cli/smoovie.py` and regenerate the cab in its own change (it would touch `test_roundtrip.py`).
