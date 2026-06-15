# HEALPix DFT Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a gridless forward/adjoint DFT on a full-sky equatorial HEALPix grid (plus a dirty-map convenience and a thin Holoscan operator) that turns unstopped TART residual visibilities into a residual intensity image on a sidereally-fixed sky.

**Architecture:** An `xp`-injectable pure core in `src/kremetart/utils/healpix_dft.py` (NumPy for CPU tests, CuPy in the pipeline) holds all the math: pixel grid, `forward`/`adjoint` transpose pair, `dirty_map`, the host-side `C(t)` baseline rotation (`equatorial_baselines`), and an end-to-end `image_frame`. A thin `src/kremetart/operators/dft_healpix.py` Holoscan operator binds `xp=cp` and just shuffles tensors into `image_frame`. Mirrors the host/device split already used in `rephasing.py`.

**Tech Stack:** Python 3.10+, NumPy, healpy (pixel grid), astropy (the `C(t)` oracle), CuPy + Holoscan (pipeline operator). Tests use pytest with `xp=np`.

**Spec:** `docs/superpowers/specs/2026-06-15-healpix-dft-operator-design.md`

---

### Task 1: Module scaffold + `make_pixel_grid`

**Files:**
- Create: `src/kremetart/utils/healpix_dft.py`
- Test: `tests/test_healpix_dft.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_healpix_dft.py
"""Unit tests for the HEALPix gridless DFT operator (CPU, xp=numpy)."""

import numpy as np
import pytest

from kremetart.utils.healpix_dft import make_pixel_grid

LIGHTSPEED = 299792458.0


def test_make_pixel_grid_shape_and_unit_norm():
    nside = 8
    pix = make_pixel_grid(nside, xp=np)
    assert pix.shape == (12 * nside**2, 3)
    np.testing.assert_allclose(np.linalg.norm(pix, axis=1), 1.0, atol=1e-12)


def test_make_pixel_grid_nested_matches_healpy():
    hp = pytest.importorskip("healpy")
    nside = 4
    pix = make_pixel_grid(nside, xp=np)  # nest=True default
    expected = np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), axis=1)
    np.testing.assert_allclose(pix, expected, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: cannot import name 'make_pixel_grid'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/kremetart/utils/healpix_dft.py
"""Gridless forward/adjoint DFT on a full-sky equatorial HEALPix grid.

The imaging step of the TART streaming pipeline. Pixels are represented by their Cartesian
unit vectors (direction cosines); the dirty-map kernel is the bare geometric delay
``exp(-2pi i (nu/c) b . s)`` with no ``(n-1)`` reference term and no ``1/n`` Jacobian (the
HEALPix grid is equal-area). See docs/superpowers/specs/2026-06-15-healpix-dft-operator-design.md.

The math is ``xp``-injectable: pass ``xp=numpy`` (CPU tests) or ``xp=cupy`` (GPU pipeline).
Only the per-frame frame-rotation ``C(t)`` is computed on the host with astropy (O(n_time)),
exactly the host/device split used in :mod:`kremetart.utils.rephasing`.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

LIGHTSPEED = 299792458.0


def make_pixel_grid(nside: int, *, nest: bool = True, xp: ModuleType = np):
    """Return the HEALPix pixel unit vectors (direction cosines).

    Args:
        nside: HEALPix resolution; ``npix = 12 * nside**2``.
        nest: Use NESTED ordering (default; index locality for the streaming detector).
        xp: Array module for the returned array (``numpy`` or ``cupy``).

    Returns:
        ``(npix, 3)`` array of unit vectors, declared to live in the equatorial (ICRS) frame.
    """
    import healpy as hp

    npix = hp.nside2npix(nside)
    vec = hp.pix2vec(nside, np.arange(npix), nest=nest)  # tuple of three (npix,) arrays
    grid = np.stack(vec, axis=1).astype(np.float64)  # (npix, 3)
    return xp.asarray(grid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
.venv/bin/ruff check src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py --fix
git add src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
git commit -m "feat: add HEALPix pixel-grid (direction cosines) for the DFT imager"
```

---

### Task 2: `dft_forward` / `dft_adjoint` (Hermitian transpose pair)

**Files:**
- Modify: `src/kremetart/utils/healpix_dft.py`
- Test: `tests/test_healpix_dft.py`

The leading axis of `baselines`/`vis` is the **row** axis — a flattened `(time, baseline)` sample, since the rotated baseline differs per timestamp. The pure functions need not know that; they just see `nrow` rows.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_healpix_dft.py
from kremetart.utils.healpix_dft import dft_adjoint, dft_forward


def test_forward_matches_explicit_fringe():
    """forward of a single-pixel source is the geometric fringe on every row/channel."""
    rng = np.random.default_rng(1)
    pix = make_pixel_grid(4, xp=np)
    npix = pix.shape[0]
    baselines = rng.standard_normal((5, 3))
    freqs = np.array([1.40e9, 1.575e9])
    image = np.zeros(npix)
    image[3] = 2.0
    vis = dft_forward(image, baselines, pix, freqs, xp=np)
    assert vis.shape == (5, 2)
    g = baselines @ pix[3]  # (nrow,)
    expected = 2.0 * np.exp(2j * np.pi * (freqs[None, :] / LIGHTSPEED) * g[:, None])
    np.testing.assert_allclose(vis, expected, rtol=1e-12, atol=1e-12)


def test_forward_adjoint_are_hermitian_transposes():
    """<forward(image), data> == <image, adjoint(data)>  (the adjointness dot-product test)."""
    rng = np.random.default_rng(0)
    pix = make_pixel_grid(8, xp=np)
    npix = pix.shape[0]
    nrow, nchan = 12, 3
    baselines = rng.standard_normal((nrow, 3))
    freqs = np.array([1.40e9, 1.50e9, 1.575e9])
    image = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    data = rng.standard_normal((nrow, nchan)) + 1j * rng.standard_normal((nrow, nchan))
    lhs = np.vdot(dft_forward(image, baselines, pix, freqs, xp=np), data)
    rhs = np.vdot(image, dft_adjoint(data, baselines, pix, freqs, xp=np))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py -k "forward or adjoint" -v`
Expected: FAIL — `cannot import name 'dft_forward'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/kremetart/utils/healpix_dft.py

def _delay(baselines, pix_vec):
    """Geometric delay matrix b . s in metres, shape (nrow, npix)."""
    return baselines @ pix_vec.T


def _phase(baselines, pix_vec, freqs, xp):
    """2*pi*(nu/c)*(b . s), shape (nrow, nchan, npix)."""
    g = _delay(baselines, pix_vec)  # (nrow, npix)
    inv_wl = xp.asarray(freqs) / LIGHTSPEED  # (nchan,) cycles per metre
    return 2.0 * xp.pi * inv_wl[None, :, None] * g[:, None, :]


def dft_forward(image, baselines, pix_vec, freqs, *, xp: ModuleType = np):
    """Image -> visibilities (phasesign +1).

    Args:
        image: ``(npix,)`` sky (real in production; complex accepted for the adjoint test).
        baselines: ``(nrow, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
        pix_vec: ``(npix, 3)`` pixel unit vectors from :func:`make_pixel_grid`.
        freqs: ``(nchan,)`` frequencies in Hz.
        xp: Array module.

    Returns:
        ``(nrow, nchan)`` complex visibilities.
    """
    kernel = xp.exp(1j * _phase(baselines, pix_vec, freqs, xp))  # (nrow, nchan, npix)
    return kernel @ xp.asarray(image)  # (nrow, nchan)


def dft_adjoint(vis, baselines, pix_vec, freqs, *, xp: ModuleType = np):
    """Visibilities -> image (phasesign -1); the exact Hermitian transpose of :func:`dft_forward`.

    Args:
        vis: ``(nrow, nchan)`` complex visibilities.
        baselines, pix_vec, freqs, xp: as in :func:`dft_forward`.

    Returns:
        ``(npix,)`` complex image (caller takes ``Re`` / normalises; see :func:`dirty_map`).
    """
    kernel = xp.exp(-1j * _phase(baselines, pix_vec, freqs, xp))  # conj of forward
    return xp.einsum("rcj,rc->j", kernel, xp.asarray(vis))  # (npix,)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py -k "forward or adjoint" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
.venv/bin/ruff check src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py --fix
git add src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
git commit -m "feat: add HEALPix DFT forward/adjoint transpose pair"
```

---

### Task 3: `dirty_map`

**Files:**
- Modify: `src/kremetart/utils/healpix_dft.py`
- Test: `tests/test_healpix_dft.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_healpix_dft.py
from kremetart.utils.healpix_dft import dirty_map


def test_dirty_map_recovers_point_source():
    """dirty_map of a forward-modelled single-pixel source peaks (value 1) at that pixel."""
    rng = np.random.default_rng(2)
    pix = make_pixel_grid(16, xp=np)
    npix = pix.shape[0]
    nrow = 300
    baselines = rng.standard_normal((nrow, 3)) * 2.0
    freqs = np.array([1.575e9])
    src = 1234
    image = np.zeros(npix)
    image[src] = 1.0
    vis = dft_forward(image, baselines, pix, freqs, xp=np)
    weights = np.ones((nrow, 1))
    dmap = dirty_map(vis, weights, baselines, pix, freqs, xp=np)
    assert dmap.shape == (npix,)
    assert dmap.dtype == np.float64
    assert int(np.argmax(dmap)) == src
    np.testing.assert_allclose(dmap[src], 1.0, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py::test_dirty_map_recovers_point_source -v`
Expected: FAIL — `cannot import name 'dirty_map'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/kremetart/utils/healpix_dft.py

def dirty_map(vis, weights, baselines, pix_vec, freqs, *, xp: ModuleType = np):
    """Weighted adjoint dirty map: ``Re{ adjoint(weights * vis) } / sum(weights)``.

    Implements the design-doc dirty-map equation directly (equal-area grid: no 1/n factor).

    Args:
        vis: ``(nrow, nchan)`` complex residual visibilities.
        weights: ``(nrow, nchan)`` gain-corrected weights ``w_corr``.
        baselines, pix_vec, freqs, xp: as in :func:`dft_forward`.

    Returns:
        ``(npix,)`` real dirty image.
    """
    vis = xp.asarray(vis)
    weights = xp.asarray(weights)
    img = dft_adjoint(weights * vis, baselines, pix_vec, freqs, xp=xp)
    return img.real / weights.sum()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py::test_dirty_map_recovers_point_source -v`
Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
.venv/bin/ruff check src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py --fix
git add src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
git commit -m "feat: add HEALPix dirty-map convenience (weighted adjoint)"
```

---

### Task 4: `equatorial_baselines` — host `C(t)` rotation (astropy backend)

**Files:**
- Modify: `src/kremetart/utils/healpix_dft.py`
- Test: `tests/test_healpix_dft.py`

Rotates the fixed ITRS baselines into the equatorial frame so that, for a pixel `s` (ICRS unit
vector), `b_rot[t] . s == b_itrs . s_itrs(t)` — the same physical delay `rephasing.py` produces.
The native (GPU) backend is a later phase and must raise `NotImplementedError`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_healpix_dft.py
from kremetart.utils.healpix_dft import equatorial_baselines


def test_equatorial_baselines_matches_itrs_unit_vectors():
    """b_rot . s_icrs equals b_itrs . s_itrs(t) from the tested rephasing machinery."""
    pytest.importorskip("astropy")
    from kremetart.utils.rephasing import _itrs_unit_vectors

    rng = np.random.default_rng(3)
    itrs_bl = rng.standard_normal((7, 3)) * 3.0
    times = np.array([1.6e9, 1.6e9 + 600.0, 1.6e9 + 3600.0])
    ra, dec = 1.2, -0.35
    s_icrs = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    b_rot = equatorial_baselines(itrs_bl, times, backend="astropy", xp=np)  # (n_time, nbl, 3)
    delay_imager = b_rot @ s_icrs  # (n_time, nbl)
    s_itrs = _itrs_unit_vectors(ra, dec, times)  # (n_time, 3)
    delay_ref = np.einsum("bi,ti->tb", itrs_bl, s_itrs)
    # C(t) is a pure Earth-orientation rotation, so it reproduces the full astropy ICRS->ITRS
    # source transform up to stellar aberration (the non-rotational ICRS<->GCRS term, ~20 arcsec):
    # ~3e-4 of the baseline length, far below the ~0.9 deg pixel. A real bug (axis/transpose/sign)
    # would be O(baseline) metres.
    np.testing.assert_allclose(delay_imager, delay_ref, rtol=3e-3, atol=1e-3)


def test_equatorial_baselines_native_not_implemented():
    with pytest.raises(NotImplementedError):
        equatorial_baselines(np.zeros((2, 3)), np.array([1.6e9]), backend="native")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py -k equatorial -v`
Expected: FAIL — `cannot import name 'equatorial_baselines'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/kremetart/utils/healpix_dft.py

def _icrs_to_itrs_matrices(times: np.ndarray) -> np.ndarray:
    """Per-timestamp ICRS->ITRS rotation matrices R(t), shape (n_time, 3, 3) (host, O(n_time)).

    Column ``i`` of ``R(t)`` is the ITRS image of the ``i``-th ICRS axis, so
    ``s_itrs(t) = R(t) @ s_icrs``. The axes are transformed as unit-sphere directions (no distance,
    so the transform is a pure rotation and avoids geocentric parallax) via the same astropy path as
    :func:`kremetart.utils.rephasing._itrs_unit_vectors`, folding in frame bias, precession,
    nutation and Earth rotation. Being a pure rotation it reproduces the full source transform only
    up to stellar aberration (~20 arcsec), negligible against the ~0.9 deg pixel.
    """
    import astropy.units as u
    from astropy.coordinates import ICRS, ITRS, UnitSphericalRepresentation
    from astropy.time import Time

    tt = Time(np.asarray(times), format="unix", scale="utc")
    # The three ICRS axes (+x, +y, +z) as directions at infinity.
    axes = ICRS(UnitSphericalRepresentation(lon=[0.0, 90.0, 0.0] * u.deg, lat=[0.0, 0.0, 90.0] * u.deg))
    mats = np.empty((tt.size, 3, 3), dtype=np.float64)
    for k in range(tt.size):
        itrs = axes.transform_to(ITRS(obstime=tt[k]))
        mats[k] = itrs.cartesian.xyz.value  # (component, axis) -> columns of R(t)
    return mats


def equatorial_baselines(itrs_baselines, times, *, backend: str = "astropy", xp: ModuleType = np):
    """Rotate fixed ITRS baselines into the equatorial frame for each timestamp.

    Args:
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors (e.g. from rephasing's ``_itrs_baselines``).
        times: ``(n_time,)`` unix-second timestamps.
        backend: ``"astropy"`` (the oracle, host-side) or ``"native"`` (GPU polynomial; later phase).
        xp: Array module for the returned array.

    Returns:
        ``(n_time, nbl, 3)`` equatorial-rotated baselines ``b_pq(t)`` in metres.
    """
    if backend == "astropy":
        b = np.asarray(itrs_baselines)
        rot = _icrs_to_itrs_matrices(times)  # (n_time, 3, 3)
        b_rot = np.einsum("bi,tik->tbk", b, rot)  # b_itrs @ R(t)
        return xp.asarray(b_rot)
    if backend == "native":
        raise NotImplementedError("GPU-native C(t) backend is a later phase; use backend='astropy'.")
    raise ValueError(f"unknown backend {backend!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py -k equatorial -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
.venv/bin/ruff check src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py --fix
git add src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
git commit -m "feat: add astropy C(t) baseline rotation for HEALPix imaging"
```

---

### Task 5: `image_frame` — end-to-end per-frame dirty image

**Files:**
- Modify: `src/kremetart/utils/healpix_dft.py`
- Test: `tests/test_healpix_dft.py`

Orchestrates the full path: `C(t)` rotation, flatten `(time, baseline)` into rows, `dirty_map`.
This holds all the operator logic so it is testable with `xp=np`; the Holoscan operator (Task 6)
is a thin tensor wrapper around it.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_healpix_dft.py
from kremetart.utils.healpix_dft import image_frame


def test_image_frame_recovers_source_through_ctime():
    """End-to-end: model a source per timestamp with C(t), image it back to the right pixel."""
    pytest.importorskip("astropy")
    rng = np.random.default_rng(5)
    nside = 16
    pix = make_pixel_grid(nside, xp=np)
    itrs_bl = rng.standard_normal((20, 3)) * 3.0
    times = np.array([1.6e9, 1.6e9 + 60.0, 1.6e9 + 120.0])
    freqs = np.array([1.575e9])
    src = 1500  # valid pixel index for nside=16 (npix=3072)

    b_rot = equatorial_baselines(itrs_bl, times, xp=np)  # (n_time, nbl, 3)
    n_time, nbl = b_rot.shape[:2]
    s = pix[src]
    vis = np.empty((n_time, nbl, 1), dtype=complex)
    for t in range(n_time):
        vis[t, :, 0] = np.exp(2j * np.pi * (freqs[0] / LIGHTSPEED) * (b_rot[t] @ s))
    wgt = np.ones((n_time, nbl, 1))

    dmap = image_frame(vis, wgt, times, itrs_bl, pix, freqs, xp=np)
    assert dmap.shape == (pix.shape[0],)
    assert int(np.argmax(dmap)) == src
    np.testing.assert_allclose(dmap[src], 1.0, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py::test_image_frame_recovers_source_through_ctime -v`
Expected: FAIL — `cannot import name 'image_frame'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/kremetart/utils/healpix_dft.py

def image_frame(vis, weights, times, itrs_baselines, pix_vec, freqs, *, ctime_backend: str = "astropy", xp: ModuleType = np):
    """Per-frame dirty image from unstopped residual visibilities.

    Rotates the ITRS baselines by ``C(t)``, flattens ``(time, baseline)`` into the row axis
    (the rotated baseline differs per timestamp), and adjoint-DFTs onto the fixed grid.

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
    n_time, nbl, nchan = vis.shape
    rows = b_rot.reshape(n_time * nbl, 3)
    vis_rows = xp.asarray(vis).reshape(n_time * nbl, nchan)
    wgt_rows = xp.asarray(weights).reshape(n_time * nbl, nchan)
    return dirty_map(vis_rows, wgt_rows, rows, pix_vec, freqs, xp=xp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_healpix_dft.py::test_image_frame_recovers_source_through_ctime -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite, format, lint, commit**

```bash
.venv/bin/python -m pytest tests/test_healpix_dft.py -v
.venv/bin/ruff format src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
.venv/bin/ruff check src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py --fix
git add src/kremetart/utils/healpix_dft.py tests/test_healpix_dft.py
git commit -m "feat: add end-to-end HEALPix image_frame (C(t) + dirty map)"
```

---

### Task 6: `HealpixDFTOperator` — thin Holoscan wrapper

**Files:**
- Create: `src/kremetart/operators/dft_healpix.py`

Mirrors `src/kremetart/operators/dft_lm.py` (cupy at module top, `setup`/`compute` ports). It only
shuffles GPU tensors into `image_frame(..., xp=cp)`; all tested logic lives in `healpix_dft.py`.
**Not** CI-tested — like the existing `dft_lm.py`/`dft_lm_tiled.py` operators it requires CuPy +
Holoscan + a GPU. Correctness is covered by the Task 1-5 pure-function tests it delegates to.

- [ ] **Step 1: Write the implementation**

```python
# src/kremetart/operators/dft_healpix.py
"""Holoscan operator: full-sky HEALPix dirty-map imager (GPU-resident, xp=cupy).

Thin wrapper around :func:`kremetart.utils.healpix_dft.image_frame`. Receives unstopped residual
visibilities in the raw-visibility layout plus their timestamps, forms the per-frame equatorial
baseline rotation C(t) on the host, and images onto a fixed full-sky equatorial HEALPix grid.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.utils.healpix_dft import image_frame, make_pixel_grid


class HealpixDFTOperator(Operator):
    """Adjoint (dirty-map) HEALPix DFT imager.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors (constant for the array).
        freqs: ``(nchan,)`` frequencies in Hz.
        nest: NESTED HEALPix ordering (default True).
        ctime_backend: ``C(t)`` backend ("astropy" now, "native" later).
    """

    def __init__(self, fragment, nside, itrs_baselines, freqs, *args, nest=True, ctime_backend="astropy", **kwargs):
        self.nside = nside
        self.itrs_baselines = cp.asnumpy(itrs_baselines)  # host; rotation runs on host per frame
        self.freqs = cp.asarray(freqs)
        self.nest = nest
        self.ctime_backend = ctime_backend
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Build the fixed equatorial pixel grid once, on device.
        self.pix_vec = make_pixel_grid(self.nside, nest=self.nest, xp=cp)

    def setup(self, spec: OperatorSpec):
        spec.input("VISIBILITY")
        spec.input("WEIGHT")
        spec.input("time")
        spec.output("cube")
        spec.output("time_out")
        spec.output("freq_out")

    def compute(self, op_input, op_output, context):
        vis = cp.asarray(op_input.receive("VISIBILITY"))  # (n_time, nbl, nchan)
        wgt = cp.asarray(op_input.receive("WEIGHT"))  # (n_time, nbl, nchan)
        times = cp.asarray(op_input.receive("time"))  # (n_time,)

        dmap = image_frame(
            vis, wgt, cp.asnumpy(times), self.itrs_baselines, self.pix_vec, self.freqs,
            ctime_backend=self.ctime_backend, xp=cp,
        )

        # Output layout: (ncorr=1, ntime_out=1, nfreq_out=1, npix)
        cube = dmap[None, None, None, :]
        time_out = cp.mean(times, keepdims=True)
        freq_out = cp.mean(self.freqs, keepdims=True)
        op_output.emit(hs.as_tensor(cube), "cube")
        op_output.emit(hs.as_tensor(time_out), "time_out")
        op_output.emit(hs.as_tensor(freq_out), "freq_out")
```

- [ ] **Step 2: Verify it imports where CuPy/Holoscan are available**

On a GPU machine: `python -c "from kremetart.operators.dft_healpix import HealpixDFTOperator"`
Expected: no error. (In CI without CuPy this import is expected to fail; that matches the existing
`dft_lm.py` operators, which carry no tests.)

- [ ] **Step 3: Format, lint, commit**

```bash
.venv/bin/ruff format src/kremetart/operators/dft_healpix.py
.venv/bin/ruff check src/kremetart/operators/dft_healpix.py --fix
git add src/kremetart/operators/dft_healpix.py
git commit -m "feat: add HealpixDFTOperator Holoscan wrapper"
```

---

## Self-Review

**Spec coverage:**
- §2 direction cosines → Task 1 (`make_pixel_grid` via `pix2vec`, NESTED).
- §3.1 `xp`-injectable core + §4 math (no `(n-1)`/`1/n`) → Tasks 2-3.
- §3.2 swappable `C(t)` (astropy now, native raises) → Task 4.
- §3.3 thin Holoscan operator, unstopped inputs → Tasks 5-6.
- §6 scalar single-pol, length-1 corr axis → operator `cube` shape `(1,1,1,npix)` (Task 6).
- §7 plain matmul (no tiling) → Tasks 2-3 use single matmul/einsum.
- §8 tests: adjointness (Task 2), grid sanity (Task 1), point source (Tasks 3, 5), `C(t)` consistency (Task 4). The GPU `xp`-parity test is explicitly deferred (out-of-scope/GPU-gated) — consistent with the spec.

**Placeholder scan:** none — every code/test step is complete.

**Type consistency:** `make_pixel_grid`, `dft_forward`, `dft_adjoint`, `dirty_map`, `equatorial_baselines`, `image_frame` signatures are identical across their definitions and call sites (tests, `image_frame`, operator). Row-axis flattening convention `(n_time*nbl, …)` is consistent between `image_frame` and the pure functions.

**Note on `requires-python`:** astropy/healpy run fine under the repo `.venv`; tests invoke `.venv/bin/python` directly (the `uv run` path is blocked by the tart `python_version >= '3.11'` markers, unrelated to this work).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-15-healpix-dft-operator.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
