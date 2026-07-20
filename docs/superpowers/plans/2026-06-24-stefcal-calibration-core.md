# StefCAL Calibration Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the acquisition StEFCal step of the Stage-1 calibration operator — a unit-flux ENU sky model plus an alternating per-antenna complex-gain solver — as validated `xp`-injectable library math.

**Architecture:** Two self-contained `utils/` modules (`skymodel.py`, `stefcal.py`), pure array math injected with `xp=numpy` (CPU/tests) or `xp=cupy` (future GPU operator), gated by a hermetic simulation round-trip. A third, opt-in task adds catalogue glue + a real-data sanity check against TART's own gain snapshot. No Holoscan, no `cli`/`core`/`cab`, no flux/EKF/subtraction (all deferred per the spec).

**Tech Stack:** Python 3.10+, NumPy, pytest, `uv`. Reference spec: `docs/superpowers/specs/2026-06-24-stefcal-calibration-core-design.md`.

## Global Constraints

- Python 3.10+; modern typing (`X | Y`, `list[int]`); type hints on every signature.
- `utils/` modules import heavy deps at module top (these are not `cli/`).
- Every function `xp`-injectable: keyword-only `xp: ModuleType = np`, default NumPy.
- Google-style docstrings (Args/Returns/Raises), concise.
- After every change: `uv run ruff format . && uv run ruff check . --fix` (non-negotiable).
- No test artifacts in the repo tree; the real-data check is opt-in (env-gated) and excluded from required CI.
- `LIGHTSPEED = 299792458.0` defined locally in `skymodel.py` (module stays decoupled from `healpix_dft`).
- ENU axes are `(East, North, Up)`; az measured from North toward East. Source/baseline sign convention is self-consistent for the simulation gate; reconciled with the correlator only in the opt-in real-data check.

---

## File Structure

- `src/kremetart/utils/skymodel.py` (create) — `enu_direction_cosines`, `model_visibilities`.
- `src/kremetart/utils/stefcal.py` (create) — `stefcal_solve`, `referenced_phases`.
- `src/kremetart/utils/satellites.py` (modify) — add `frame_source_directions` (Task 3 only).
- `tests/test_skymodel.py` (create).
- `tests/test_stefcal.py` (create).
- `tests/test_stefcal_realdata.py` (create, Task 3 only).

---

## Task 1: Sky-model coherency util

**Files:**
- Create: `src/kremetart/utils/skymodel.py`
- Test: `tests/test_skymodel.py`

**Interfaces:**
- Produces:
  - `enu_direction_cosines(az, el, *, xp=np) -> (nsrc, 3)` — az/el in **radians**, returns ENU unit vectors.
  - `model_visibilities(s_enu, bl_enu, freqs, *, xp=np) -> (nbl, nchan)` complex — unit-flux model `M_pq = Σ_s exp(2πi (ν/c) b_pq·ŝ_s)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skymodel.py`:

```python
"""Unit tests for the unit-flux ENU sky model."""

import numpy as np

from kremetart.utils.skymodel import LIGHTSPEED, enu_direction_cosines, model_visibilities


def test_enu_direction_cosines_cardinal_points():
    # az from North toward East; el is altitude above horizon.
    # zenith (el=90): straight up.
    np.testing.assert_allclose(enu_direction_cosines(0.0, np.pi / 2), [0.0, 0.0, 1.0], atol=1e-12)
    # az=90deg, el=0: due East on the horizon.
    np.testing.assert_allclose(enu_direction_cosines(np.pi / 2, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
    # az=0, el=0: due North on the horizon.
    np.testing.assert_allclose(enu_direction_cosines(0.0, 0.0), [0.0, 1.0, 0.0], atol=1e-12)


def test_enu_direction_cosines_unit_norm_and_shape():
    az = np.array([0.1, 1.0, 2.0, 3.0])
    el = np.array([0.2, 0.5, 0.9, 1.2])
    s = enu_direction_cosines(az, el)
    assert s.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-12)


def test_model_visibilities_single_source_analytic_fringe():
    bl = np.array([[10.0, 0.0, 0.0]])  # one 10 m baseline along East
    s = enu_direction_cosines(np.array([np.pi / 2]), np.array([0.0]))  # due East -> (1,0,0)
    freqs = np.array([1.5e9])
    M = model_visibilities(s, bl, freqs)
    expected = np.exp(2j * np.pi * (1.5e9 / LIGHTSPEED) * 10.0)  # b.s = 10
    assert M.shape == (1, 1)
    np.testing.assert_allclose(M[0, 0], expected, rtol=1e-10)


def test_model_visibilities_is_flux_one_superposition():
    rng = np.random.default_rng(0)
    bl = rng.uniform(-5, 5, size=(7, 3))
    az = rng.uniform(0, 2 * np.pi, 4)
    el = rng.uniform(0.1, 1.5, 4)
    s = enu_direction_cosines(az, el)
    freqs = np.array([1.575e9])
    M = model_visibilities(s, bl, freqs)
    # equals the per-source sum with unit weights
    per_src = np.stack(
        [model_visibilities(s[i : i + 1], bl, freqs)[:, 0] for i in range(4)], axis=0
    ).sum(axis=0)
    np.testing.assert_allclose(M[:, 0], per_src, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_skymodel.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kremetart.utils.skymodel'`.

- [ ] **Step 3: Write the implementation**

Create `src/kremetart/utils/skymodel.py`:

```python
"""Unit-flux sky model in the local ENU frame.

The acquisition StEFCal step (docs/superpowers/specs/2026-06-24-stefcal-calibration-core-design.md)
models every catalogued source at flux 1 and needs the geometric coherency of each source on each
baseline. Both functions are ``xp``-injectable (``xp=numpy`` on CPU, ``xp=cupy`` on GPU) and operate
purely in the ENU frame -- the equatorial rotation is an imaging concern, strictly downstream of
calibration. Decoupled from the imaging DFT: this evaluates over the ~100 visible sources, not the
full HEALPix grid.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

LIGHTSPEED = 299792458.0


def enu_direction_cosines(az, el, *, xp: ModuleType = np):
    """ENU unit vectors for source azimuth/elevation.

    Args:
        az: azimuth in radians, measured from North toward East. Scalar or ``(nsrc,)``.
        el: elevation (altitude above the horizon) in radians. Scalar or ``(nsrc,)``.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(..., 3)`` array of ``(East, North, Up)`` unit vectors; ``(3,)`` for scalar inputs.
    """
    az = xp.asarray(az)
    el = xp.asarray(el)
    cos_el = xp.cos(el)
    east = xp.sin(az) * cos_el
    north = xp.cos(az) * cos_el
    up = xp.sin(el)
    return xp.stack([east, north, up], axis=-1)


def model_visibilities(s_enu, bl_enu, freqs, *, xp: ModuleType = np):
    """Unit-flux model visibilities ``M_pq = sum_s exp(2*pi*i*(nu/c)*b_pq . s_s)``.

    Args:
        s_enu: ``(nsrc, 3)`` source ENU unit vectors (e.g. from :func:`enu_direction_cosines`).
        bl_enu: ``(nbl, 3)`` ENU baseline vectors in metres.
        freqs: ``(nchan,)`` frequencies in Hz.
        xp: array module.

    Returns:
        ``(nbl, nchan)`` complex unit-flux model visibilities.
    """
    s_enu = xp.asarray(s_enu)
    bl_enu = xp.asarray(bl_enu)
    inv_wl = xp.asarray(freqs) / LIGHTSPEED  # (nchan,) cycles per metre
    delay = bl_enu @ s_enu.T  # (nbl, nsrc) metres
    phase = 2.0 * xp.pi * inv_wl[None, :, None] * delay[:, None, :]  # (nbl, nchan, nsrc)
    return xp.exp(1j * phase).sum(axis=-1)  # (nbl, nchan)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_skymodel.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/skymodel.py tests/test_skymodel.py
git commit -m "feat: add unit-flux ENU sky model for calibration"
```

---

## Task 2: StEFCal solver

**Files:**
- Create: `src/kremetart/utils/stefcal.py`
- Test: `tests/test_stefcal.py`

**Interfaces:**
- Consumes: `enu_direction_cosines`, `model_visibilities` from Task 1 (tests only).
- Produces:
  - `stefcal_solve(vis, model, a1, a2, n_ant, *, ref_ant=0, weight=None, g0=None, max_iter=100, tol=1e-8, xp=np) -> (gains: (n_ant,) complex, info: dict)`. `info` has keys `iterations: int`, `converged: bool`, `max_change: float`. Dead antennas come out `NaN`; gauge fixed so `gains[ref_ant] == 1`.
  - `referenced_phases(gains, ref_ant, *, xp=np) -> (n_ant,)` real — `angle(g_p) - angle(g_ref)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stefcal.py`:

```python
"""Unit tests for the acquisition StEFCal solver (simulation round-trip gate)."""

import numpy as np
import pytest

from kremetart.utils.skymodel import enu_direction_cosines, model_visibilities
from kremetart.utils.stefcal import referenced_phases, stefcal_solve

_FREQS = np.array([1.575e9])
_N_ANT = 24


def _layout(seed=0):
    """A planar 24-antenna ENU layout and its baseline antenna-index arrays."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(_N_ANT, 3))
    pos[:, 2] = 0.0  # TART elements are coplanar (Up = 0)
    a1, a2 = np.triu_indices(_N_ANT, k=1)
    bl = pos[a1] - pos[a2]
    return a1, a2, bl


def _sky(seed=1, nsrc=30):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0.0, 2.0 * np.pi, nsrc)
    el = rng.uniform(np.radians(20.0), np.radians(90.0), nsrc)
    return enu_direction_cosines(az, el)


def _true_gains(seed=2):
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.5, 2.0, _N_ANT)
    phase = rng.uniform(-np.pi, np.pi, _N_ANT)
    return amp * np.exp(1j * phase)


def _synth(g, model, a1, a2, noise=0.0, seed=3):
    vis = (g[a1] * model[:, 0] * np.conj(g[a2]))[:, None]
    if noise:
        rng = np.random.default_rng(seed)
        vis = vis + noise * (rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape))
    return vis


def test_stefcal_recovers_gains_noiseless():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    g_hat, info = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)
    assert info["converged"]
    g_true_ref = g_true / g_true[0]  # same gauge (ref antenna 0 -> 1)
    np.testing.assert_allclose(g_hat, g_true_ref, atol=1e-6)
    np.testing.assert_allclose(g_hat[0], 1.0 + 0.0j, atol=1e-12)


def test_referenced_phases_match_truth_up_to_gauge():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    g_hat, _ = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)
    got = referenced_phases(g_hat, 0)
    want = np.angle(g_true) - np.angle(g_true[0])
    # compare as unit complex to sidestep 2pi wrapping
    np.testing.assert_allclose(np.exp(1j * got), np.exp(1j * want), atol=1e-6)


def test_gauge_invariant_to_global_complex_scale():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    c = 0.7 * np.exp(1j * 1.1)  # arbitrary global complex factor
    vis_a = _synth(g_true, model, a1, a2)
    vis_b = _synth(c * g_true, model, a1, a2)  # V scales by |c|^2; phases referenced-invariant
    pa = referenced_phases(stefcal_solve(vis_a, model, a1, a2, _N_ANT, ref_ant=0)[0], 0)
    pb = referenced_phases(stefcal_solve(vis_b, model, a1, a2, _N_ANT, ref_ant=0)[0], 0)
    np.testing.assert_allclose(np.exp(1j * pa), np.exp(1j * pb), atol=1e-6)


def test_stefcal_flags_dead_antenna():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    dead = 5
    touch = (a1 == dead) | (a2 == dead)
    weight = np.ones((a1.size, 1))
    weight[touch] = 0.0
    vis[touch] = 0.0  # reader zeroes dead baselines too
    g_hat, info = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0, weight=weight)
    assert np.isnan(g_hat[dead])
    live = np.arange(_N_ANT) != dead
    assert np.all(np.isfinite(g_hat[live]))
    np.testing.assert_allclose(g_hat[live], (g_true / g_true[0])[live], atol=1e-6)


def test_stefcal_recovers_with_noise():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2, noise=1e-3)
    g_hat, _ = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)
    np.testing.assert_allclose(g_hat, g_true / g_true[0], atol=5e-3)


def test_stefcal_ref_dead_raises():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    vis = _synth(_true_gains(), model, a1, a2)
    touch = (a1 == 0) | (a2 == 0)
    weight = np.ones((a1.size, 1))
    weight[touch] = 0.0
    with pytest.raises(ValueError):
        stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0, weight=weight)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stefcal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kremetart.utils.stefcal'`.

- [ ] **Step 3: Write the implementation**

Create `src/kremetart/utils/stefcal.py`:

```python
"""Acquisition StEFCal: alternating per-antenna complex-gain least squares.

The cold-start solver of the Stage-1 calibration operator
(docs/superpowers/specs/2026-06-24-stefcal-calibration-core-design.md). Holding all other antenna
gains fixed, ``V_pq = g_p M_pq conj(g_q)`` is linear in ``g_p``; alternating that antenna-by-antenna
gives a robust, phase-wrap-free cold start. Every source is modelled at unit flux, so the solved
amplitudes are unreliable and only the gauge-referenced phases are kept downstream. ``xp``-injectable
(``xp=numpy`` CPU / ``xp=cupy`` GPU); the per-antenna reduction is a segment sum over the bipartite
baseline-antenna incidence.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def _seg_sum_complex(idx, vals, n: int, *, xp: ModuleType):
    """Per-segment complex sum: ``out[k] = sum(vals[idx == k])`` over ``n`` segments."""
    real = xp.bincount(idx, weights=vals.real, minlength=n)
    imag = xp.bincount(idx, weights=vals.imag, minlength=n)
    return real + 1j * imag


def stefcal_solve(
    vis,
    model,
    a1,
    a2,
    n_ant: int,
    *,
    ref_ant: int = 0,
    weight=None,
    g0=None,
    max_iter: int = 100,
    tol: float = 1e-8,
    xp: ModuleType = np,
):
    """Solve per-antenna complex gains by alternating least squares (StEFCal).

    Args:
        vis: ``(nbl, nchan)`` observed visibilities.
        model: ``(nbl, nchan)`` unit-flux model visibilities (:func:`...skymodel.model_visibilities`).
        a1: ``(nbl,)`` int antenna index of the first antenna of each baseline.
        a2: ``(nbl,)`` int antenna index of the second antenna of each baseline.
        n_ant: number of antennas.
        ref_ant: reference antenna whose gain is pinned to 1 (fixes both gauges). Must be live.
        weight: optional ``(nbl,)`` or ``(nbl, nchan)`` real weights; ``0`` flags a baseline out.
        g0: optional ``(n_ant,)`` complex initial gains (warm start); defaults to unity.
        max_iter: maximum alternating iterations.
        tol: convergence threshold on the max relative gain change over live antennas.
        xp: array module.

    Returns:
        ``(gains, info)``: ``gains`` is ``(n_ant,)`` complex with ``gains[ref_ant] == 1`` and dead
        antennas ``NaN``; ``info`` has ``iterations`` (int), ``converged`` (bool), ``max_change``
        (float, the last iteration's max relative change).

    Raises:
        ValueError: if ``ref_ant`` has no live (non-zero-weight) baselines.
    """
    vis = xp.asarray(vis)
    model = xp.asarray(model)
    a1 = xp.asarray(a1)
    a2 = xp.asarray(a2)
    nbl, nchan = vis.shape
    if weight is None:
        weight = xp.ones((nbl, nchan))
    else:
        weight = xp.asarray(weight)
        if weight.ndim == 1:
            weight = weight[:, None]
        weight = xp.broadcast_to(weight, (nbl, nchan))

    # Directed baselines: forward (p=a1, partner=a2, V, M) + reverse (p=a2, partner=a1, conjV, conjM).
    p_idx = xp.concatenate([a1, a2])
    q_idx = xp.concatenate([a2, a1])
    vdir = xp.concatenate([vis, xp.conj(vis)], axis=0)  # (2*nbl, nchan)
    mdir = xp.concatenate([model, xp.conj(model)], axis=0)
    wdir = xp.concatenate([weight, weight], axis=0)

    # Live antennas: those carrying at least one non-zero-weight baseline.
    deg = xp.bincount(p_idx, weights=wdir.sum(axis=1), minlength=n_ant)  # (n_ant,) total weight
    live = deg > 0
    if not bool(live[ref_ant]):
        raise ValueError(f"ref_ant {ref_ant} has no live baselines")

    g = xp.ones(n_ant, dtype=xp.complex128) if g0 is None else xp.asarray(g0).astype(xp.complex128)
    converged = False
    change = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        z = mdir * xp.conj(g[q_idx])[:, None]  # (2*nbl, nchan): V_dir = g[p] * z
        num = _seg_sum_complex(p_idx, (wdir * xp.conj(z) * vdir).sum(axis=1), n_ant, xp=xp)
        den = xp.bincount(p_idx, weights=(wdir * (z.real**2 + z.imag**2)).sum(axis=1), minlength=n_ant)
        g_new = xp.where(den > 0, num / xp.where(den > 0, den, 1.0), g)
        if it % 2 == 0:
            g_new = 0.5 * (g_new + g)  # StEFCal even-iteration stabiliser
        delta = xp.abs(g_new - g)
        scale = xp.where(xp.abs(g) > 1e-12, xp.abs(g), 1.0)
        change = float(xp.max(xp.where(live, delta / scale, 0.0)))
        g = g_new
        if change < tol:
            converged = True
            break

    nan_c = complex(float("nan"), float("nan"))
    g = xp.where(live, g, nan_c)
    g = g / g[ref_ant]  # gauge: g_ref -> 1 (live, finite)
    return g, {"iterations": it, "converged": converged, "max_change": change}


def referenced_phases(gains, ref_ant: int, *, xp: ModuleType = np):
    """Gauge-referenced gain phases (amplitudes discarded).

    Args:
        gains: ``(n_ant,)`` complex gains.
        ref_ant: reference antenna; its phase is subtracted from all phases.
        xp: array module.

    Returns:
        ``(n_ant,)`` real phases ``angle(g_p) - angle(g_ref)`` in radians (``NaN`` for dead antennas).
    """
    gains = xp.asarray(gains)
    return xp.angle(gains) - xp.angle(gains[ref_ant])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stefcal.py -q`
Expected: PASS (7 passed). If `test_stefcal_recovers_gains_noiseless` is marginally off at `atol=1e-6`, the cause is iteration count, not correctness — confirm by raising `max_iter` in the call; do not loosen the assertion without that check.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/stefcal.py tests/test_stefcal.py
git commit -m "feat: add acquisition StefCAL solver with gauge fixing"
```

---

## Task 3 (opt-in): catalogue glue + real-data sanity check

This task wires the solver to real TART data and TART's own gain snapshot. It is **secondary** and env-gated; it must never gate required CI. Skip it if you only need the validated core.

**Files:**
- Modify: `src/kremetart/utils/satellites.py`
- Test: `tests/test_stefcal_realdata.py`

**Interfaces:**
- Consumes: `read_hdf_as_msv4`, `partition_datatree`; the existing `_tart_api_fetch`, `_frame_times_and_site`, `_load_catalog_cache` in `satellites.py`; `enu_direction_cosines`, `model_visibilities`, `stefcal_solve`, `referenced_phases`.
- Produces: `frame_source_directions(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch, cache_path=None, nframes=None) -> list[list[tuple[str, float, float]]]` — per frame, a list of `(name, az_rad, el_rad)` aligned 1:1 with the imaged frame order.

- [ ] **Step 1: Read the existing per-frame catalogue logic**

Read `src/kremetart/utils/satellites.py` lines ~196-214 (the `times_unix` / `per_frame` construction inside `satellite_tracks`). `frame_source_directions` reuses the same cache-aware fetch loop but returns ENU az/el (radians) instead of grouping into ICRS tracks. Confirm `_tart_api_fetch`/`_load_catalog_cache` return source dicts with `az`/`el` in **degrees**.

- [ ] **Step 2: Write the failing test**

Create `tests/test_stefcal_realdata.py`:

```python
"""Opt-in real-data sanity check: StefCAL phases vs TART's own gain snapshot.

Env-gated (KREMETART_REALDATA=1) and excluded from required CI: it pins the ENU az / baseline-sign
convention against TART's solution, which may need iteration. Uses the bundled catalogue cache so it
never queries the (slow) TART API.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KREMETART_REALDATA") != "1", reason="set KREMETART_REALDATA=1 to run"
)


def test_stefcal_phases_track_tart_snapshot(ref_hdf, catalog_cache, catalog_elevation):
    from kremetart.utils import partition_datatree
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.satellites import frame_source_directions
    from kremetart.utils.skymodel import enu_direction_cosines, model_visibilities
    from kremetart.utils.stefcal import referenced_phases, stefcal_solve

    node = partition_datatree(read_hdf_as_msv4(ref_hdf))
    main = node.ds
    antenna = node["antenna_xds"].to_dataset(inherit=False)
    names = list(antenna.antenna_name.values)
    index = {n: i for i, n in enumerate(names)}
    a1 = np.array([index[n] for n in main.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in main.baseline_antenna2_name.values])
    enu = antenna.ANTENNA_POSITION_ENU.values
    bl = enu[a1] - enu[a2]
    n_ant = len(names)
    freqs = np.asarray(main.frequency.values)

    vis = np.asarray(main.VISIBILITY.values)[0, :, :, 0]  # (nbl, nchan), first frame
    weight = np.asarray(main.WEIGHT.values)[0, :, :, 0]

    per_frame = frame_source_directions(
        [ref_hdf], catalog_elevation, cache_path=catalog_cache, nframes=1
    )
    az = np.array([a for _, a, _ in per_frame[0]])
    el = np.array([e for _, _, e in per_frame[0]])
    s = enu_direction_cosines(az, el)
    model = model_visibilities(s, bl, freqs)

    gain = node["gain_xds"].to_dataset(inherit=False)
    dead = np.where(np.asarray(gain.ANTENNA_FLAG.values))[0]
    ref = int(np.setdiff1d(np.arange(n_ant), dead)[0])  # first live antenna

    g_hat, info = stefcal_solve(vis, model, a1, a2, n_ant, ref_ant=ref, weight=weight)
    assert info["converged"]

    got = referenced_phases(g_hat, ref)
    snap = np.angle(np.asarray(gain.GAIN.values))
    want = snap - snap[ref]
    live = np.isfinite(g_hat)
    # circular distance, robust to sign/convention offset; generous bound (diagnostic, not a gate).
    d = np.angle(np.exp(1j * (got[live] - want[live])))
    print(f"[realdata] circular RMS phase diff = {np.sqrt(np.mean(d**2)):.3f} rad over {live.sum()} ant")
    assert np.all(np.isfinite(got[live]))
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_stefcal_realdata.py -q`
Expected: FAIL — `ImportError: cannot import name 'frame_source_directions'`.
(If it instead reports `1 skipped`, you forgot `KREMETART_REALDATA=1` — set it: `KREMETART_REALDATA=1 uv run pytest tests/test_stefcal_realdata.py -q`.)

- [ ] **Step 4: Add `frame_source_directions` to `satellites.py`**

Append to `src/kremetart/utils/satellites.py`:

```python
def frame_source_directions(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch, cache_path=None, nframes=None):
    """Per-frame ``(name, az_rad, el_rad)`` source lists aligned 1:1 with the imaged frame order.

    Reuses the cache-aware fetch loop of :func:`satellite_tracks` but returns ENU az/el (radians)
    for the calibration sky model rather than grouping into ICRS tracks. Catalogue az/el are stored
    in degrees and converted to radians here.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (same order as the imaged frames).
        elevation_deg: elevation cutoff (deg) for catalogue sources.
        fetch: ``callable(lon, lat, datestr, elevation_deg) -> list[dict]``; injectable for tests.
        cache_path: optional catalogue cache zarr path; ``None`` disables caching.
        nframes: optional cap on the number of leading frames processed.

    Returns:
        ``list`` (one entry per frame) of ``list[(name, az_rad, el_rad)]``.
    """
    times_unix, lat, lon, _alt = _frame_times_and_site(hdf_paths)
    if nframes is not None:
        times_unix = times_unix[:nframes]
    datestrs = [datetime.datetime.fromtimestamp(float(t), tz=datetime.timezone.utc).isoformat() for t in times_unix]

    cached = _load_catalog_cache(cache_path, lat, lon, elevation_deg) if cache_path else None
    out: list[list[tuple[str, float, float]]] = []
    for datestr in datestrs:
        sources = cached[datestr] if (cached is not None and datestr in cached) else fetch(lon, lat, datestr, elevation_deg)
        out.append([(str(s["name"]), np.radians(float(s["az"])), np.radians(float(s["el"]))) for s in sources])
    return out
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `KREMETART_REALDATA=1 uv run pytest tests/test_stefcal_realdata.py -q -s`
Expected: PASS (prints the circular RMS phase diff). A *large* RMS is a convention mismatch to chase down (baseline sign `b = pos(a1)-pos(a2)`, az handedness) — the assertion only requires finite phases, so the test passes while surfacing the number. Also confirm the default suite still skips it: `uv run pytest tests/test_stefcal_realdata.py -q` → `1 skipped`.

- [ ] **Step 6: Confirm no regression in existing catalogue tests, then commit**

```bash
uv run pytest tests/test_satellites.py -q
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/satellites.py tests/test_stefcal_realdata.py
git commit -m "feat: add catalogue glue + opt-in StefCAL real-data check"
```

---

## Final verification (after the chosen tasks)

- [ ] Run the new suite and a structure guard:

Run: `uv run pytest tests/test_skymodel.py tests/test_stefcal.py tests/test_structure.py -q`
Expected: PASS. `test_structure.py` confirms the new modules live under `utils/` (not `core/`) and that no `cli`/`core`/`cab` invariant was disturbed.

- [ ] Confirm clean format/lint:

Run: `uv run ruff format --check . && uv run ruff check .`
Expected: no changes, no errors.

---

## Self-Review notes

- **Spec coverage:** §3 modules → Tasks 1 (`skymodel`) & 2 (`stefcal`); §4.1/§4.2 functions and edge cases → Tasks 1/2 (single-source fringe, gauge, flagged antenna, ref-dead raise, no-data guard via `den==0`); §5.1 simulation gate → Task 2 tests; §5.2 real-data check + catalogue glue → Task 3; §6 dev workflow → every task's format/lint/commit step. Deferred items (operator, equatorial-rotation move, flux/EKF) are correctly absent.
- **Placeholder scan:** all code blocks are complete; no TBD/TODO.
- **Type consistency:** `stefcal_solve`/`referenced_phases`/`model_visibilities`/`enu_direction_cosines` signatures are identical across the plan body, tests, and Task 3 consumer.
