# Reweighted-L1 FISTA Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic, `xp`-injectable FISTA solver with backtracking and an outer reweighted-L1 loop in `src/kremetart/opt/fista.py`, so later self-cal code can recover non-negative source fluxes.

**Architecture:** Operator-agnostic solver exactly like `opt/cg.py` — the forward operator `A` and its adjoint `AH` are plain callables, data `y` is complex, the unknown `x` is real. A private inner routine `_fista_single` runs FISTA-with-backtracking for one fixed weight vector; the public `fista` wraps it in the Candès–Wakin–Boyd reweighting loop and returns `(x, info)`.

**Tech Stack:** Python 3.10+, numpy (cupy-compatible via `xp` injection), pytest, ruff.

## Global Constraints

- Python 3.10+; modern typing (`X | Y`, `list[int]`).
- `opt/` follows `utils/` import rules: `import numpy as np` and stdlib at module top — **no** lazy/in-function imports.
- `xp`-injectable: every array op goes through the `xp` parameter (default `np`); no bare `np.` inside the algorithms.
- ruff: line length 120; `select = E/F/I/N/W`; `N802/N803/N812` ignored but **`N806` is NOT** — argument names `A`/`AH` are allowed, but **every local must be lowercase** (Lipschitz local is `lipschitz`, never `L`). Use `lam`, never `lambda`.
- Google-style docstrings (Args/Returns), concise, matching `cg.py`.
- Solver is a utility like `cg` — **no** `cli/`, `core/`, or cab files.
- After every code change run: `uv run ruff format . && uv run ruff check . --fix`.
- Conventional Commits, single type, imperative, < 72 chars; sign-off line `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Tests use `xp=np`, stay offline, write no repo artefacts.

## File Structure

| File | Responsibility |
|---|---|
| `src/kremetart/opt/fista.py` (create) | `_soft_threshold` prox helper, `_objective` helper, `_fista_single` inner solver, public `fista`. |
| `src/kremetart/opt/__init__.py` (modify) | Export `fista` alongside `cg`. |
| `tests/test_fista.py` (create) | All unit + integration tests. |

---

### Task 1: Proximal operator (`_soft_threshold`)

**Files:**
- Create: `src/kremetart/opt/fista.py`
- Test: `tests/test_fista.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `_soft_threshold(z, tau, *, positive: bool, xp=np) -> ndarray` — element-wise prox of the (optionally non-negative) weighted-L1 penalty. `positive=True` → `max(z - tau, 0)`; `positive=False` → `sign(z) * max(|z| - tau, 0)`. `tau` is a scalar or array broadcastable to `z`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fista.py`:

```python
"""Tests for the reweighted-L1 FISTA solver (src/kremetart/opt/fista.py)."""

from __future__ import annotations

import numpy as np

from kremetart.opt.fista import _soft_threshold


def test_soft_threshold_signed():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    out = _soft_threshold(z, 1.0, positive=False, xp=np)
    np.testing.assert_allclose(out, [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_soft_threshold_positive():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    out = _soft_threshold(z, 1.0, positive=True, xp=np)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 0.0, 2.0])


def test_soft_threshold_vector_tau():
    z = np.array([2.0, 2.0, 2.0])
    tau = np.array([0.5, 1.0, 3.0])
    out = _soft_threshold(z, tau, positive=False, xp=np)
    np.testing.assert_allclose(out, [1.5, 1.0, 0.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fista.py -v`
Expected: FAIL — `ImportError: cannot import name '_soft_threshold'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/kremetart/opt/fista.py`:

```python
"""Reweighted-L1 FISTA solver (xp-injectable; CPU via numpy, GPU via cupy).

Generic and operator-agnostic, like :mod:`kremetart.opt.cg`: the forward operator ``A`` and its
Hermitian adjoint ``AH`` are plain callables ``x -> A @ x`` and ``r -> Aᴴ @ r``. Minimises, over a
**real** vector ``x``, ``0.5 ||W^½ (A x - y)||² + λ Σ wᵢ |xᵢ|`` with FISTA + backtracking and an
outer Candès–Wakin–Boyd reweighting loop. Used at the outset of self-calibration to recover
non-negative source fluxes from calibrated visibilities; the caller wires ``A`` to the imaging DFT
or the per-source sky model (out of scope here, mirroring how the imager wires up ``cg``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np


def _soft_threshold(z, tau, *, positive: bool, xp: ModuleType = np):
    """Proximal operator of the (optionally non-negative) weighted-L1 penalty.

    Args:
        z: real array, the point at which to evaluate the prox.
        tau: scalar or array (broadcastable to ``z``) of non-negative thresholds.
        positive: if True, clamp at 0 (prox of L1 + non-negativity indicator).
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``max(z - tau, 0)`` if ``positive`` else ``sign(z) * max(|z| - tau, 0)``.
    """
    if positive:
        return xp.maximum(z - tau, 0.0)
    return xp.sign(z) * xp.maximum(xp.abs(z) - tau, 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fista.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/opt/fista.py tests/test_fista.py
git commit -m "feat: add weighted-L1 soft-threshold prox

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Inner FISTA with backtracking (single weight)

**Files:**
- Modify: `src/kremetart/opt/fista.py`
- Test: `tests/test_fista.py`

**Interfaces:**
- Consumes: `_soft_threshold` (Task 1).
- Produces:
  - `_objective(A, y, weight, x, lam, xp) -> float` — `0.5 Σ W|A x − y|² + λ Σ|x|`.
  - `_fista_single(A, AH, y, w, x_start, *, lam, weight, positive, L0, eta, max_iter, tol, xp) -> tuple[ndarray, int, bool, float]` returning `(x, iters, converged, lipschitz)`.
  - `fista(A, AH, y, *, lam, weight=None, x0=None, positive=True, L0=None, eta=2.0, max_iter=500, tol=1e-5, max_reweight=0, reweight_eps=1e-3, reweight_tol=1e-3, xp=np) -> tuple[ndarray, dict]`. In this task `fista` performs a **single** unweighted solve (ignores the reweighting params); the outer loop is added in Task 3. `info` keys: `iterations` (`list[int]`), `reweights` (`int`), `objective` (`float`), `lipschitz` (`float`), `converged` (`bool`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fista.py`:

```python
from kremetart.opt.fista import fista


def _identity_ops():
    return (lambda x: x), (lambda r: r)


def test_identity_recovers_soft_threshold():
    # A = I, real data: argmin 0.5||x - y||² + λ||x||₁  ==  soft_threshold(y, λ)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    A, AH = _identity_ops()
    x, info = fista(A, AH, y, lam=0.3, positive=False, tol=1e-10, max_iter=2000)
    expect = np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0)
    np.testing.assert_allclose(x, expect, atol=1e-5)
    assert info["converged"]


def test_backtracking_recovers_from_tiny_l0():
    # A badly underestimated L0 must still converge (backtracking inflates lipschitz).
    rng = np.random.default_rng(1)
    y = rng.standard_normal(50)
    A, AH = _identity_ops()
    x, info = fista(A, AH, y, lam=0.3, positive=False, L0=1e-6, tol=1e-10, max_iter=2000)
    expect = np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0)
    np.testing.assert_allclose(x, expect, atol=1e-5)
    assert info["lipschitz"] > 1e-6  # grew toward the true Lipschitz constant (~1)


def test_positive_constraint():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(50)  # has negative entries
    A, AH = _identity_ops()
    x, _ = fista(A, AH, y, lam=0.1, positive=True, tol=1e-10, max_iter=2000)
    assert np.all(x >= 0.0)
    np.testing.assert_allclose(x, np.maximum(y - 0.1, 0.0), atol=1e-5)


def test_zero_data_returns_zeros():
    A, AH = _identity_ops()
    y = np.zeros(10)
    x, info = fista(A, AH, y, lam=0.5, positive=True)
    np.testing.assert_allclose(x, 0.0)
    assert info["converged"]
    assert info["reweights"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fista.py -v`
Expected: FAIL — `ImportError: cannot import name 'fista'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/kremetart/opt/fista.py`:

```python
def _objective(A: Callable, y, weight, x, lam: float, xp: ModuleType) -> float:
    """Full objective ``0.5 Σ W|A x − y|² + λ Σ|x|`` (plain L1, for diagnostics)."""
    r = A(x) - y
    sq = xp.abs(r) ** 2
    if weight is not None:
        sq = weight * sq
    return 0.5 * float(sq.sum()) + lam * float(xp.abs(x).sum())


def _fista_single(
    A: Callable,
    AH: Callable,
    y,
    w,
    x_start,
    *,
    lam: float,
    weight,
    positive: bool,
    L0: float | None,
    eta: float,
    max_iter: int,
    tol: float,
    xp: ModuleType,
):
    """FISTA with backtracking for one fixed reweighting vector ``w``.

    Returns:
        ``(x, iters, converged, lipschitz)``.
    """

    def fval(r):
        sq = xp.abs(r) ** 2
        if weight is not None:
            sq = weight * sq
        return 0.5 * float(sq.sum())

    def grad_at(r):
        wr = r if weight is None else weight * r
        return xp.real(AH(wr))  # x is real -> take the real part of the complex adjoint

    threshold_w = lam * w  # per-element L1 weight (λ · reweight weight)
    lipschitz = 1.0 if L0 is None else float(L0)
    x = xp.asarray(x_start).copy()
    x_prev = x.copy()
    v = x.copy()
    t = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        r_v = A(v) - y
        f_v = fval(r_v)
        g_v = grad_at(r_v)
        bt = 0
        while True:
            z = v - g_v / lipschitz
            x_new = _soft_threshold(z, threshold_w / lipschitz, positive=positive, xp=xp)
            diff = x_new - v
            rhs = f_v + float((diff * g_v).sum()) + 0.5 * lipschitz * float((diff * diff).sum())
            if fval(A(x_new) - y) <= rhs + 1e-12 or bt >= 100:
                break
            lipschitz *= eta
            bt += 1
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        v = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x_prev = x
        x = x_new
        t = t_new
        change = float(xp.linalg.norm(x - x_prev))
        scale = max(float(xp.linalg.norm(x)), 1e-12)
        if change / scale < tol:
            converged = True
            break
    return x, iters, converged, lipschitz


def fista(
    A: Callable,
    AH: Callable,
    y,
    *,
    lam: float,
    weight=None,
    x0=None,
    positive: bool = True,
    L0: float | None = None,
    eta: float = 2.0,
    max_iter: int = 500,
    tol: float = 1e-5,
    max_reweight: int = 0,
    reweight_eps: float = 1e-3,
    reweight_tol: float = 1e-3,
    xp: ModuleType = np,
):
    """Minimise ``0.5 ||W^½(A x − y)||² + λ Σ wᵢ|xᵢ|`` over real ``x`` via reweighted-L1 FISTA.

    Args:
        A: forward operator, a callable ``x -> A @ x`` (real ``x`` -> complex output).
        AH: Hermitian adjoint of ``A``, a callable ``r -> Aᴴ @ r``.
        y: ``(m,)`` complex data (any shape ``A`` returns).
        lam: L1 strength ``λ`` (``0`` allowed -> plain least squares).
        weight: optional real inverse-variance ``W`` broadcastable to ``y``; ``None`` -> ones.
        x0: optional real warm start; ``None`` -> zeros.
        positive: enforce ``x >= 0`` in the prox.
        L0: initial Lipschitz estimate; ``None`` -> ``1.0``.
        eta: backtracking growth factor (> 1).
        max_iter: max inner FISTA iterations per reweight round.
        tol: inner relative-change stopping tolerance.
        max_reweight: outer reweighting rounds (``0`` -> plain L1).
        reweight_eps: ``ε`` in ``wᵢ = 1/(|xᵢ| + ε)``.
        reweight_tol: outer between-round relative-change stop.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(x, info)``: real ``x`` and a dict with ``iterations``, ``reweights``, ``objective``,
        ``lipschitz``, ``converged``.
    """
    y = xp.asarray(y)
    weight = None if weight is None else xp.asarray(weight)
    real_dtype = y.real.dtype
    if x0 is None:
        x = xp.zeros(AH(y).shape, dtype=real_dtype)
    else:
        x = xp.asarray(x0, dtype=real_dtype).copy()
    w = xp.ones(x.shape, dtype=real_dtype)
    x, iters, converged, lipschitz = _fista_single(
        A, AH, y, w, x, lam=lam, weight=weight, positive=positive,
        L0=L0, eta=eta, max_iter=max_iter, tol=tol, xp=xp,
    )
    info = {
        "iterations": [iters],
        "reweights": 0,
        "objective": _objective(A, y, weight, x, lam, xp),
        "lipschitz": lipschitz,
        "converged": converged,
    }
    return x, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fista.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/opt/fista.py tests/test_fista.py
git commit -m "feat: add inner FISTA with backtracking line search

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Outer reweighted-L1 loop

**Files:**
- Modify: `src/kremetart/opt/fista.py` (replace the body of `fista` from Task 2)
- Test: `tests/test_fista.py`

**Interfaces:**
- Consumes: `_fista_single`, `_objective` (Task 2).
- Produces: `fista` extended so `max_reweight > 0` runs the Candès–Wakin–Boyd loop — solve, set `w = 1/(|x| + reweight_eps)`, repeat up to `max_reweight` rounds, breaking early when the between-round relative change `< reweight_tol`. `info["iterations"]` gets one entry per round; `info["reweights"] = len(iterations) - 1`. Signature and other `info` keys unchanged from Task 2.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fista.py`:

```python
def _gaussian_operator(rng, m, n):
    mat = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    return mat, (lambda x: mat @ x), (lambda r: mat.conj().T @ r)


def test_reweighting_debiases_sparse_recovery():
    rng = np.random.default_rng(3)
    n, m = 20, 60
    mat, A, AH = _gaussian_operator(rng, m, n)
    x_true = np.zeros(n)
    x_true[[2, 7, 15]] = [1.0, 2.0, 1.5]
    y = mat @ x_true

    x_l1, info_l1 = fista(A, AH, y, lam=0.2, positive=True, tol=1e-9, max_iter=4000)
    x_rw, info_rw = fista(
        A, AH, y, lam=0.2, positive=True, tol=1e-9, max_iter=4000,
        max_reweight=8, reweight_eps=1e-3,
    )

    err_l1 = np.linalg.norm(x_l1 - x_true)
    err_rw = np.linalg.norm(x_rw - x_true)
    assert err_rw <= err_l1 + 1e-9     # reweighting never worse than plain L1
    assert err_rw < 1e-2               # and essentially exact here
    assert info_l1["reweights"] == 0
    assert info_rw["reweights"] >= 1
    np.testing.assert_array_equal(np.where(x_rw > 1e-3)[0], [2, 7, 15])  # exact support
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fista.py::test_reweighting_debiases_sparse_recovery -v`
Expected: FAIL — `max_reweight` is ignored in Task 2, so `x_rw == x_l1` and `err_rw < 1e-2` fails (plain L1 is shrinkage-biased) and `info_rw["reweights"] >= 1` fails.

- [ ] **Step 3: Replace `fista` with the reweighting version**

In `src/kremetart/opt/fista.py`, replace the Task 2 `fista` body (everything after the docstring, from `y = xp.asarray(y)` to `return x, info`) with:

```python
    y = xp.asarray(y)
    weight = None if weight is None else xp.asarray(weight)
    real_dtype = y.real.dtype
    if x0 is None:
        x = xp.zeros(AH(y).shape, dtype=real_dtype)
    else:
        x = xp.asarray(x0, dtype=real_dtype).copy()
    w = xp.ones(x.shape, dtype=real_dtype)

    iterations: list[int] = []
    converged = False
    lipschitz = 1.0 if L0 is None else float(L0)
    for ell in range(max_reweight + 1):
        x_prev_round = x.copy()
        x, iters, converged, lipschitz = _fista_single(
            A, AH, y, w, x, lam=lam, weight=weight, positive=positive,
            L0=L0, eta=eta, max_iter=max_iter, tol=tol, xp=xp,
        )
        iterations.append(iters)
        if ell == max_reweight:
            break
        w = 1.0 / (xp.abs(x) + reweight_eps)  # Candès–Wakin–Boyd reweighting
        denom = max(float(xp.linalg.norm(x_prev_round)), 1e-12)
        if float(xp.linalg.norm(x - x_prev_round)) / denom < reweight_tol:
            break  # support / values have stabilised

    info = {
        "iterations": iterations,
        "reweights": len(iterations) - 1,
        "objective": _objective(A, y, weight, x, lam, xp),
        "lipschitz": lipschitz,
        "converged": converged,
    }
    return x, info
```

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `uv run pytest tests/test_fista.py -v`
Expected: PASS (8 passed) — the earlier `max_reweight=0` tests still pass (loop runs once), and the reweighting test now passes.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/opt/fista.py tests/test_fista.py
git commit -m "feat: add reweighted-L1 outer loop to FISTA

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Export and sky-model integration test

**Files:**
- Modify: `src/kremetart/opt/__init__.py`
- Test: `tests/test_fista.py`

**Interfaces:**
- Consumes: `fista` (Task 3); `enu_direction_cosines`, `model_visibilities` from `kremetart.utils.skymodel`.
- Produces: `from kremetart.opt import fista` import path; an integration test recovering injected fluxes through the real per-source operator built from `model_visibilities`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fista.py`:

```python
from kremetart.utils.skymodel import enu_direction_cosines, model_visibilities


def test_exported_from_opt_package():
    from kremetart.opt import fista as fista_pkg

    assert fista_pkg is fista


def test_recover_source_fluxes_through_model_visibilities():
    # Per-source forward operator: column j is the unit-flux model visibility of source j.
    az = np.radians([10.0, 120.0, 250.0])
    el = np.radians([70.0, 40.0, 55.0])
    s = enu_direction_cosines(az, el)
    bl_enu = np.array(
        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 0.0],
         [5.0, 1.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.0, 0.0]]
    )
    freqs = np.array([1.575e9])
    cols = np.stack(
        [model_visibilities(s[j : j + 1], bl_enu, freqs).ravel() for j in range(s.shape[0])],
        axis=1,
    )  # (nbl*nchan, nsrc)

    def A(x):
        return cols @ x

    def AH(r):
        return cols.conj().T @ r

    flux_true = np.array([1.0, 0.5, 2.0])
    y = cols @ flux_true
    x, info = fista(A, AH, y, lam=1e-3, positive=True, tol=1e-10, max_iter=5000, max_reweight=4)
    np.testing.assert_allclose(x, flux_true, atol=1e-2)
    assert np.all(x >= 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fista.py -k "exported or recover_source" -v`
Expected: FAIL — `test_exported_from_opt_package` fails with `ImportError: cannot import name 'fista' from 'kremetart.opt'` (not yet exported).

- [ ] **Step 3: Export `fista` from the package**

Replace `src/kremetart/opt/__init__.py` with:

```python
"""Optimisation solvers for kremetart (xp-injectable; CPU via numpy, GPU via cupy)."""

from kremetart.opt.cg import cg
from kremetart.opt.fista import fista

__all__ = ["cg", "fista"]
```

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `uv run pytest tests/test_fista.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/opt/__init__.py tests/test_fista.py
git commit -m "feat: export fista and add sky-model recovery test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §2 problem (real `x`, complex `A`, `∇f = Re{Aᴴ(W⊙r)}`) → Task 2 `_fista_single.grad_at`. ✓
- §3 interface & `info` keys → Task 2 `fista` signature + `info`; `iterations`/`reweights` finalised in Task 3. ✓
- §4.1 FISTA + backtracking → Task 2 `_fista_single`. ✓
- §4.2 prox (both modes) → Task 1 `_soft_threshold`. ✓
- §4.3 reweighting loop + early stop → Task 3. ✓
- §5 edge cases: zero data → Task 2 `test_zero_data_returns_zeros`; tiny `L0` → Task 2 `test_backtracking_recovers_from_tiny_l0`; `reweight_eps` guard → Task 3 code. ✓
- §6 conventions (placement, export, ruff/N806, no cab) → Global Constraints + Task 4. ✓
- §7 tests 1–7 → test 1 (Task 1), tests 2/4/5/7 (Task 2), test 3 (Task 3), test 6 (Task 4). ✓
- §8 out of scope (driver) → not implemented, by design. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step states expected output. ✓

**Type consistency:** `_soft_threshold(z, tau, *, positive, xp)`, `_objective(A, y, weight, x, lam, xp)`, `_fista_single(...) -> (x, iters, converged, lipschitz)`, and `fista(...) -> (x, info)` are referenced identically in every task that uses them. `info` keys (`iterations`, `reweights`, `objective`, `lipschitz`, `converged`) match between Task 2 and Task 3. ✓

---

## Execution Handoff
