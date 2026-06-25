# Reweighted-L1 Deconvolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reweighted-L1 deconvolution stage that is a drop-in sibling of `TikhonovOperator` in the streaming HEALPix pipeline, so we can compare it against Tikhonov and produce clean sparse images for downstream self-calibration.

**Architecture:** The deconvolution reduces to one reused image-space Hessian matvec: `∇f(x)=Hx−b`, `f(x)=½⟨x,Hx⟩−⟨b,x⟩` (the data-fit constant cancels and the operator never needs the visibilities `y`). We extend `opt/fista.py` with a `fista_quadratic(hess, b, …)` entry sharing the existing prox/backtracking/reweighting core, add a GPU `L1ReweightOperator` mirroring `TikhonovOperator`, and select between them with a new `--regulariser` flag (gated on the existing `eta>0`).

**Tech Stack:** Python 3.10+, numpy/cupy (`xp`-injection), Holoscan operators, Typer + hip-cargo cabs, pytest, ruff.

## Global Constraints

- `opt/fista.py` follows `utils/` import rules: imports at module top, no lazy/function-body imports; generic and `xp`-injectable; no cab.
- ruff: line length 120; `select = E/F/I/N/W`; `N802/N803/N812` ignored, **`N806` enforced** — no uppercase locals (`lipschitz` not `L`; `hx`/`hess`/`b` are argument names, allowed under N803). Use `lam`, never `lambda`. No lambda-assignment (E731).
- The existing public `fista(A, AH, y, …)` signature and behaviour must NOT change — the 12 tests in `tests/test_fista.py` are the regression gate for Task 1.
- After every code change: `uv run ruff format . && uv run ruff check . --fix`.
- `operators/l1_reweight.py` is inherently GPU-only: `cupy`/`holoscan` imported at module top (as in `operators/tikhonov.py`). It is NOT imported by any CPU test.
- Conventional Commit messages; end commit bodies with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- `eta>0` stays the activation gate; `regulariser` defaults to `"tikhonov"` so existing invocations are unchanged. The CLI option type is **plain `str`** (round-trip-safe), validated in `compose()`.
- `fista_quadratic` defaults: `max_iter=200, tol=1e-5, max_reweight=2, reweight_eps=1e-3, reweight_tol=1e-3, eta=2.0` (backtracking growth). The solver's `eta` (backtracking) is distinct from the operator/pipeline `eta` (regulariser strength → `lam=eta·wsum`).

---

### Task 1: Refactor `fista` onto a shared smooth-term core

Behaviour-preserving refactor: pull the inner FISTA+backtracking loop and the outer reweighting loop into private helpers driven by two smooth-term callables (`value(x)->f` and `value_grad(x)->(f, grad)`), and rebuild `fista` on top. No new behaviour — the existing tests are the gate.

**Files:**
- Modify: `src/kremetart/opt/fista.py`
- Test (regression only, unchanged): `tests/test_fista.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (private, for Task 2): `_fista_single(value_grad, value, x_start, threshold_w, *, positive, L0, eta, max_iter, tol, xp) -> (x, iters, converged, lipschitz)` and `_reweighted_fista(value_grad, value, x_init, *, lam, positive, L0, eta, max_iter, tol, max_reweight, reweight_eps, reweight_tol, xp) -> (x, info)`. `value_grad(x) -> (float, ndarray)`, `value(x) -> float`.

- [ ] **Step 1: Replace the body of `src/kremetart/opt/fista.py` with the refactor**

Keep the module docstring's first paragraph; the imports are unchanged (`from __future__ import annotations`, `import math`, `from collections.abc import Callable`, `from types import ModuleType`, `import numpy as np`). Replace everything from `def _soft_threshold` to the end of the file with:

```python
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


def _fista_single(value_grad, value, x_start, threshold_w, *, positive, L0, eta, max_iter, tol, xp):
    """FISTA with backtracking for one fixed prox threshold and smooth term.

    Args:
        value_grad: callable ``x -> (f(x), grad(x))`` for the smooth data term (momentum point).
        value: callable ``x -> f(x)`` for the smooth data term (backtracking trials; no gradient).
        x_start: real warm-start iterate.
        threshold_w: per-element prox threshold ``lam * w`` (broadcastable to ``x``).

    Returns:
        ``(x, iters, converged, lipschitz)``.
    """
    lipschitz = 1.0 if L0 is None else float(L0)
    x = xp.asarray(x_start).copy()
    x_prev = x.copy()
    v = x.copy()
    t = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        f_v, g_v = value_grad(v)
        bt = 0
        while True:
            z = v - g_v / lipschitz
            x_new = _soft_threshold(z, threshold_w / lipschitz, positive=positive, xp=xp)
            diff = x_new - v
            rhs = f_v + float((diff * g_v).sum()) + 0.5 * lipschitz * float((diff * diff).sum())
            # +1e-12: numerical slack so float round-off near equality doesn't force a needless backtrack
            if value(x_new) <= rhs + 1e-12 or bt >= 100:
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


def _reweighted_fista(
    value_grad,
    value,
    x_init,
    *,
    lam: float,
    positive: bool,
    L0,
    eta: float,
    max_iter: int,
    tol: float,
    max_reweight: int,
    reweight_eps: float,
    reweight_tol: float,
    xp: ModuleType,
):
    """Outer Candès–Wakin–Boyd reweighting loop shared by ``fista`` and ``fista_quadratic``."""
    x = xp.asarray(x_init)
    w = xp.ones(x.shape, dtype=x.dtype)
    iterations: list[int] = []
    converged = False
    lipschitz = 1.0 if L0 is None else float(L0)
    for ell in range(max(max_reweight, 0) + 1):  # negative count -> a single plain-L1 solve
        x_prev_round = x.copy()
        x, iters, converged, lipschitz = _fista_single(
            value_grad, value, x, lam * w, positive=positive, L0=L0, eta=eta, max_iter=max_iter, tol=tol, xp=xp
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
        "objective": value(x) + lam * float(xp.abs(x).sum()),
        "lipschitz": lipschitz,
        "converged": converged,
    }
    return x, info


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
        x_init = xp.zeros(AH(y).shape, dtype=real_dtype)
    else:
        x_init = xp.asarray(x0, dtype=real_dtype).copy()

    def fval(r):
        sq = xp.abs(r) ** 2
        if weight is not None:
            sq = weight * sq
        return 0.5 * float(sq.sum())

    def value(x):
        return fval(A(x) - y)

    def value_grad(x):
        r = A(x) - y
        wr = r if weight is None else weight * r
        return fval(r), xp.real(AH(wr))  # x is real -> take the real part of the complex adjoint

    return _reweighted_fista(
        value_grad,
        value,
        x_init,
        lam=lam,
        positive=positive,
        L0=L0,
        eta=eta,
        max_iter=max_iter,
        tol=tol,
        max_reweight=max_reweight,
        reweight_eps=reweight_eps,
        reweight_tol=reweight_tol,
        xp=xp,
    )
```

- [ ] **Step 2: Format and lint**

Run: `uv run ruff format src/kremetart/opt/fista.py && uv run ruff check src/kremetart/opt/fista.py --fix`
Expected: no errors. (`_objective` is gone — confirm no leftover reference to it.)

- [ ] **Step 3: Run the existing FISTA tests (the regression gate)**

Run: `uv run pytest tests/test_fista.py -q`
Expected: 12 passed. The refactor changed structure only; `x`, `info["converged"]`, `info["reweights"]`, `info["lipschitz"]`, and `info["iterations"]` are computed exactly as before.

- [ ] **Step 4: Commit**

```bash
git add src/kremetart/opt/fista.py
git commit -m "refactor: drive fista on shared value/value_grad smooth-term core"
```

---

### Task 2: Add `fista_quadratic`

The image-space entry: minimise `½⟨x,Hx⟩ − ⟨b,x⟩ + λ Σ wᵢ|xᵢ|` over real `x` (≥0 when `positive`). `hess` is the Hessian matvec `x -> H x`; `b` the un-normalised dirty. The real part of `hess(x)` is taken internally (H is real-symmetric over real `x`), so a complex-returning `H = Aᴴ W A` matvec works too.

**Files:**
- Modify: `src/kremetart/opt/fista.py` (append `fista_quadratic`)
- Modify: `src/kremetart/opt/__init__.py`
- Test: `tests/test_fista.py` (append cases)

**Interfaces:**
- Consumes: `_reweighted_fista` (Task 1); `kremetart.utils.healpix_dft.hessian_healpix(baselines, pix_vec, freqs, weights, *, beam=None, xp=np) -> (matvec, diagonal)`; `kremetart.utils.healpix_dft.make_pixel_grid`, `dft_forward`; `kremetart.utils.beam.airy_power_beam`.
- Produces (for Task 3): `fista_quadratic(hess, b, *, lam, x0=None, positive=True, L0=None, eta=2.0, max_iter=200, tol=1e-5, max_reweight=2, reweight_eps=1e-3, reweight_tol=1e-3, xp=np) -> (x, info)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fista.py`:

```python
def test_fista_quadratic_matches_least_squares():
    # Quadratic form H = Aᴴ W A, b = Re{Aᴴ W y} must reach the same optimum as the A/AH path.
    rng = np.random.default_rng(21)
    n, m = 12, 40
    mat = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    weight = rng.uniform(0.5, 1.5, size=m)
    x_true = np.zeros(n)
    x_true[[1, 4, 9]] = [1.0, 2.0, 0.5]
    y = mat @ x_true

    def A(x):
        return mat @ x

    def AH(r):
        return mat.conj().T @ r

    def H(x):
        return AH(weight * A(x))  # complex; fista_quadratic takes the real part

    b = np.real(AH(weight * y))
    x_ls, _ = fista(A, AH, y, weight=weight, lam=0.2, positive=False, tol=1e-11, max_iter=8000)
    x_q, _ = fista_quadratic(H, b, lam=0.2, positive=False, tol=1e-11, max_iter=8000, max_reweight=0)
    np.testing.assert_allclose(x_q, x_ls, atol=1e-4)
    assert not np.iscomplexobj(x_q)


def test_fista_quadratic_identity_recovers_soft_threshold():
    # H = I, b = y, signed L1: argmin ½xᵀx − bᵀx + λ|x|₁ == soft_threshold(y, λ).
    rng = np.random.default_rng(22)
    y = rng.standard_normal(40)
    x, info = fista_quadratic(lambda x: x, y, lam=0.3, positive=False, tol=1e-11, max_iter=4000)
    np.testing.assert_allclose(x, np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0), atol=1e-5)
    assert info["converged"]


def test_fista_quadratic_backtracking_from_tiny_l0():
    rng = np.random.default_rng(23)
    y = rng.standard_normal(40)
    x, info = fista_quadratic(lambda x: x, y, lam=0.3, positive=False, L0=1e-6, tol=1e-11, max_iter=4000)
    np.testing.assert_allclose(x, np.sign(y) * np.maximum(np.abs(y) - 0.3, 0.0), atol=1e-5)
    assert info["lipschitz"] > 0.5  # grew from 1e-6 toward ‖H‖₂ = 1


def test_fista_quadratic_positive_constraint():
    rng = np.random.default_rng(24)
    y = rng.standard_normal(40)  # has negatives
    x, _ = fista_quadratic(lambda x: x, y, lam=0.1, positive=True, tol=1e-11, max_iter=4000)
    assert np.all(x >= 0.0)
    np.testing.assert_allclose(x, np.maximum(y - 0.1, 0.0), atol=1e-5)


def test_fista_quadratic_reweighting_debiases():
    rng = np.random.default_rng(25)
    n, m = 20, 60
    mat = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[[3, 8, 14]] = [1.0, 2.0, 1.5]
    y = mat @ x_true

    def H(x):
        return mat.conj().T @ (mat @ x)

    b = np.real(mat.conj().T @ y)
    x_l1, info_l1 = fista_quadratic(H, b, lam=0.2, positive=True, tol=1e-10, max_iter=6000, max_reweight=0)
    x_rw, info_rw = fista_quadratic(H, b, lam=0.2, positive=True, tol=1e-10, max_iter=6000, max_reweight=8)
    assert np.linalg.norm(x_rw - x_true) <= np.linalg.norm(x_l1 - x_true) + 1e-9
    assert np.linalg.norm(x_rw - x_true) < 1e-2
    assert info_l1["reweights"] == 0 and info_rw["reweights"] >= 1
    np.testing.assert_array_equal(np.where(x_rw > 1e-3)[0], [3, 8, 14])


def test_fista_quadratic_zero_rhs_short_circuit():
    x0 = np.array([0.4, 0.0, 1.1])
    x, info = fista_quadratic(lambda x: x, np.zeros(3), lam=0.5, positive=True, x0=x0)
    np.testing.assert_allclose(x, x0)  # b == 0 -> the warm start is unchanged
    assert info["converged"] and info["reweights"] == 0


def test_fista_quadratic_recovers_sparse_sky_through_hessian():
    # End-to-end on the real image-space Hessian: plant point sources, build H/b, recover with FISTA.
    from kremetart.utils.beam import airy_power_beam
    from kremetart.utils.healpix_dft import dft_adjoint, dft_forward, hessian_healpix, make_pixel_grid

    nside = 8
    pix = make_pixel_grid(nside, nest=True, xp=np)  # (npix, 3)
    npix = pix.shape[0]
    rng = np.random.default_rng(26)
    bl = rng.standard_normal((40, 3)) * 2.0  # metres
    freqs = np.array([1.575e9])
    boresight = np.array([0.0, 0.0, 1.0])
    beam = airy_power_beam(pix, boresight, freqs, xp=np)  # (1, npix)
    weights = np.ones((bl.shape[0], 1))

    sky = np.zeros(npix)
    up = np.where(pix @ boresight > 0.6)[0]  # well inside the beam
    src = up[rng.choice(up.size, size=3, replace=False)]
    sky[src] = np.array([1.0, 2.0, 1.5])

    vis = dft_forward(sky, bl, pix, freqs, beam=beam, xp=np)  # (nbl, nchan) complex
    hmv, hdiag = hessian_healpix(bl, pix, freqs, weights, beam=beam, xp=np)
    b = np.real(dft_adjoint(weights * vis, bl, pix, freqs, beam=beam, xp=np))  # un-normalised dirty = Re{B Mᴴ W y}

    x, info = fista_quadratic(
        hmv, b, lam=1e-3 * float(weights.sum()), positive=True, L0=float(hdiag.max()),
        tol=1e-9, max_iter=4000, max_reweight=3,
    )
    assert np.all(x >= 0.0)
    # The three planted pixels are the brightest recovered, in the right flux order.
    top3 = np.argsort(x)[-3:]
    assert set(top3.tolist()) == set(src.tolist())


def test_fista_quadratic_exported_from_opt_package():
    from kremetart.opt import fista_quadratic as fq_pkg

    assert fq_pkg is fista_quadratic
```

Update the import at the top of `tests/test_fista.py`:

```python
from kremetart.opt.fista import _soft_threshold, fista, fista_quadratic
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_fista.py -q -k quadratic`
Expected: FAIL/ERROR — `cannot import name 'fista_quadratic'`.

- [ ] **Step 3: Append `fista_quadratic` to `src/kremetart/opt/fista.py`**

```python
def fista_quadratic(
    hess: Callable,
    b,
    *,
    lam: float,
    x0=None,
    positive: bool = True,
    L0: float | None = None,
    eta: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    max_reweight: int = 2,
    reweight_eps: float = 1e-3,
    reweight_tol: float = 1e-3,
    xp: ModuleType = np,
):
    """Minimise ``0.5⟨x,Hx⟩ − ⟨b,x⟩ + λ Σ wᵢ|xᵢ|`` over real ``x`` via reweighted-L1 FISTA.

    For the image-space deconvolution: ``hess`` is the Hessian matvec ``x -> H x`` with
    ``H = B Mᴴ W M B`` (:func:`kremetart.utils.healpix_dft.hessian_healpix`) and ``b`` the
    un-normalised dirty image (the imager's normalised dirty times ``Σw``). The full least-squares
    constant ``½ yᴴ W y`` is dropped — it cancels in the backtracking test and the caller has only
    ``b``, not ``y`` — so ``info["objective"]`` is the data fit up to that additive constant plus the
    L1 penalty. The real part of ``hess(x)`` is taken internally (``H`` is real-symmetric over real
    ``x``), so a complex-returning ``H = Aᴴ W A`` matvec is accepted.

    Args:
        hess: callable ``x -> H x`` (SPD over the reals; real or complex output, real part used).
        b: ``(n,)`` real right-hand side (the un-normalised dirty image).
        lam: L1 strength ``λ`` (``= eta·Σw`` from the operator).
        x0: optional real warm start; ``None`` -> zeros.
        positive: enforce ``x >= 0`` in the prox.
        L0: initial Lipschitz estimate; pass ``diag(H).max()`` for a tight, free seed; ``None`` -> 1.0.
        eta: backtracking growth factor (> 1) — distinct from the regulariser strength.
        max_iter, tol, max_reweight, reweight_eps, reweight_tol, xp: as in :func:`fista`.

    Returns:
        ``(x, info)`` as in :func:`fista`.
    """
    b = xp.asarray(b)
    real_dtype = b.real.dtype
    if x0 is None:
        x_init = xp.zeros(b.shape, dtype=real_dtype)
    else:
        x_init = xp.asarray(x0, dtype=real_dtype).copy()

    def value(x):
        hx = xp.real(hess(x))
        return 0.5 * float((x * hx).sum()) - float((b * x).sum())

    def value_grad(x):
        hx = xp.real(hess(x))
        f = 0.5 * float((x * hx).sum()) - float((b * x).sum())
        return f, hx - b

    return _reweighted_fista(
        value_grad,
        value,
        x_init,
        lam=lam,
        positive=positive,
        L0=L0,
        eta=eta,
        max_iter=max_iter,
        tol=tol,
        max_reweight=max_reweight,
        reweight_eps=reweight_eps,
        reweight_tol=reweight_tol,
        xp=xp,
    )
```

- [ ] **Step 4: Export from `src/kremetart/opt/__init__.py`**

Replace the file contents with:

```python
from kremetart.opt.cg import cg
from kremetart.opt.fista import fista, fista_quadratic

__all__ = ["cg", "fista", "fista_quadratic"]
```

- [ ] **Step 5: Format, lint, and run the tests**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pytest tests/test_fista.py -q`
Expected: all pass (the original 12 plus the 8 new quadratic cases = 20 passed).

- [ ] **Step 6: Commit**

```bash
git add src/kremetart/opt/fista.py src/kremetart/opt/__init__.py tests/test_fista.py
git commit -m "feat: add fista_quadratic image-space reweighted-L1 solver"
```

---

### Task 3: `L1ReweightOperator` (GPU operator)

A drop-in sibling of `TikhonovOperator`: identical ports and per-frame contract, swapping CG for `fista_quadratic`. GPU-only (`cupy`/`holoscan` at module top), so it has NO direct CPU test — its numerics are already covered by Task 2 (`fista_quadratic`) and `tests/test_healpix_dft.py` (`hessian_healpix`), exactly as `TikhonovOperator`'s are covered by `test_cg.py` + `test_healpix_dft.py`.

**Files:**
- Create: `src/kremetart/operators/l1_reweight.py`

**Interfaces:**
- Consumes: `kremetart.opt.fista.fista_quadratic` (Task 2); `kremetart.utils.healpix_dft.hessian_healpix`, `make_pixel_grid`; `kremetart.utils.beam.airy_power_beam`, `GROUND_PLANE_DIAMETER`.
- Produces (for Task 4): class `L1ReweightOperator(fragment, nside, freqs, eta, *args, nest=True, apply_beam=True, ground_plane_diameter=GROUND_PLANE_DIAMETER, max_iter=200, tol=1e-5, max_reweight=2, reweight_eps=1e-3, positive=True, use_warm_start=True, **kwargs)`. Inputs `cube, WEIGHT, B_ROT, BORESIGHT, time_out`; outputs `cube, dirty, time_out` (byte-identical port set to `TikhonovOperator`).

- [ ] **Step 1: Create `src/kremetart/operators/l1_reweight.py`**

```python
"""Holoscan operator: per-frame reweighted-L1 deconvolution via FISTA (GPU-resident, xp=cupy).

A drop-in sibling of :class:`kremetart.operators.tikhonov.TikhonovOperator`: identical ports and
per-frame contract, but it solves ``min ½⟨x,Hx⟩ − ⟨b,x⟩ + λ Σ wᵢ|xᵢ|`` (non-negative, sparse) with
:func:`kremetart.opt.fista.fista_quadratic` instead of the Tikhonov CG normal-equation solve. ``H``
is the image-space Hessian (:func:`kremetart.utils.healpix_dft.hessian_healpix`), ``b`` the
un-normalised dirty image (the imager's normalised dirty times ``Σw``), and ``λ = eta·Σw`` makes
``eta`` a frame-invariant fraction of the central PSF value ``Σw`` (matching the Tikhonov knob). The
Lipschitz step is seeded from the closed-form ``diag(H).max()`` that ``hessian_healpix`` returns, so
backtracking almost never fires. Selected via ``smoovie``'s ``--regulariser l1``. See
docs/superpowers/specs/2026-06-25-reweighted-l1-deconvolution-design.md.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.opt.fista import fista_quadratic
from kremetart.utils.beam import GROUND_PLANE_DIAMETER, airy_power_beam
from kremetart.utils.healpix_dft import hessian_healpix, make_pixel_grid


class L1ReweightOperator(Operator):
    """Per-frame reweighted-L1 deconvolution (sparse non-negative image) via FISTA.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        freqs: ``(nchan,)`` frequencies in Hz.
        eta: regularisation strength as a fraction of ``Σw`` (``λ = eta·Σw``); must be > 0.
        nest: NESTED HEALPix ordering (default True).
        apply_beam: build the Airy beam into ``H`` (must match the imager's setting).
        ground_plane_diameter: Airy aperture diameter in metres.
        max_iter: maximum inner FISTA iterations per reweight round.
        tol: inner relative-change tolerance.
        max_reweight: outer Candès–Wakin–Boyd reweighting rounds.
        reweight_eps: ``ε`` in ``wᵢ = 1/(|xᵢ| + ε)``.
        positive: enforce ``x >= 0`` (the sky is non-negative).
        use_warm_start: seed each frame's FISTA with the previous frame's solution.
    """

    def __init__(
        self,
        fragment,
        nside,
        freqs,
        eta,
        *args,
        nest=True,
        apply_beam=True,
        ground_plane_diameter=GROUND_PLANE_DIAMETER,
        max_iter=200,
        tol=1e-5,
        max_reweight=2,
        reweight_eps=1e-3,
        positive=True,
        use_warm_start=True,
        **kwargs,
    ):
        self.nside = nside
        self.freqs = cp.asarray(freqs)
        self.eta = float(eta)
        self.nest = nest
        self.apply_beam = apply_beam
        self.ground_plane_diameter = ground_plane_diameter
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_reweight = int(max_reweight)
        self.reweight_eps = float(reweight_eps)
        self.positive = positive
        self.use_warm_start = use_warm_start
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        self.pix_vec = make_pixel_grid(self.nside, nest=self.nest, xp=cp)
        self.x_prev = None  # device-resident warm-start state

    def setup(self, spec: OperatorSpec):
        spec.input("cube")  # imager dirty map = RHS (normalised)
        spec.input("WEIGHT")
        spec.input("B_ROT")
        spec.input("BORESIGHT")
        spec.input("time_out")
        spec.output("cube")  # regularised image -> IWP
        spec.output("dirty")  # raw dirty passthrough -> writer
        spec.output("time_out")

    def compute(self, op_input, op_output, context):
        dirty = cp.asarray(op_input.receive("cube"))  # (1, npix)
        weights = cp.asarray(op_input.receive("WEIGHT"))  # (1, nbl, nchan)
        b_rot = cp.asarray(op_input.receive("B_ROT"))  # (1, nbl, 3)
        boresight = cp.asarray(op_input.receive("BORESIGHT"))  # (1, 3)
        time_out = cp.asarray(op_input.receive("time_out"))  # (1,)

        w = weights[0]  # (nbl, nchan)
        wsum = w.sum()
        # Fully-flagged frame: the imager's dirty is the all-zero no-data map and λ = eta·Σw = 0, so
        # there is nothing to solve. Pass the zero map straight through (the IWP reads it as a no-data
        # frame and coasts); leave the warm-start untouched so the next live frame still seeds well.
        if float(wsum) == 0.0:
            zeros = cp.zeros(self.pix_vec.shape[0], dtype=cp.float64)
            op_output.emit(hs.as_tensor(zeros[None, :]), "cube")
            op_output.emit(hs.as_tensor(dirty), "dirty")
            op_output.emit(hs.as_tensor(time_out), "time_out")
            return

        beam = None
        if self.apply_beam:
            beam = airy_power_beam(
                self.pix_vec, boresight[0], self.freqs, diameter=self.ground_plane_diameter, xp=cp
            )  # (nchan, npix)

        rows = b_rot[0]  # (nbl, 3)
        hmv, hdiag = hessian_healpix(rows, self.pix_vec, self.freqs, w, beam=beam, xp=cp)
        b = dirty[0] * wsum  # un-normalise the imager's normalised dirty to the Hessian RHS
        lam = self.eta * wsum

        x0 = None
        if self.use_warm_start and self.x_prev is not None and bool(cp.all(cp.isfinite(self.x_prev))):
            x0 = self.x_prev

        x, _info = fista_quadratic(
            hmv,
            b,
            lam=float(lam),
            x0=x0,
            positive=self.positive,
            L0=float(hdiag.max()),
            max_iter=self.max_iter,
            tol=self.tol,
            max_reweight=self.max_reweight,
            reweight_eps=self.reweight_eps,
            xp=cp,
        )
        self.x_prev = x

        op_output.emit(hs.as_tensor(x[None, :]), "cube")
        op_output.emit(hs.as_tensor(dirty), "dirty")
        op_output.emit(hs.as_tensor(time_out), "time_out")
```

- [ ] **Step 2: Lint and syntax-check (no GPU needed)**

The module imports cupy/holoscan at top, so it cannot be imported on a CPU runner. Verify it lints and parses:

Run: `uv run ruff format src/kremetart/operators/l1_reweight.py && uv run ruff check src/kremetart/operators/l1_reweight.py && python -c "import ast; ast.parse(open('src/kremetart/operators/l1_reweight.py').read())"`
Expected: no ruff errors; `ast.parse` prints nothing (success).

- [ ] **Step 3: Commit**

```bash
git add src/kremetart/operators/l1_reweight.py
git commit -m "feat: add L1ReweightOperator (FISTA deconvolution sibling of Tikhonov)"
```

---

### Task 4: Wire `--regulariser` into the pipeline

`eta>0` stays the activation gate; `regulariser` (`"tikhonov"` default, or `"l1"`) picks the operator. Plain `str` type (round-trip-safe), validated in `compose()`. Both `cli/smoovie.py` and `core/smoovie.py` gain the param (the `test_structure.py` mirror requires it on both); the pre-commit hook regenerates `cabs/smoovie.yml`.

**Files:**
- Modify: `src/kremetart/core/smoovie.py`
- Modify: `src/kremetart/cli/smoovie.py`
- Regenerated: `src/kremetart/cabs/smoovie.yml` (by the pre-commit hook)

**Interfaces:**
- Consumes: `L1ReweightOperator` (Task 3); existing `TikhonovOperator`.
- Produces: `smoovie(..., regulariser="tikhonov", ...)` on both cli and core; `SmooviePipeline(..., regulariser=...)`; `image_via_app(..., regulariser=...)`.

- [ ] **Step 1: `core/smoovie.py` — import the operator**

Add next to the existing TikhonovOperator import (`src/kremetart/core/smoovie.py:28`):

```python
from kremetart.operators.l1_reweight import L1ReweightOperator
from kremetart.operators.tikhonov import TikhonovOperator
```

- [ ] **Step 2: `core/smoovie.py` — `SmooviePipeline.__init__` gains `regulariser`**

Add the keyword (after `eta: float | None = None,` in `__init__`, ~line 57) and store it (after `self.eta = eta`, ~line 69):

```python
        eta: float | None = None,
        regulariser: str = "tikhonov",
        holder: LatestFrameHolder | None = None,
```
```python
        self.eta = eta
        self.regulariser = regulariser
        self.holder = holder
```

- [ ] **Step 3: `core/smoovie.py` — `compose()` selects the operator**

Replace the Tikhonov-instantiation block (currently `tikhonov = TikhonovOperator(self, self.nside, self.freqs, self.eta, name="tikhonov", nest=..., apply_beam=..., ground_plane_diameter=...)` and the three `add_flow` lines that reference `tikhonov`, ~lines 126-145) with a regulariser-keyed `deconv` operator. Keep the surrounding `if regularise:` / `else:` structure and the variable name used in the flows:

```python
        if regularise:
            if self.regulariser == "tikhonov":
                deconv = TikhonovOperator(
                    self,
                    self.nside,
                    self.freqs,
                    self.eta,
                    name="tikhonov",
                    nest=self.nest,
                    apply_beam=self.apply_beam,
                    ground_plane_diameter=self.ground_plane_diameter,
                )
            elif self.regulariser == "l1":
                deconv = L1ReweightOperator(
                    self,
                    self.nside,
                    self.freqs,
                    self.eta,
                    name="l1reweight",
                    nest=self.nest,
                    apply_beam=self.apply_beam,
                    ground_plane_diameter=self.ground_plane_diameter,
                )
            else:
                raise ValueError(f"unknown regulariser {self.regulariser!r}; expected 'tikhonov' or 'l1'")
            # Imager dirty is the deconvolution RHS; the reader fans the data that builds the Hessian.
            self.add_flow(imager, deconv, {("cube", "cube"), ("time_out", "time_out")})
            self.add_flow(
                reader,
                deconv,
                {("WEIGHT", "WEIGHT"), ("B_ROT", "B_ROT"), ("BORESIGHT", "BORESIGHT")},
            )
            self.add_flow(deconv, iwp, {("cube", "cube"), ("time_out", "time_out")})
            self.add_flow(deconv, writer, {("dirty", "dirty_raw")})
        else:
            self.add_flow(imager, iwp, {("cube", "cube"), ("time_out", "time_out")})
```

- [ ] **Step 4: `core/smoovie.py` — thread `regulariser` through `image_via_app` and `smoovie`**

In `image_via_app`, add `regulariser: str = "tikhonov",` after `eta: float | None = None,` in the signature, and pass `regulariser=regulariser,` to the `SmooviePipeline(...)` call (next to `eta=eta,`).

In `smoovie`, add `regulariser: str = "tikhonov",` after `eta: float | None = None,` in the signature, and pass `regulariser=regulariser,` to the `image_via_app(...)` call (next to `eta=eta,`). Add a one-line note to the `smoovie` docstring: ``regulariser`` chooses the deconvolution when ``eta>0`` — ``"tikhonov"`` (CG, default) or ``"l1"`` (reweighted-L1 FISTA).

- [ ] **Step 5: `cli/smoovie.py` — add the `--regulariser` option**

In the signature, after the `eta` option block (`src/kremetart/cli/smoovie.py:101-106`), insert:

```python
    regulariser: Annotated[
        str,
        typer.Option(
            help="Deconvolution regulariser when eta>0: 'tikhonov' (CG, default) or 'l1' (reweighted-L1 FISTA).",
        ),
    ] = "tikhonov",
```

Then add `regulariser=regulariser,` immediately after every `eta=eta,` line in all three parameter dicts (the `preflight_remote_must_exist` dict, the `smoovie_core(...)` call, and the `run_in_container` dict).

- [ ] **Step 6: Format, regenerate the cab, and run the CPU guards**

Run: `uv run ruff format . && uv run ruff check . --fix`
Then regenerate the cab (mirrors the pre-commit hook), activating the venv so the `image:` field resolves:

Run: `uv run hip-cargo generate-cabs --module src/kremetart/cli/smoovie.py --output-dir src/kremetart/cabs`
Then: `uv run pytest tests/test_roundtrip.py tests/test_structure.py -q`
Expected: PASS. If `test_roundtrip_smoovie` fails on option ordering, regenerate the wrapper and adopt it verbatim:
`uv run python -c "from pathlib import Path; from hip_cargo.core.generate_function import generate_function; generate_function(Path('src/kremetart/cabs/smoovie.yml'), output_file=Path('src/kremetart/cli/smoovie.py'), config_file=Path('pyproject.toml'))"`, then `uv run ruff format src/kremetart/cli/smoovie.py` and re-run the test.

- [ ] **Step 7: Commit (re-stage if pre-commit regenerates the cab)**

```bash
git add src/kremetart/core/smoovie.py src/kremetart/cli/smoovie.py src/kremetart/cabs/smoovie.yml
git commit -m "feat: select tikhonov or l1 deconvolution via --regulariser"
```

If the pre-commit hook reports it modified `cabs/smoovie.yml`, run `git add -u && git commit` again to include the regenerated cab.

---

### Task 5: `scripts/compare_regularisers.py` — Tikhonov vs L1

A host/CPU script (sibling of `scripts/validate_tart_gains.py`) that images one TART frame three ways — raw dirty, Tikhonov (`cg` + `hessian_healpix`), reweighted-L1 (`fista_quadratic` + `hessian_healpix`) — at matched `eta`, and reports point-source concentration over above-horizon pixels. No GPU/Holoscan.

**Files:**
- Create: `scripts/compare_regularisers.py`

**Interfaces:**
- Consumes: `kremetart.opt.cg.cg`, `kremetart.opt.fista.fista_quadratic`; `kremetart.utils.healpix_dft` (`make_pixel_grid`, `hessian_healpix`, `image_frame`, `zenith_icrs_vectors`); `kremetart.utils.beam.airy_power_beam`; `kremetart.utils.partition_datatree`; `kremetart.utils.read_tart_hdf.read_hdf_as_msv4`.

- [ ] **Step 1: Create `scripts/compare_regularisers.py`**

```python
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
```

- [ ] **Step 2: Lint and smoke-test the CLI (no data needed)**

Run: `uv run ruff format scripts/compare_regularisers.py && uv run ruff check scripts/compare_regularisers.py && uv run python scripts/compare_regularisers.py --help`
Expected: no ruff errors; `--help` prints the usage and exits 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_regularisers.py
git commit -m "feat: add Tikhonov-vs-L1 deconvolution comparison script"
```

---

## Notes for the implementer

- Tasks 1–2 and 5 are CPU-testable end to end. Task 3 (GPU operator) and the `smoovie` host-wiring
  in Task 4 only run on a GPU box; on CPU CI they are guarded by lint/parse (Task 3) and by
  `test_structure.py` + `test_roundtrip.py` (Task 4).
- Do not hand-edit `src/kremetart/cabs/smoovie.yml` — it is generated. Task 4 Step 6 regenerates it.
- Leave any unrelated working-tree changes untouched; commit only the files each task names.
