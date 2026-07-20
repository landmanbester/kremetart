# Reweighted-L1 FISTA Solver — Design

**Date:** 2026-06-24
**Status:** Approved (design); implementation pending.
**Location:** `src/kremetart/opt/fista.py`

## 1. Purpose

Provide a small, generic FISTA solver with backtracking and an outer reweighted-L1
loop. It exists so we can recover **non-negative source fluxes** from calibrated
visibilities at the outset of self-calibration — those flux estimates are what let us
move from the unit-flux acquisition StefCAL (phase-only) to **amplitude** calibration.

The solver itself knows nothing about radio astronomy: it is operator-agnostic and
`xp`-injectable, exactly like `opt/cg.py`. A later, separate deconvolution/self-cal
driver wires the forward operator `A` to either `dft_forward` (full HEALPix grid) or
`model_visibilities` (per-source) and calls this solver. **That wiring is out of scope
here** — this spec covers only `opt/fista.py` and its tests.

## 2. Problem Statement

Minimise, over a **real** vector `x` (length `n`; fluxes / image pixels):

```
F(x) = f(x) + g(x)
f(x) = ½ ‖ W^½ (A x − y) ‖²₂      (smooth data term)
g(x) = λ Σᵢ wᵢ |xᵢ|              (non-smooth reweighted-L1 penalty)
```

- `A : ℝⁿ → ℂᵐ` is a linear forward operator (real flux → complex visibilities),
  supplied as a callable `A(x) -> A @ x`.
- `Aᴴ : ℂᵐ → ℂⁿ` is its Hermitian adjoint, supplied as a callable `AH(r) -> Aᴴ @ r`.
- `y ∈ ℂᵐ` is the (complex) data; `W ∈ ℝᵐ₊` is per-datum inverse-variance weight.
- `λ > 0` is the L1 strength; `wᵢ ≥ 0` are the reweighting weights (all 1 in round 0).

Because `x` is real, the gradient of the smooth term is the **real part** of the
complex adjoint applied to the weighted residual:

```
r(x)  = A x − y                         (∈ ℂᵐ)
f(x)  = ½ Σₘ Wₘ |rₘ|²                    (∈ ℝ)
∇f(x) = Re{ Aᴴ( W ⊙ r(x) ) }            (∈ ℝⁿ)
```

`W` is the inverse-variance weight applied **once** in the residual (matching how
`dirty_map` / `hessian_healpix` consume `weights`), not its square root.

`f` is convex with Lipschitz-continuous gradient (constant `L = ‖Aᴴ W A‖₂`); the
backtracking line search estimates `L` so the caller never has to.

## 3. Public Interface

```python
def fista(
    A: Callable,                 # x -> A @ x      (real x -> complex)
    AH: Callable,                # r -> Aᴴ @ r     (complex -> complex; real part taken internally)
    y,                           # (m,) complex data
    *,
    lam: float,                  # λ, L1 regularization strength (> 0; 0 allowed -> plain LS)
    weight=None,                 # (m,) real inverse-variance W; None -> all ones
    x0=None,                     # warm start; None -> zeros
    positive: bool = True,       # non-negativity in the prox
    L0: float | None = None,     # initial Lipschitz estimate; None -> 1.0
    eta: float = 2.0,            # backtracking growth factor (> 1)
    max_iter: int = 500,         # max inner FISTA iterations per reweight round
    tol: float = 1e-5,           # inner relative-change stopping tolerance
    max_reweight: int = 0,       # outer reweighting rounds (0 -> plain L1, no reweighting)
    reweight_eps: float = 1e-3,  # ε in wᵢ = 1/(|xᵢ| + ε)
    reweight_tol: float = 1e-3,  # outer stopping: relative change between rounds
    xp: ModuleType = np,         # array module (numpy / cupy)
) -> tuple[ndarray, dict]:
    ...
```

Returns `(x, info)`:

- `x`: `(n,)` real solution (`≥ 0` when `positive=True`).
- `info`: dict with
  - `iterations`: `list[int]` — inner iterations used, one entry per reweight round.
  - `reweights`: `int` — number of reweighting rounds actually performed.
  - `objective`: `float` — final `F(x)`.
  - `lipschitz`: `float` — final Lipschitz estimate.
  - `converged`: `bool` — whether the **last** inner solve hit `tol` before `max_iter`.

The `(result, info)` shape mirrors `stefcal_solve`; `cg` returns a bare array because it
has no comparable diagnostics worth surfacing.

## 4. Algorithm

### 4.1 Inner loop — FISTA with backtracking (Beck & Teboulle 2009)

State: iterate `x`, momentum point `v` (their `y`), momentum scalar `t`, Lipschitz `lipschitz`.

Initialise `x = x_prev = x0` (or zeros), `v = x0`, `t = 1`, `lipschitz = L0 or 1.0`.

Each iteration `k`:

1. Compute once at the momentum point `v`: `grad = ∇f(v)` and `f_v = f(v)`
   (reuse `A v` for both).
2. **Backtracking** — starting from the current `lipschitz` (never decreased between
   iterations), repeat:
   - `z = v − (1/lipschitz) · grad`
   - `x_new = prox(z; τ = (lam/lipschitz)·w)`  (see §4.2)
   - accept if `f(x_new) ≤ f_v + ⟨x_new − v, grad⟩ + (lipschitz/2)‖x_new − v‖²`
     (the `g` terms cancel from the full FISTA condition, leaving this smooth test);
   - else `lipschitz ← eta · lipschitz` and retry.
   Each trial costs one extra forward `A(x_new)` to evaluate `f(x_new)`.
3. `t_new = (1 + √(1 + 4 t²)) / 2`
4. `v = x_new + ((t − 1)/t_new) · (x_new − x)`   (momentum extrapolation)
5. `x_prev = x`, `x = x_new`, `t = t_new`.
6. **Stop** when `‖x − x_prev‖ / max(‖x‖, ε_machine) < tol`, or after `max_iter`.

`⟨·,·⟩` and all norms are the **real** Euclidean inner product (the iterates are real).

### 4.2 Proximal operator (weighted L1, optional non-negativity)

For threshold vector `τᵢ ≥ 0`:

```
positive=True :  proxᵢ(z) = max(zᵢ − τᵢ, 0)
positive=False:  proxᵢ(z) = sign(zᵢ) · max(|zᵢ| − τᵢ, 0)
```

### 4.3 Outer loop — reweighted L1 (Candès, Wakin & Boyd 2008)

```
w ← 1                                   # round 0 is plain (unweighted) L1
x ← x0
for ℓ in 0 … max_reweight:
    x_prev_round ← x
    x, last_info ← inner FISTA to convergence, weights = w, warm start = x   (§4.1)
    if ℓ == max_reweight:               # no reweight after the final solve
        break
    w ← 1 / (|x| + reweight_eps)
    if ‖x − x_prev_round‖ / max(‖x_prev_round‖, ε_machine) < reweight_tol:
        break                           # support/values have stabilised
return x, info
```

`max_reweight=0` runs exactly one inner solve with `w=1` (plain L1). `reweight_eps`
keeps the weight update finite and sets the effective sparsity scale.

## 5. Edge Cases

- `y` all zero **or** `weight` all zero ⇒ `∇f ≡ 0` ⇒ `prox` of zeros ⇒ returns `x0`
  (or zeros). No division by zero.
- `lam = 0` ⇒ prox is identity (or non-negative clamp) ⇒ FISTA reduces to projected
  gradient least squares; still valid.
- Tiny / underestimated `L0` ⇒ backtracking inflates `lipschitz` until the descent test
  passes; the iterate cannot diverge. (Tested explicitly.)
- `reweight_eps > 0` guarantees the reweight update never divides by zero.

## 6. Conventions & Constraints

- **Placement:** `src/kremetart/opt/fista.py`, alongside `cg.py`. Generic and
  `xp`-injectable; `numpy` imported at module top (`opt/` follows `utils/` import rules —
  no lazy imports). No `cli/`, `core/`, or cab: this is a solver utility, like `cg`.
- **Export:** add `fista` to `opt/__init__.py`'s imports and `__all__`.
- **Naming / ruff:** line length 120; `select = E/F/I/N/W`; `N802/N803/N812` ignored but
  **`N806` is not**. So `A`/`AH` *argument* names are allowed (N803), but every **local**
  must be lowercase — the Lipschitz local is `lipschitz`, never `L`; the prox result is
  `x_new`, not `X`. Use `lam` (not the reserved `lambda`).
- **Docstrings:** Google style (Args / Returns), concise, matching `cg.py`.
- **Python 3.10+** modern typing (`X | Y`, `list[int]`).

## 7. Test Plan (TDD, `tests/test_fista.py`)

All tests use `xp=np`, `tempfile` for any artefacts (none expected), and stay offline.

1. **Prox closed form** — `positive=True` and `positive=False` against hand-computed
   soft-threshold values on a small vector.
2. **`A = I` identity** — with `A = AH = identity`, `weight=None`, the minimiser is exactly
   `soft_threshold(y, lam)`; assert `fista` matches to `tol`. (`positive=False`, real `y`.)
3. **Sparse non-negative recovery** — build a known `k`-sparse `x_true ≥ 0`, a small dense
   complex operator `A` (e.g. random Gaussian or a small DFT matrix), `y = A x_true`; small
   `lam`. Assert recovered support matches and `‖x − x_true‖` is small; assert the
   `max_reweight>0` error is `≤` the plain-L1 (`max_reweight=0`) error.
4. **Backtracking guard** — deliberately tiny `L0` (e.g. `1e-6`, far below the true `L`)
   still converges to the same optimum as a well-chosen `L0`; `info["lipschitz"]` grew.
5. **Non-negativity** — `positive=True` ⇒ `all(x >= 0)`, even with data that would push
   some coefficients negative under signed L1.
6. **Integration with the sky model** — inject a few non-negative source fluxes, form
   `y = model_visibilities(...) @ fluxes` (real fluxes → complex vis), recover the fluxes
   via `fista` with `A`/`AH` built from `model_visibilities`. Confirms the solver works
   through the real operator that the future amplitude-cal driver will use.
7. **Zero-data short-circuit** — `y = 0` (or `weight = 0`) returns `x0` unchanged and
   reports `converged`.

## 8. Out of Scope

- The deconvolution / self-cal **driver** that builds `A` from `dft_forward` or
  `model_visibilities`, runs the solve per frame/interval, and feeds fluxes back into
  amplitude calibration. (Future work — a separate spec, mirroring how the imager wires
  up `cg`.)
- Gain-amplitude estimation itself; this solver only supplies the flux estimates that
  step will consume.
- GPU (`cupy`) CI coverage: the solver is `xp`-injectable and structurally GPU-ready, but
  tests run on `numpy` only (consistent with the rest of the suite).
