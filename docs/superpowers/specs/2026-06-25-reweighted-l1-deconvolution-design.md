# Reweighted-L1 Deconvolution — Design

**Date:** 2026-06-25
**Status:** Draft (design); implementation pending.
**Locations:**
- `src/kremetart/opt/fista.py` — new `fista_quadratic` solver entry (additive).
- `src/kremetart/operators/l1_reweight.py` — new `L1ReweightOperator` (sibling of `TikhonovOperator`).
- `src/kremetart/core/smoovie.py` + `src/kremetart/cli/smoovie.py` — `--regulariser` selector.
- `tests/test_fista.py` — new quadratic-path cases.
- `scripts/compare_regularisers.py` — Tikhonov-vs-L1 comparison.

**Builds on:**
- `docs/superpowers/specs/2026-06-24-reweighted-l1-fista-design.md` — the FISTA solver this extends.
- `docs/superpowers/specs/2026-06-23-tikhonov-cg-regularisation-design.md` — the Tikhonov operator this mirrors.

## 1. Purpose

Add a **reweighted-L1 deconvolution stage** that is a drop-in sibling of `TikhonovOperator` in the
streaming HEALPix pipeline. Tikhonov (`(H + λI)x = b`, solved by CG) returns a *regularised dirty
image* — it suppresses high-spatial-frequency noise but leaves the dirty-beam structure. Reweighted
L1 actually **deconvolves** toward a sparse, non-negative set of point sources, which is what the
TART sky is (a handful of bright satellites plus sidelobes over a mostly-empty sphere) and what a
downstream self-calibration step wants as its model. The Candès–Wakin–Boyd reweighting debiases the
flux amplitudes that plain L1 shrinks low — exactly the bias that would otherwise corrupt amplitude
gain solutions.

The deliverable is the operator plus the means to **compare it against Tikhonov** on real TART
frames, so we can judge whether L1 gives the cleaner image we expect before wiring it into self-cal.

## 2. Why this reduces to the Hessian

The smooth data term is `f(x) = ½‖W^½(M B x − y)‖²` over a **real, non-negative** sky `x`, where
`M` is the geometric DFT, `B` the per-channel Airy beam, `W` the inverse-variance weights, and `y`
the visibilities. Expanding:

```
f(x) = ½⟨x, H x⟩ − ⟨b, x⟩ + c
H    = B Mᴴ W M B          (image-space Hessian; kremetart.utils.healpix_dft.hessian_healpix)
b    = Re{ B Mᴴ W y }      (the un-normalised dirty image = imager dirty × Σw)
c    = ½ yᴴ W y            (constant in x)
```

FISTA's smooth term needs only `∇f(x)` and `f(x)`:

```
∇f(x) = H x − b
q(x)  = ½⟨x, H x⟩ − ⟨b, x⟩ = f(x) − c
```

Two consequences that make this efficient and that fix the interface:

1. **The constant `c` cancels.** FISTA's backtracking sufficient-decrease test compares the data
   term at two points plus linear/quadratic correction terms; `c` appears on both sides and drops
   out. The operator only ever has `b` (the imager's dirty), never `y`, so it *cannot* form `c` —
   and does not need to. The reported objective is `q(x) + λ‖x‖₁`, i.e. the data fit up to the
   irrelevant additive constant `c`.
2. **Everything is the Hessian matvec.** Both `∇f` and `q` are pure `H·x` contractions. `H` is built
   **once per frame** by `hessian_healpix` and the returned matvec is reused across *every* FISTA
   iteration and *every* reweighting round — the reweighting loop only changes the L1 weights `w`,
   never the geometry. This is the whole optimisation: the deconvolution boils down to one reused
   Hessian matvec.

## 3. Frame choice: ICRS, per-frame Hessian

The pipeline images in the equatorial **ICRS** frame, where `b_rot(t)` and the beam rotate every
frame, so `H` genuinely differs per frame and is rebuilt each frame (within-frame matvec reuse only).
We deliberately keep this rather than switching to a fixed local (ENU/ITRS) frame where `H` would be
frame-invariant and built once, because:

- **The long-term goal is multi-TART co-imaging into one shared ICRS frame.** Several arrays at
  different sites contribute to the *same* ICRS pixels; a per-array local-frame Hessian cannot
  compose into that, whereas a per-frame ICRS Hessian is exactly the primitive that sums array
  contributions in the common frame. ICRS per-frame is forward-compatible; local-frame is a dead end.
- At TART scale the per-frame rebuild is GPU-cheap (§7), so there is no demonstrator cost to pay.

Cross-frame Hessian reuse is therefore explicitly **not** pursued (see §10).

## 4. Solver: `fista_quadratic` (extends `opt/fista.py`)

Add a second public entry to `opt/fista.py`, alongside the existing `fista(A, AH, y, …)`. Both become
thin wrappers over a shared private reweighting/backtracking core, so the existing least-squares entry
and its tests are unchanged.

```python
def fista_quadratic(
    hess: Callable,              # x -> H @ x   (real x -> real H x; SPD over the reals)
    b,                           # (n,) real RHS = the un-normalised dirty image
    *,
    lam: float,                  # λ, the L1 strength (> 0)
    x0=None,                     # warm start; None -> zeros
    positive: bool = True,       # non-negativity in the prox
    L0: float | None = None,     # initial Lipschitz estimate; None -> 1.0
    eta: float = 2.0,            # backtracking growth factor (> 1)
    max_iter: int = 200,         # max inner FISTA iterations per reweight round
    tol: float = 1e-5,           # inner relative-change stopping tolerance
    max_reweight: int = 2,       # outer reweighting rounds (0 -> plain L1)
    reweight_eps: float = 1e-3,  # ε in wᵢ = 1/(|xᵢ| + ε)
    reweight_tol: float = 1e-3,  # outer between-round relative-change stop
    xp: ModuleType = np,
) -> tuple[ndarray, dict]:
    """Minimise ½⟨x,Hx⟩ − ⟨b,x⟩ + λ Σ wᵢ|xᵢ| over real x (≥ 0 when positive)."""
```

Internals:

- `grad(x) = hess(x) − b`; `q(x) = ½⟨x, hess(x)⟩ − ⟨b, x⟩`. The `hess(x)` matvec is computed once
  per point and reused for both `grad` and `q` (mirroring how the LS path reuses `A(v) − y`).
- Reuses the existing non-negative soft-threshold prox, FISTA momentum, backtracking line search,
  and CWB reweighting from the current module — no numerical logic is duplicated.
- `info` keys are identical to `fista`: `iterations` (list[int]), `reweights` (int), `objective`
  (float, `= q(x) + λ‖x‖₁`), `lipschitz` (float), `converged` (bool).

**Two distinct `eta`s.** The solver's `eta` is the **backtracking growth factor** (inherited
verbatim from `fista`, default `2.0`). It is unrelated to the operator/pipeline `eta`, which is the
**regulariser strength** (fraction of `Σw`) and maps to the solver's `lam = eta·wsum`. The operator
leaves the solver's `eta` at its `2.0` default; the two never interact.

**Refactor note.** Factor the current inner loop in `fista` so it drives on a smooth-term that
exposes `f_and_grad(x)` (at the momentum point) and `f(x)` (at backtracking trials). `fista` builds
that smooth-term from `(A, AH, y, weight)`; `fista_quadratic` builds it from `(hess, b)`. The shared
core holds the prox, momentum, backtracking, reweighting, and `info` assembly. The existing
`fista` public signature and behaviour must not change (the sky-model test in `test_fista.py` is the
guard).

**Lipschitz.** Because the smooth term is *quadratic*, its gradient Lipschitz constant is exactly
`‖H‖₂`, and `diag(H).max() ≤ ‖H‖₂` (Rayleigh bound for a PSD matrix). The operator passes
`L0 = float(hdiag.max())` — a free, tight lower bound from the diagonal `hessian_healpix` already
returns — so backtracking starts at the right scale and almost never fires. Backtracking remains as
the spec-mandated safety net (it inflates `L0` upward if the bound is loose); it is never removed.

## 5. Operator: `L1ReweightOperator` (`operators/l1_reweight.py`)

A drop-in sibling of `TikhonovOperator`, GPU-resident (`xp=cupy`), with **identical ports** so the
two are interchangeable in `compose()`:

- **Inputs:** `cube` (imager dirty = RHS), `WEIGHT`, `B_ROT`, `BORESIGHT`, `time_out`.
- **Outputs:** `cube` (regularised image → IWP), `dirty` (raw dirty passthrough → writer), `time_out`.

`__init__` mirrors `TikhonovOperator`'s shared args and swaps the CG knobs for FISTA knobs:

```python
def __init__(self, fragment, nside, freqs, eta, *args,
             nest=True, apply_beam=True, ground_plane_diameter=GROUND_PLANE_DIAMETER,
             max_iter=200, tol=1e-5, max_reweight=2, reweight_eps=1e-3,
             positive=True, use_warm_start=True, **kwargs): ...
```

`compute()` per frame (mirroring `TikhonovOperator.compute`):

1. Receive `dirty, weights, b_rot, boresight, time_out`; `w = weights[0]`, `wsum = w.sum()`.
2. **Fully-flagged frame** (`wsum == 0`): emit the all-zero map on `cube`, pass `dirty` through,
   emit `time_out`, leave the warm-start untouched — identical to `TikhonovOperator`.
3. Build the per-frame Airy `beam` (when `apply_beam`), then
   `hmv, hdiag = hessian_healpix(b_rot[0], pix_vec, freqs, w, beam=beam, xp=cp)`.
4. `b = dirty[0] * wsum` (un-normalise to the Hessian RHS, as Tikhonov does);
   `lam = self.eta * wsum`; `L0 = float(hdiag.max())`.
5. Warm start `x0 = self.x_prev` when `use_warm_start` and the previous solution is finite.
6. `x, _info = fista_quadratic(hmv, b, lam=lam, x0=x0, positive=self.positive, L0=L0,
   max_iter=self.max_iter, tol=self.tol, max_reweight=self.max_reweight,
   reweight_eps=self.reweight_eps, xp=cp)`; store `self.x_prev = x`.
7. Emit `x[None, :]` on `cube`, `dirty` on `dirty`, `time_out` on `time_out`.

No Jacobi preconditioner: the Lipschitz step (`1/L`) replaces it. The Airy `B = 0` below the horizon
zeroes those Hessian rows/columns; the gradient there is 0 and the non-negative prox keeps those
pixels at 0 (no divergence, no division — consistent with the dirty/Tikhonov handling).

## 6. Pipeline wiring: `--regulariser` selector

`eta > 0` stays the **activation gate** (unchanged), and a new `regulariser` choice picks the
algorithm. Defaulting it to `"tikhonov"` keeps every existing `eta`-only invocation behaving exactly
as today — no recipe or cab migration.

- **`core/smoovie.py`:** add `regulariser: str = "tikhonov"` to `SmooviePipeline.__init__`,
  `image_via_app`, and `smoovie`. In `compose()`, the existing `regularise = eta is not None and
  eta > 0` branch instantiates `TikhonovOperator` when `regulariser == "tikhonov"` and
  `L1ReweightOperator` when `regulariser == "l1"`; the add_flow wiring and the writer's
  `regularised`/`dirty_raw` schema are identical for both (only the operator class differs). An
  unknown value raises `ValueError`.
- **`cli/smoovie.py`:** add
  `regulariser: Annotated[Literal["tikhonov", "l1"], typer.Option(help="…")] = "tikhonov"` and thread
  it through the three `dict(...)` parameter blocks and the `smoovie_core(...)` call, exactly like the
  other options. The pre-commit hook regenerates `cabs/smoovie.yml`; `tests/test_roundtrip.py` guards
  the cli↔cab agreement.

## 7. Cost (TART scale)

- nbl = 276, nchan = 1, npix = 49 152 (nside 64) / 196 608 (nside 128, the `smoovie` default).
- Kernel `exp(1j·_phase)` is `(nbl, nchan, npix)` complex128 → 217 MB (nside 64) / 868 MB (nside 128),
  built once per frame.
- One matvec ≈ 1.3·10⁷ complex MACs — sub-millisecond on a GPU. Per-frame cost is
  `matvec-count × that`; FISTA + reweighting calls the matvec on the order of hundreds of times per
  frame. The dominant lever is the §2/§4 reuse (one matvec per iteration via the well-seeded
  Lipschitz step), not the kernel build.

## 8. Edge cases

- **Fully-flagged frame** (`wsum == 0`): handled exactly as `TikhonovOperator` (zero map, warm-start
  preserved). `lam = eta·wsum = 0` and `b = 0` never reach the solver.
- **Below-horizon pixels** (`B = 0`): zero Hessian rows/cols; non-negative prox keeps them at 0.
- **`eta` set but `regulariser` unknown:** `ValueError` from `compose()`.
- **First frame / no warm start:** `x0 = None` → zeros; FISTA starts from the dirty's gradient step.
- **Loose `L0`:** backtracking inflates `lipschitz`; the iterate cannot diverge (guaranteed by the
  inherited FISTA backtracking, already tested in `test_fista.py`).

## 9. Test Plan (TDD, `tests/test_fista.py`, CPU `xp=np`)

All tests stay offline and use `numpy`. The operator's numerics are validated through
`fista_quadratic` + `hessian_healpix` on the host (no GPU/Holoscan), mirroring how the Tikhonov spec
exercised `cg` + `hessian_healpix` directly.

1. **Quadratic matches least-squares.** For a small dense complex operator `A`, real weights `W`,
   data `y`: build `H = Aᴴ W A` (matvec) and `b = Re{Aᴴ W y}`. Assert `fista_quadratic(H, b, lam,
   positive=False)` matches `fista(A, AH, y, weight=W, lam, positive=False)` to `tol`.
2. **Closed-form prox optimum.** With `H = I`, `b = y` (real), `positive=False`, the minimiser is
   `soft_threshold(y, lam)`; assert `fista_quadratic` matches.
3. **Backtracking from a tiny `L0`.** A deliberately small `L0` (e.g. `1e-6`, far below `‖H‖₂`) still
   converges to the same optimum as a well-seeded `L0`; assert `info["lipschitz"]` grew above it.
4. **Non-negativity.** `positive=True` ⇒ `all(x >= 0)` even when signed L1 would push coefficients
   negative.
5. **Reweighting debiases.** On a known `k`-sparse non-negative `x_true`, `max_reweight>0` recovers
   amplitudes closer to `x_true` than `max_reweight=0`; support matches.
6. **Recover a sparse sky through `hessian_healpix`.** Plant a few non-negative point sources on a
   small HEALPix grid, form `H`/`b` via `hessian_healpix` + `dft_forward` (beam on), recover with
   `fista_quadratic`. This is the integration test proving the operator's solve path end-to-end on
   the real Hessian.
7. **Zero RHS short-circuit.** `b = 0` returns `x0` unchanged, `converged` reported.

`tests/test_roundtrip.py` / `tests/test_structure.py` pick up the new `--regulariser` CLI option.

## 10. Comparison deliverable: `scripts/compare_regularisers.py`

A host/CPU script (sibling of `scripts/validate_tart_gains.py`) that images one TART frame three
ways — raw dirty, Tikhonov (`cg` + `hessian_healpix`), reweighted-L1 (`fista_quadratic` +
`hessian_healpix`) — at matched `eta`, and reports, over above-horizon pixels:

- point-source concentration (e.g. peak/`Σ|x|`, or fraction of flux in the brightest `k` pixels),
- recovered flux at the catalogue satellite positions vs. the dirty/Tikhonov estimates.

This answers "how does reweighted-L1 do against Tikhonov" directly, with no GPU dependency.

## 11. Conventions

- `opt/fista.py` follows `utils/` import rules (imports at module top; no lazy imports); generic and
  `xp`-injectable; no cab. ruff line length 120, `select = E/F/I/N/W`, `N806` enforced (no uppercase
  locals — `lipschitz`, not `L`; `hess`/`b` are argument names, allowed under N803).
- `operators/l1_reweight.py` is an inherently GPU-only operator: `cupy`/`holoscan` at module top, as
  in `tikhonov.py`.
- Google-style docstrings, concise, matching `cg.py` / the existing `fista`.
- After every change: `uv run ruff format . && uv run ruff check . --fix`.

## 12. Out of Scope

- **Cross-frame / local-frame Hessian reuse** (§3) — deferred in favour of ICRS per-frame for
  multi-TART forward-compatibility.
- **The self-cal driver** that feeds recovered fluxes back into amplitude gain estimation — the next,
  separate spec (this stage produces the clean flux image that step consumes).
- **complex64 / float32 kernel**, **imager↔regulariser kernel fusion**, and **diagonal-preconditioned
  (variable-metric) FISTA** — identified optimisations below the confidence bar for this pass; revisit
  if profiling on real hardware shows the per-frame solve is a bottleneck.
- **GPU (`cupy`) CI coverage** — numerics are `numpy`-tested, consistent with the rest of the suite.
- **Exposing the full FISTA tuning surface** (`max_iter`, `max_reweight`, `reweight_eps`, …) at the
  CLI — kept as operator defaults for now; promotable to CLI flags later if experimentation needs it.
