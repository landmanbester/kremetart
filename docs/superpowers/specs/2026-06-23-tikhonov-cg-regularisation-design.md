# Tikhonov-Regularised Imaging via Conjugate Gradients — Design

**Date:** 2026-06-23
**Status:** Approved (brainstorming) — ready for implementation
**Scope:** A generic CUDA-capable (xp-injectable) conjugate-gradient solver, an efficient
HEALPix Hessian operator, and a Holoscan operator inserted between the imager and the
IWP-Kalman filter that solves a per-frame Tikhonov-regularised normal-equations problem to
correct the dirty image for the instrument's sampling density and primary-beam response. The
IWP then operates on the regularised image.

---

## 1. Purpose

The HEALPix imager produces a per-frame dirty image with the measurement operator
`R = M ∘ diag(B)` (geometric DFT `M` plus the Airy power beam `B`). The dirty image is the
apparent sky convolved with the snapshot PSF and tapered by the beam. This change forms the
Tikhonov/MAP estimate of the sky by solving the regularised normal equations with conjugate
gradients, correcting (within the measured subspace) for both the sampling PSF (`MᴴWM`) and the
beam (`B²`).

The CG solver and Hessian are intentionally generic so the same machinery applies to the
**residual** once a calibration operator exists (solve `(H+λI)x = Rᴴ W r` with `r` the residual
instead of the data) — only the RHS changes.

---

## 2. Key decisions (from brainstorming + adversarial review)

1. **Operator does NOT consume visibilities.** The Hessian `H = B Mᴴ W M B` needs only weights,
   geometry, and beam; the RHS is the imager's dirty image. The new operator therefore receives
   `WEIGHT`, `B_ROT`, `BORESIGHT` (fanned from the reader) plus the imager's dirty map and
   timestamp. No `VISIBILITY` port. The imager is unchanged.
2. **Off by default; `η>0` inserts the operator.** Holoscan graphs are static, so `--eta` unset
   leaves today's `reader → imager → iwp → writer` path untouched; `η>0` composes the operator in.
   `η` must be strictly positive (`λ = η·Σw > 0`) for `H + λI` to be SPD.
3. **IWP consumes the regularised image; the raw dirty is kept.** The durable zarr stores both
   `dirty` (raw) and `regularised`; IWP filters `regularised`.
4. **`opt/cg.py` new subpackage.** A dedicated `src/kremetart/opt/` for solvers (user's call; a
   deliberate deviation from the documented `utils/` location — `tests/test_structure.py` only
   enforces the cli/core/cabs bijection, so it does not object).
5. **Efficient Hessian + Jacobi preconditioner + warm-start**, the last two behind flags so the
   self-adjointness/reference CI tests run unpreconditioned and zero-initialised.

---

## 3. Math — Tikhonov normal equations (un-normalised Hessian)

Per frame, solve for the real sky image `x` (npix,), with `λ = η·Σw` (η a dimensionless fraction
of the central PSF value `psf[0,0] = Σw`, so the regulariser has consistent meaning across
frames with different weighting/flagging):

```
(H + λ I) x = b ,   H x = Σ_c B[c] ⊙ Re{ M_cᴴ W_c M_c (B[c] ⊙ x) } ,   b = Σ_c B[c] ⊙ Re{ M_cᴴ W_c vis_c } ,   λ = η·Σw
```

- `M_c` is the per-channel geometric DFT, `B` the power beam `(nchan, npix)`, `W_c` the per-channel
  weights; channels are summed into a single MFS image (matching the imager). `H` is symmetric
  positive semi-definite over the reals → `H + λI` is SPD → CG converges.
- **RHS reuses the imager's dirty map.** The imager emits `dirty = b / Σw` (normalised by the total
  weight `Σw = Σ_{r,c} W`). The operator recovers the un-normalised RHS as `b = dirty · Σw` — no
  extra adjoint, no visibilities.
- **Hessian diagonal is closed-form** (because `|kernel| = 1`): `diag(H)_j = Σ_c B[c,j]² · w_sum_c`,
  with `w_sum_c = Σ_r W[r,c]`. For the single-channel TART case this is `B² · Σw`. Computed from
  `beam` and `weights` alone — no kernel.
- **Jacobi preconditioner:** `M_diag = diag(H) + λ = Σw·(B² + η)`; apply `r → r / M_diag`.
  `1/M_diag` is precomputed once per frame (cached) since `B`, `W` and `Σw` are fixed within a frame.
- Below-horizon pixels (`B = 0`) reduce to `λ x = b = 0` → remain exactly 0 (no division blowup).
- The image is real: the matvec takes `Re{·}` and CG runs over `ℝ^npix`.
- If the imager runs with `apply_beam=False`, the operator uses `beam=None` (then `diag(H)_j = Σw`),
  so its `R` matches the imager exactly.

> Scaling `λ = η·Σw` makes `η` a frame-invariant knob: both `H` and `λ` scale with `Σw`, so the
> solution depends only on the dimensionless ratio `η` regardless of how a frame is weighted/flagged.

---

## 4. Components

### 4.1 `opt/__init__.py`, `opt/cg.py`
Generic preconditioned conjugate gradients, xp-injectable, plain-callable operators (no
`scipy.LinearOperator`, which is host/NumPy-oriented and does not ride cupy cleanly):

```python
def cg(A, b, *, x0=None, M=None, maxiter: int = 100, tol: float = 1e-5, xp=np):
    """Solve A x = b for SPD A via (optionally preconditioned) CG.

    A:  callable(x) -> A @ x (the linear operator).
    M:  optional callable(r) -> M⁻¹ r (Jacobi/other preconditioner); None = unpreconditioned.
    x0: optional warm-start initial guess; None = zeros.
    Returns x (same shape as b). Relative-residual stopping ‖r‖ ≤ tol·‖b‖.
    """
```

`M=None, x0=None` is the clean reference path the CI tests use.

### 4.2 `utils/healpix_dft.py::hessian_healpix`
```python
def hessian_healpix(b_rot, pix_vec, freqs, weights, *, beam=None, xp=np) -> tuple[callable, "ndarray"]:
    """Return (matvec, diagonal) for the per-frame image-space Hessian H = B Mᴴ W M B.

    matvec:   callable(x: (npix,)) -> H x (real, (npix,)). Builds the DFT kernel once and reuses
              it across calls (forward then weighted adjoint), so each CG iteration is two
              contractions — no kernel rebuild.
    diagonal: (npix,) closed-form diag(H) = Σ_c B[c]² · Σ_r W[r,c]  (no kernel).
    """
```
`b_rot` is the per-frame `(nbl, 3)` rotated baselines (rows); `weights` is `(nbl, nchan)`. Lives
beside `dft_forward`/`dft_adjoint`; reuses `_phase`.

### 4.3 `operators/tikhonov.py::TikhonovOperator`
- Ctor: `nside`, `freqs`, `nest`, `apply_beam`, `ground_plane_diameter`, `eta`, `maxiter`, `tol`,
  `use_preconditioner=True`, `use_warm_start=True`. Builds `pix_vec` on device in `start()`;
  holds `self.x_prev = None` (device-resident warm-start state).
- Inputs: `cube` (RHS dirty) + `time_out` from the imager; `WEIGHT`, `B_ROT`, `BORESIGHT` from the
  reader. No `VISIBILITY`.
- `compute`: build `beam` from `BORESIGHT` (mirroring the imager); `(Hmv, Hdiag) =
  hessian_healpix(b_rot, pix_vec, freqs, weights, beam=beam)`; `wsum = weights.sum()`;
  `lam = eta * wsum`; `A = lambda x: Hmv(x) + lam*x`; `b = dirty * wsum`;
  `M = (lambda r: r * inv_Mdiag)` with `inv_Mdiag = 1/(Hdiag + lam)` when
  `use_preconditioner` else `None`; `x0 = self.x_prev` when `use_warm_start` and `x_prev` is finite
  else `None`; `x = cg(A, b, x0=x0, M=M, maxiter=..., tol=...)`; `self.x_prev = x`. Emit
  `cube = x[None,:]` (→ IWP), `dirty` = raw passthrough (→ writer), `time_out`.
- **Warm-start guard:** first frame (`x_prev is None`) or a non-finite `x_prev` → fall back to
  zeros. (No scene-change detector — YAGNI for a sidereal stream.)

### 4.4 `core/smoovie.py`
- Thread `eta` through `smoovie → image_via_app → SmooviePipeline`.
- In `compose`, when `eta` is set (`>0`): insert `TikhonovOperator` between imager and IWP, fan the
  reader's `WEIGHT`/`B_ROT`/`BORESIGHT` to it, route its `dirty` passthrough to the writer.
  Otherwise compose today's graph unchanged.

### 4.5 `cli/smoovie.py`
- Add only `--eta` (`float | None`, default `None` = off). `maxiter`/`tol`/the feature flags stay
  internal so `core.smoovie` mirrors `cli.smoovie` (test_structure) with just `eta` added.

### 4.6 `operators/io.py::HealpixWriterOperator`
- Generalise the stored-variable list: default `("dirty", "filtered", "znorm")`; when `eta>0` the
  app builds it as `("dirty", "regularised", "filtered", "znorm")`, taking `dirty` (raw) from the
  Tikhonov operator and `regularised`/`filtered`/`znorm` from the IWP.

---

## 5. Data flow (η>0)

```
reader ─┬───────────────────────────► imager ──(cube=dirty RHS, time)──► tikhonov ──► iwp ──► writer
        └─(WEIGHT, B_ROT, BORESIGHT)───────────────────────────────────►  │                    ▲
                                            tikhonov emits: cube=regularised ┘                    │
                                                            dirty=raw passthrough ────────────────┘
```

The live web sink (when serving) keeps its current wiring; its `raw` panel shows `regularised`
when η>0. Raw dirty is retained only in the durable zarr.

---

## 6. Caveats (documented, not blockers)

- **Per-frame underdetermination:** ~276 visibilities vs ~196k pixels (nside=128). `H` is rank
  ≤ ~2·n_vis; per single frame the solve is λ-dominated and corrects sampling only within the
  measured subspace. Full sampling-density correction still emerges from the IWP accumulation.
- **Warm-start is well-motivated, not just perf:** the sky is ~stationary in the fixed ICRS grid,
  so the previous frame's solution seeds the next; only the beam/sampling change per frame.
  Device-resident across frames; reset on first frame / non-finite guard.
- **IWP retuning:** the regularised image has different units than the `/Σw` dirty map and
  λ-dependent, spatially-correlated noise; `sigma2`/`noise` and the `znorm` interpretation will
  shift. The per-pixel-independence assumption is already an approximation for the (PSF-correlated)
  dirty map; Tikhonov increases the correlation in degree, not kind.
- **Kernel memory:** the cached kernel is `(nrow, nchan, npix)` complex — ~0.4–0.9 GB at nside=128,
  ~3 GB at nside=256 (complex128). complex64 is a future lever if needed.

---

## 7. Testing (CPU, xp=numpy)

- `hessian_healpix`:
  - **self-adjointness:** `⟨H x, y⟩ == ⟨x, H y⟩` for random real `x`, `y` (M=None reference).
  - **diagonal:** the closed-form `diagonal` equals the explicit `diag` of a densely-assembled `H`
    on a small grid.
- `cg` (unpreconditioned, zero-init reference):
  - solves random SPD systems to agree with `np.linalg.solve`;
  - `(H+λI)x = b` recovered to a dense direct solve on a small grid;
  - preconditioned CG reaches the same solution in fewer iterations;
  - warm-start (`x0` near the solution) reduces iterations and lands on the same answer.
- A forward-modelled point source is sharper (lower sidelobes) after the solve than in the raw
  dirty map.
- Round-trip + structure tests pass with the new `eta` parameter.
- GPU operators are not CPU-unit-tested; an end-to-end `SmooviePipeline` smoke run with `eta>0`
  confirms the port wiring and that `dirty`/`regularised` are both written.

---

## 8. Out of scope

- Calibration / residual formation (the CG + Hessian are built generic to accept it later).
- Exposing `maxiter`/`tol`/preconditioner/warm-start on the CLI (internal for now).
- A configurable scene-change detector for warm-start (first-frame + non-finite guard only).
- Non-identity Tikhonov priors (wavelet/gradient) and positivity constraints.
- A separate live-viewer panel for the raw dirty map (kept in the durable zarr only).
