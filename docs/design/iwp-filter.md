# Per-pixel IWP–Kalman whitening filter for smoovie

**Date:** 2026-06-17
**Status:** Design approved, ready for implementation plan
**Scope:** A single new Holoscan operator that adds the integrated-Wiener-process (IWP)
state evolution from `docs/tex/unified/kremetart_design.tex` (sec:iwp, sec:kf, sec:frame)
to the `smoovie` streaming pipeline, as a first increment toward the full calibration /
detection machinery.

## 1. Motivation and context

`smoovie` today is a three-operator Holoscan app (`src/kremetart/core/smoovie.py`):

```
reader (HealpixZarrReaderOperator) → imager (HealpixDFTOperator) → writer (HealpixWriterOperator)
```

The imager emits one `(1, npix)` HEALPix dirty-map row plus a mean timestamp per
sub-integration; the writer regions them into a `(TIME, PIX)` zarr; the host renders the
maps to an mp4.

The design note describes, per equatorial pixel, an **IWP–Kalman whitening filter** over
the sidereally-fixed HEALPix light curves (sec:frame). Because the imager already produces
maps on a fixed equatorial grid, each pixel's time series is directly a quiescent light
curve the IWP is built to track. This spec adds that filter as **one new streaming
operator** to get a feel for the architectural changes required, without yet building
calibration, robust likelihoods, or the sequential detector.

The filter holds its Kalman state `(x_{k|k}, P_{k|k})` as instance attributes, receives
each new dirty map (the observation `y_k`) and timestamp from the imager, runs the exact
predict + update recursion, and emits the filtered flux and the whitened (normalised)
innovation downstream.

## 2. Decisions (locked)

| Axis | Decision |
|---|---|
| Granularity | **Per-pixel across the whole map** — `npix` independent filters, vectorised over the HEALPix grid in cupy. |
| Outputs | **Normalised innovation `z_k`** (the whitening diagnostic) **and filtered flux `x_{k|k}[0]`** (denoised movie). |
| Model order | **q = 1** (constant-velocity / locally-linear IWP; the doc default). |
| Hyperparameters | Constant scalar **driving variance σ²** and **measurement noise R**, exposed as CLI knobs. Not estimated online (matches the "driving variances are fixed offline" stance). |
| Activation | **Always-on** — every `smoovie` run computes the filter and writes its outputs. No gating flag. |
| Output surfacing | **Approach A — three movies:** dirty (unchanged primary mp4) + `<movie>.filtered.mp4` + `<movie>.znorm.mp4`. |
| Persistence | **Durable `<movie>.zarr`** holding `dirty`/`filtered`/`znorm`. Output only; prepared MSv4 + empty config stay in a temp dir. |
| Overwrite | `--overwrite` flag; **fail-fast** with `FileExistsError` if `<movie>.zarr` already exists and the flag is unset, raised *before* any prepare/imaging work. |
| Diffuse-prior init | Internal constant (not a CLI knob). |
| Parameter names | `iwp_sigma` (σ²), `iwp_noise` (R), `overwrite`. |

## 3. Mathematics (q = 1)

Per-pixel state `x = (f, ḟ)ᵀ ∈ ℝ²`; observation `y_k` = dirty-map pixel value; `H = (1, 0)`.

**Exact discretisation (eq. AQ), scalar Δ per frame:**

```
A(Δ) = [[1, Δ],
        [0, 1]]

Q(Δ) = σ² · [[Δ³/3, Δ²/2],
             [Δ²/2, Δ  ]]
```

**Predict** (`x_{k|k-1} = A x_{k-1|k-1}`, `P_{k|k-1} = A P A^T + Q`), vectorised over the
`npix` pixels (`X:(npix,2)`, `P:(npix,2,2)`; `A`, `Q` are `(2,2)` and broadcast).

**Update** (`H = (1,0)`, scalar `R`):

```
e_k = y_k − x_{k|k-1}[0]                        (innovation, (npix,))
S_k = P_{k|k-1}[0,0] + R                        (innovation variance, (npix,))
K_k = P_{k|k-1}[:, :, 0] / S_k                  (gain, (npix,2))
x_{k|k} = x_{k|k-1} + K_k · e_k
P_{k|k} = (I − K_k H) P_{k|k-1} (I − K_k H)^T + K_k R K_k^T    (Joseph form)
z_k = e_k / sqrt(S_k)                            (normalised innovation)
filtered_k = x_{k|k}[0]
```

Joseph form is used (over the shorter `(I − KH)P`) for numerical robustness across the many
parallel filters, per sec:kf.

**Irregular cadence (hard requirement, sec:statespace).** `Δ_k = t_k − t_{k-1}` is read
from the timestamp stream every frame. No stage caches fixed-step `A`/`Q` matrices or
assumes uniform Δ.

**Diffuse-prior initialisation.** `start()` sets `X = 0`, `P = diag(diffuse, diffuse)` with
a large internal constant; `t_prev = None`. Frame 0 therefore skips the predict (no Δ yet)
and runs update-only against the diffuse prior: `filtered_0 ≈ y_0` and `z_0 ≈ 0` (a warm-up
sample). Frames ≥ 1 run the full predict + update.

## 4. Module layout

Mirrors the existing `HealpixDFTOperator` ↔ `utils/healpix_dft.py` split (thin GPU operator
over xp-generic numerical kernels). New files:

### `src/kremetart/utils/iwp.py` (xp-generic, CPU-testable)

Pure functions parameterised by `xp` (numpy *or* cupy), importable without a GPU:

- `iwp_transition(dt: float, sigma2: float, *, xp) -> tuple[Array, Array]` → `(A, Q)` for q=1.
- `kalman_predict(X, P, A, Q, *, xp) -> tuple[Array, Array]` → `(X_pred, P_pred)`, vectorised over pixels.
- `kalman_update(X_pred, P_pred, y, R, *, xp) -> tuple[Array, Array, Array, Array]` → `(X_kk, P_kk, e, S)`, Joseph form.

These carry public (non-underscore) names — they are imported by the operator (see
`python-standards.md` §6) — and import numpy at module top (no GPU deps).

### `src/kremetart/operators/iwp_kalman.py` (GPU operator)

`cupy` / `holoscan` imported at module top (inherently GPU-only operator, like
`dft_healpix.py`).

```python
class IWPKalmanOperator(Operator):
    def __init__(self, fragment, npix, *args, sigma2, noise, **kwargs): ...
    def start(self):
        # X = cp.zeros((npix, 2)); P = diffuse diag; t_prev = None
    def setup(self, spec):
        spec.input("cube"); spec.input("time_out")
        spec.output("cube"); spec.output("filtered"); spec.output("znorm"); spec.output("time_out")
    def compute(self, op_input, op_output, context):
        # receive cube (1,npix) + time_out (1,)
        # predict (if t_prev is not None) then update via utils.iwp (xp=cp)
        # emit cube (passthrough), filtered (1,npix), znorm (1,npix), time_out (1,)
```

The operator passes `cube` through so a single writer handles all three map streams from
one upstream.

## 5. App, writer, and renderer changes

### `core/smoovie.py`

- **Pipeline:** `reader → imager → iwp → writer`. Wire `imager → iwp` on `{("cube","cube"),
  ("time_out","time_out")}` and `iwp → writer` on `{("cube","cube"), ("filtered","filtered"),
  ("znorm","znorm"), ("time_out","time_out")}`.
- **`SmooviePipeline`** gains `sigma2` / `noise` and constructs the `IWPKalmanOperator`.
- **`image_via_app(..., output_zarr, overwrite, sigma, noise)`**: writes to the durable
  `output_zarr` (no `TemporaryDirectory` for the output; prepared MSv4 + empty config stay
  in a temp dir). Returns `(dirty, filtered, znorm, stamps)` — all eager-loaded before any
  temp cleanup.
- **`smoovie()`**: derive `output_zarr = Path(str(movie) + ".zarr")`. **Fail-fast:** if it
  exists and `overwrite` is False, raise `FileExistsError` before `prepare`/imaging. With
  `overwrite`, the writer's `mode="w"` clobbers. Render three movies via `render_frames`:
  - `<movie>` — dirty (unchanged).
  - `<movie>.filtered.mp4` — filtered flux (inferno/percentile, like dirty).
  - `<movie>.znorm.mp4` — normalised innovation (diverging, symmetric scale).
  Track overlays (if `overlay_catalog`) apply to all three.

### `operators/io.py` — `HealpixWriterOperator`

- `start()` scaffold gains `filtered` and `znorm` data vars alongside `dirty`, same
  `(TIME, PIX)` dims/chunks.
- `setup()` gains `filtered` / `znorm` inputs (keeps `cube`, `time_out`).
- `compute()` region-writes all three variables.

### `render_frames`

- Add a `diverging: bool = False` keyword. When True: symmetric limits centred on 0
  (`vmax = percentile(|stacked|, 99)`, `vmin = -vmax`) and a diverging cmap (`coolwarm`),
  overriding the supplied `cmap`/percentile path used for dirty and filtered.

## 6. CLI and cab

`cli/smoovie.py` and `core/smoovie.py` both gain (mirror rule, `architecture.md` §1):

- `iwp_sigma: float` — driving variance σ² (default a sensible small constant).
- `iwp_noise: float` — measurement noise R (default a sensible small constant).
- `overwrite: bool = False`.

All three are ordinary cab inputs (none are backend flags, so no `StimelaMeta(skip=True)`).
The cab regenerates automatically via the pre-commit hook. The existing `smoovie`
round-trip case in `tests/test_roundtrip.py` exercises the new parameters; no new
round-trip case is required (no new command).

`tests/test_structure.py` is unaffected: `utils/iwp.py` and `operators/iwp_kalman.py` are
not commands, so they need no `cli`/`core`/`cab` triple.

## 7. Testing

### `tests/test_iwp.py` (CPU / numpy, no GPU)

- `iwp_transition` matches the closed forms in eq. AQ for representative Δ.
- On synthetic IWP-generated data with known σ²/R, the filter's innovations are white and
  **NIS lies within its χ² bounds** (sec:nis) — the headline correctness check.
- Joseph update keeps `P` symmetric and positive semi-definite across many steps.
- xp-generic functions give identical results for `xp=numpy` (and, where a GPU is present,
  agree with `xp=cupy`).

### GPU end-to-end (extend the existing smoovie e2e)

- `<movie>.zarr` exists after the run and contains `dirty`, `filtered`, `znorm` with shape
  `(ntime, npix)`.
- `znorm` is finite; `filtered` tracks a smoothed `dirty`.
- Three mp4s are produced.

### Overwrite behaviour (host, stubbed imaging)

- A pre-existing `<movie>.zarr` raises `FileExistsError` when `overwrite` is unset and
  succeeds when set. The existing host tests that **stub `image_via_app`** must be updated
  for its new signature and `(dirty, filtered, znorm, stamps)` return.

## 8. Explicitly out of scope (this increment)

Calibration EKF and its Jacobian, gain/flux IWP states, gauge fixing / reference-antenna
handling, the robust heavy-tailed likelihood, Rao–Blackwellised fluxes, the CUSUM/GLR
sequential detector and its ring buffers, below-horizon freeze semantics, deriving `R` from
imaging weights, the q=2 (constant-acceleration) variant, the multi-scale model bank, and
offline hyperparameter (driving-variance) calibration. This increment delivers **only** the
per-pixel quiescent IWP whitening filter as a streaming Holoscan operator.

## 9. Affected files (summary)

| File | Change |
|---|---|
| `src/kremetart/utils/iwp.py` | **New** — xp-generic IWP transition + Kalman predict/update. |
| `src/kremetart/operators/iwp_kalman.py` | **New** — GPU `IWPKalmanOperator` holding `(X, P, t_prev)`. |
| `src/kremetart/operators/io.py` | `HealpixWriterOperator` writes `dirty`/`filtered`/`znorm`. |
| `src/kremetart/core/smoovie.py` | Wire `iwp` into pipeline; durable zarr + fail-fast overwrite; render 3 movies; `render_frames` diverging mode. |
| `src/kremetart/cli/smoovie.py` | Add `iwp_sigma`, `iwp_noise`, `overwrite` params. |
| `src/kremetart/cabs/smoovie.yml` | Auto-regenerated (pre-commit). |
| `tests/test_iwp.py` | **New** — CPU filter correctness + whitening/NIS. |
| `tests/<existing smoovie tests>` | Update stubs for new `image_via_app` signature; assert new zarr vars + movies. |
