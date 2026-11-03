# StefCAL calibration core — design

**Date:** 2026-06-24
**Status:** approved (brainstorm), pending implementation plan
**Scope:** the acquisition StefCAL step of the Stage-1 calibration operator, built as
validated xp-injectable math only. Streaming/GPU operator wiring is deferred.

## 1. Background and goal

`kremetart`'s Stage-1 pipeline (per `docs/tex/unified/kremetart_design.tex`) places a
**calibration operator between the reader and the imager**. The full operator is a stateful,
robust iterated-EKF that tracks gains and source fluxes. This spec covers only the **first
piece**: the StEFCal *acquisition* (cold-start) solve that aligns gain phases at the outset,
before any EKF tracking exists.

The measurement model for a baseline `(p, q)` is

```
V_pq = g_p · M_pq · conj(g_q) + noise,     g_p = exp(a_p + i ψ_p)
```

where `M_pq` is the known sky-model coherency. For this acquisition step **every catalogued
source is modelled at unit flux** (`A_s ≡ 1`); no flux solving happens yet. StEFCal exploits
that, holding all other antenna gains fixed, `V_pq` is linear in `g_p`, giving an
antenna-by-antenna complex linear least squares with no phase-wrapping or local-minimum
problem — ideal for cold start.

**Why phases only.** Because the real GNSS sources span `10^5`–`10^7` Jy while the model
assumes flux 1, the solved **amplitudes** absorb the flux mismatch and are unreliable. The
**phases**, driven by source geometry, are robust to the flux model. So once StEFCal
stabilises we keep only the referenced phases and discard amplitudes. Those phases seed the
later self-cal / IWP-EKF (out of scope here).

**Frames.** Calibration is simplest in the local **ENU** frame: the catalogue gives source
az/el, the satellites move in this frame, and the geometric fringe uses the raw ENU baselines.
Rotation into the equatorial (ICRS) frame is purely an imaging concern and happens **strictly
downstream** of calibration.

```
reader (ENU baselines, vis, catalogue)
   → sky model   M_pq               ┐
   → StefCAL     g_p (keep phases)  ┘  ALL in the ENU frame
   → [later operator] phase correction
   → equatorial_baselines  C(t)ᵀ·b        ← rephasing happens ONLY here, after calibration
   → imager
```

## 2. Scope of this first cut

**In scope** (pure `xp`-injectable host math, runs under `xp=numpy`, ready for `xp=cupy`; no
Holoscan, no GPU requirement):

1. A sky-model coherency util that builds model visibilities over the catalogue sources in the
   ENU frame, with an **optional primary-beam weight** (Airy power beam; apparent-flux
   down-weighting of low-elevation sources).
2. A StEFCal solver util (alternating per-antenna complex LS) with gauge fixing, phase
   extraction, and a **`t_int` solution interval** that pools consecutive integrations into one
   gain solution (sources move between integrations, so the model varies per integration).
3. Catalogue glue that yields per-frame `(name, az, el)` arrays aligned to the vis frames,
   with an injectable `fetch` for hermetic tests.
4. Tests: a simulation round-trip as the correctness gate (incl. a multi-integration pooled
   round-trip), plus an opt-in real-data sanity check against TART's own gain snapshot (all
   sources, beam-weighted, whole-file pooled).

**Explicitly deferred** (recorded so the core stays forward-compatible, not built now):

- The Holoscan `operators/calibration.py` + `core` driver that slots between reader and imager.
- Moving the equatorial rotation downstream of calibration in the prepared-zarr path
  (`prepare_msv4_zarr` currently rotates to equatorial *before* writing).
- Flux solving, the Student-t robust/IRLS weighting, the IWP-EKF tracking, fixed-lag
  smoothing, model subtraction, and reference-antenna failover/re-gauging.

### 2.1 Post-investigation findings (2026-06-24)

Chasing the disagreement between the StEFCal phases and TART's stored per-file gain snapshot,
established by experiment:

- **No sign/convention bug.** The ENU `b·s` convention matches the validated imaging path to
  ~5 mrad (the conjugate is fully wrong), so the baseline sign (`b = pos(a1)−pos(a2)`), forward
  fringe (`exp(+i b·s)`) and gain conjugation (`V = g_p conj(g_q)`) are all correct.
- **The solver is correct.** The noiseless simulation round-trip recovers gains to `1e-6`; with
  the *correct* flux-weighted model it recovers exactly even at `10⁶` flux dynamic range.
- **Unit flux is wrong but not the dominant real-data effect.** In simulation, equal-flux
  modelling corrupts phases (0.4–1.65 rad at realistic flux spread); on *this* real frame the
  disagreement is flux-invariant (a few bright GNSS sources dominate), so neither catalogue-`jy`
  weighting nor a flux solve moves it.
- **The snapshot is not a per-frame ground truth.** Our solution fits this frame's visibilities
  *better* than TART's snapshot (relative residual 0.54 vs 0.61; uncalibrated 0.97), and the
  snapshot is one solution per ~minute file.
- **Beam beats a hard elevation cut.** Folding the Airy beam in (low-elevation down-weighting)
  plus whole-file pooling reduces the gauge-invariant phase RMS from 23° → **18.8°**; hard
  elevation floors only make it worse (keep all sources, let the beam weight them).
- **Decision:** accept ~19° as model-limited for the acquisition step (closing it further needs a
  better element-beam model / catalogue fidelity, not an algorithm change), wire beam + `t_int`
  into the real-data check, and move on.

## 3. Architecture and module factoring

Three small additions under `src/kremetart/utils/` (host helpers, public names, `xp`-injectable
— matching the repo convention of one concern per util, e.g. `gains.py`, `healpix_dft.py`):

| Module | Responsibility | Depends on |
|---|---|---|
| `utils/skymodel.py` | `enu_direction_cosines(az, el)` → `(nsrc, 3)`; `model_visibilities(s_enu, bl_enu, freqs, *, xp)` → `(nbl, nchan)` unit-flux model. Self-contained fringe kernel over the ~100 sources. | none (pure `xp`) |
| `utils/stefcal.py` | `stefcal_solve(...)` → complex gains + convergence info; `referenced_phases(gains, ref_ant, *, xp)` → phases with amplitude discarded. | none (pure `xp`) |
| catalogue glue (in `utils/satellites.py`) | a function returning per-frame `(az, el, name)` arrays aligned 1:1 with the vis frames, reusing the existing injectable `fetch`/cache machinery. | `satellites.py` |

**No reuse of the imaging DFT.** `healpix_dft.dft_forward` evaluates over the full HEALPix grid
(`10^4`–`10^5` pixels) and carries imaging/equatorial vocabulary (`pix_vec`). The calibration
model evaluates over only the ~100 visible satellites, so `skymodel.py` carries its own small
fringe kernel and stays decoupled from the imager. Calibration and imaging share the *form* of
the fringe, not the code path.

## 4. Component detail

### 4.1 `utils/skymodel.py`

- `enu_direction_cosines(az, el, *, xp=np) -> (nsrc, 3)`: source unit vectors in ENU
  `(East, North, Up)`. Convention: az measured from North toward East, so
  `ŝ = (sin az · cos el, cos az · cos el, sin el)`. Verified (§2.1) consistent with the imaging
  path's `b = pos(a1)−pos(a2)` / `exp(+i b·s)` convention to ~5 mrad.
- `model_visibilities(s_enu, bl_enu, freqs, *, beam=None, xp=np) -> (nbl, nchan)`: model coherency

  ```
  M_pq = Σ_s B_s · exp(2πi (ν/c) · b_pq^enu · ŝ_s)
  ```

  built directly (a `(nbl, nchan, nsrc)` kernel summed over `s`); `nsrc ~ 100` keeps this tiny.
  Every source is unit flux; the optional `beam` is a precomputed `(nchan, nsrc)` real weight `B_s`
  (mirrors `dft_forward`'s `beam=`). Callers form it from `airy_power_beam(s_enu, (0,0,1), freqs)`
  — in ENU the antenna boresight is the zenith `(0,0,1)`, so `cos θ = sin(el)` and low-elevation
  sources are down-weighted. The module keeps its own `LIGHTSPEED` constant, decoupled from
  `healpix_dft`.

### 4.2 `utils/stefcal.py`

`stefcal_solve(vis, model, a1, a2, n_ant, *, t_int=None, ref_ant=0, weight=None, g0=None,
max_iter=100, tol=1e-8, xp=np) -> (gains: (n_sol, n_ant) complex, info: dict)`

- `vis`, `model`: `(ntime, nbl, nchan)` complex (a single `(nbl, nchan)` frame is accepted as
  `ntime=1`). `a1`, `a2`: `(nbl,)` int antenna indices.
- **Solution cadence.** `t_int` = consecutive integrations pooled into one gain solution; `None`
  pools all `ntime`, `1` solves every integration. Returns `n_sol = ceil(ntime / t_int)` solutions.
  Pooling sums the per-antenna reduction over time as well as baselines/channels — the model
  varies per integration (sources move) while one gain per antenna is solved over the interval.
- **Directed reduction.** Concatenate forward `(p=a1, q=a2, V, M)` and reverse
  `(p=a2, q=a1, conj V, conj M)`. For each directed row `z = M_dir · conj(g[partner])`, so
  `V_dir = g[p]·z`. Segment-sum over `p` (summing time, channels and optional `weight`):

  ```
  g_p ← Σ w·conj(z)·V_dir  /  Σ w·|z|²
  ```

  The denominator `Σ w·|z|²` is the per-antenna **scalar Hessian** (the joint Hessian is
  non-diagonal, but holding other gains fixed decouples it to one scalar per antenna), so the
  "inversion" is this division — no matrix solve.
- **Initialisation.** `g0` if given (warm start), else unity; each interval also warm-starts from
  the previous interval's solution. (The eventual operator passes its "current gain solutions".)
- **Convergence.** Even iterations average `g ← ½(g_new + g_old)` (StefCAL stabiliser); stop on
  max relative change `< tol` or `max_iter`. `info` carries `iterations`, `converged`, `max_change`,
  each an `(n_sol,)` array.
- **Gauge.** Per interval `g ← g / g[ref_ant]`, fixing `g_ref = 1` (global phase and amplitude
  together — the amplitude gauge is free because the model is unit-flux).

`referenced_phases(gains, ref_ant, *, xp=np) -> (..., n_ant)`: `angle(gains)` after the regauge
(equivalently `angle(g_p) − angle(g_ref)`); amplitudes discarded. Works on one solution or a
`(n_sol, n_ant)` stack.

**Edge cases** (mirroring existing `healpix_dft`/reader handling):

- Dead antennas (`gain==0`, baselines weight-0) drop out of the directed sums; their gain is
  `NaN` and never used.
- An antenna with no live baselines → denominator 0 → guarded (prior value / `NaN`), never
  `inf`/`nan`.
- Fully-flagged frame → `NaN` gains, `converged=False`; the caller coasts.
- `ref_ant` must be live — raise a clear `ValueError` otherwise.
- Optional per-baseline `weight` folds flags and (future) thermal weights through one path.

## 5. Testing

### 5.1 Primary — simulation round-trip (hermetic, NumPy; the gate)

In `tests/test_stefcal.py`:

1. A 24-antenna ENU layout (a fixture; real TART geometry or synthetic), `a1`/`a2` from its
   baselines, one antenna flagged (dead).
2. ~10–100 random source directions above the horizon → `M` via `model_visibilities`.
3. Random true gains `g_true` (amplitude ~1 ± spread, random phase).
4. Synthesise `V = g_true[a1] · M · conj(g_true[a2])` (+ optional small noise); zero dead
   baselines.
5. `stefcal_solve` → `g_hat`; regauge `g_true` and `g_hat` to `g_ref=1`.
6. Assert recovery up to gauge — tight (`~1e-6`) noiseless, looser with noise — and matching
   `referenced_phases`.

Plus focused cases: gauge invariance (a global complex scale on `g_true` leaves referenced
phases unchanged); flagged-antenna handling (dead → `NaN`, others unaffected); `ref_ant` dead
raises; `model_visibilities` against a single-source analytic fringe and a flux-1 superposition.

Plus a multi-integration pooled round-trip: gains constant, sources at a different position each
integration, pooled with `t_int = ntime` → recovers the gains; and `t_int` cadence shapes
(`n_sol = ceil(ntime / t_int)`).

### 5.2 Secondary — real-data sanity (opt-in, env-gated, excluded from required CI)

Per the testing rules, gated on `KREMETART_REALDATA=1` and off the required checks: solve the TART
HDF against the bundled catalogue cache using the best-matching config — **all sources,
beam-weighted, whole-file pooled** (`t_int = ntime`) — and compare `referenced_phases` to the TART
`gain_xds` snapshot. The comparison is **gauge-invariant** (remove the optimal global phase before
the RMS; referencing to a single antenna inflates the number by that antenna's own discrepancy).
Asserts the RMS is below a loose regression guard (30°; measured ~19°), not a target — per §2.1 the
snapshot is not a per-frame ground truth.

## 6. Dev workflow / conventions

- After every change: `uv run ruff format . && uv run ruff check . --fix`.
- Type hints on every signature; Google-style docstrings; heavy deps at module top in `utils/`
  (these are `utils/`, not `cli/`).
- No test artifacts in the repo tree; `tempfile.TemporaryDirectory()` for any I/O; the real-data
  check is opt-in.
- No new `cli/` command, cab, or `core/<cmd>.py` in this cut — it is library math under `utils/`,
  consumed later by the deferred calibration operator.

## 7. Open items deliberately left for later

- GPU (`xp=cupy`) validation of the same functions and the Holoscan operator wrapper.
- A higher-fidelity element-beam model in place of the Airy approximation — per §2.1 this is the
  main remaining lever on the ~19° residual vs TART's snapshot (the design doc's ongoing
  student-project beam).
- Flux solving / robustness / EKF tracking (still deferred per §2): the unit-flux + beam model is
  the acquisition stage only.
- Source selection: keep **all** catalogue sources and let the beam down-weight them — hard
  elevation floors were measured to make the agreement worse (§2.1), so there is no `el_min` knob.
```
