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

1. A sky-model coherency util that builds unit-flux model visibilities over the ~100 visible
   catalogue sources in the ENU frame.
2. A StEFCal solver util (alternating per-antenna complex LS) with gauge fixing and
   phase extraction.
3. Catalogue glue that yields per-frame `(az, el, name)` arrays aligned to the vis frames,
   with an injectable `fetch` for hermetic tests.
4. Tests: a simulation round-trip as the correctness gate, plus an opt-in real-data sanity
   check against TART's own gain snapshot.

**Explicitly deferred** (recorded so the core stays forward-compatible, not built now):

- The Holoscan `operators/calibration.py` + `core` driver that slots between reader and imager.
- Moving the equatorial rotation downstream of calibration in the prepared-zarr path
  (`prepare_msv4_zarr` currently rotates to equatorial *before* writing).
- Flux solving, the Student-t robust/IRLS weighting, the IWP-EKF tracking, fixed-lag
  smoothing, model subtraction, and reference-antenna failover/re-gauging.

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
  `ŝ = (sin az · cos el, cos az · cos el, sin el)`. This convention is self-consistent for the
  simulation test; it is reconciled with the correlator phase / reader `b = pos(a1)−pos(a2)`
  sign during the §5.2 real-data check.
- `model_visibilities(s_enu, bl_enu, freqs, *, xp=np) -> (nbl, nchan)`: unit-flux coherency

  ```
  M_pq = Σ_s exp(2πi (ν/c) · b_pq^enu · ŝ_s)
  ```

  built directly (a `(nbl, nchan, nsrc)` kernel summed over `s`); `nsrc ~ 100` keeps this tiny.
  The module keeps its own `LIGHTSPEED` constant so it stays decoupled from `healpix_dft`.

### 4.2 `utils/stefcal.py`

`stefcal_solve(vis, model, a1, a2, n_ant, *, ref_ant=0, weight=None, g0=None, max_iter=50,
tol=1e-6, xp=np) -> (gains: (n_ant,) complex, info: dict)`

- `vis`, `model`: `(nbl, nchan)` complex. `a1`, `a2`: `(nbl,)` int antenna indices.
- **Directed reduction.** Concatenate forward `(p=a1, q=a2, V, M)` and reverse
  `(p=a2, q=a1, conj V, conj M)`. For each directed row `z = M_dir · conj(g[partner])`, so
  `V_dir = g[p]·z`. Segment-sum over `p` (summing channels and optional `weight`):

  ```
  g_p ← Σ w·conj(z)·V_dir  /  Σ w·|z|²
  ```

- **Initialisation.** `g0` if given (warm start), else unity. (The eventual operator passes its
  "current gain solutions"; the core defaults to cold start.)
- **Convergence.** Even iterations average `g ← ½(g_new + g_old)` (StefCAL stabiliser); stop on
  max relative change `< tol` or `max_iter`. `info` carries `iterations`, `converged`, and the
  per-iteration max change.
- **Gauge.** After convergence `g ← g / g[ref_ant]`, fixing `g_ref = 1` (global phase and
  amplitude together — the amplitude gauge is free because the model is unit-flux).

`referenced_phases(gains, ref_ant, *, xp=np) -> (n_ant,)`: `angle(gains)` after the regauge
(equivalently `angle(g_p) − angle(g_ref)`); amplitudes discarded.

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

### 5.2 Secondary — real-data sanity (opt-in, env-gated, excluded from required CI)

Per the testing rules, gated on an env var and off the required checks: solve a sample TART HDF
frame against a cached catalogue and compare `referenced_phases` to the TART `gain_xds` snapshot
phases (re-referenced to the same antenna) within a tolerance. This pins the
`b = pos(a1)−pos(a2)` sign and the az convention against TART's own solution.

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
- Reconciling the ENU az / baseline-sign convention with the correlator (the §5.2 check is the
  first probe).
- Source selection policy (elevation cutoff for which catalogue sources enter the model) — a
  parameter with a sensible default; the design doc drops a hard `el_min` in *imaging* but the
  *model* still chooses which sources to include.
```
