# HEALPix DFT Operator — Design

**Date:** 2026-06-15
**Status:** Approved (brainstorming) — ready for implementation plan
**Scope:** Forward (image→visibilities) and adjoint (visibilities→image) gridless
DFT on a full-sky equatorial HEALPix grid, plus the dirty-map convenience used by
the streaming imager. Implements the imaging step of
`docs/tex/unified/kremetart_design.tex` §imaging (the dirty-map equation at the
`I_k(ŝ_j)` definition).

---

## 1. Purpose

Transform calibrated residual visibilities `R_pq` into a residual intensity image
on a fixed, sidereally-stable equatorial HEALPix grid, and provide the matched
forward model for prediction and (future) Tikhonov reconstruction. This is the
imaging operator referenced in the design doc: "a new HEALPix DFT operator,
structured after the existing tiled `(l,m)` operator", but with the equal-area
radiometry re-derived rather than ported.

The operator is a pathfinder for larger arrays and for eventual simultaneous
imaging of multiple TART stations (mosaicing), so it is GPU-resident from the
outset, even though TART's sizes are trivial.

---

## 2. Coordinate representation — direction cosines (decided)

Pixels are represented by their **Cartesian unit vectors** `ŝ_j = (x, y, z)`
(direction cosines), **not** colatitude/longitude.

Rationale:
- The dirty-map kernel is a 3-D dot product `b_pq · ŝ_j`; direction cosines *are*
  the components of `ŝ`. Angles would force a `sin/cos` conversion at every
  evaluation.
- `healpy.pix2vec(nside, ipix, nest=True)` returns Cartesian directly.
- The per-frame frame rotation `b_pq(t) = C(t)ᵀ b_enu` is a matmul on Cartesian
  vectors; angles would require `arctan2`/`arccos` round-trips.
- On the **equal-area** sphere there is **no `(n−1)` reference term and no `1/n`
  Jacobian** (unlike the tangent-plane `(l,m)` operator). The full 3-vector
  `(l,m,n)` enters a bare geometric dot product. This is the key structural
  simplification over `dft_lm.py`.

Colatitude/longitude is retained only as an optional diagnostic/IO convenience.
HEALPix ordering is **NESTED** (index locality), per the design doc.

---

## 3. Architecture

Mirrors the `rephasing.py` pattern: an `xp`-injectable pure core for CPU
testability, with a thin Holoscan operator binding `xp=cp` for the GPU pipeline.

### 3.1 `src/kremetart/utils/healpix_dft.py` — pure, `xp`-injectable

No top-level `cupy`/`healpy`. `healpy` is imported lazily inside grid setup
(host-only, called once); `xp` is injected into the hot path. In production the
operator passes `xp=cp` and everything stays on device between read and write;
tests pass `xp=np`.

```python
def make_pixel_grid(nside: int, *, nest: bool = True, xp=np) -> "xp.ndarray":
    """(npix, 3) array of HEALPix pixel unit vectors via pix2vec. npix = 12*nside**2."""

def dft_forward(image, baselines, pix_vec, freqs, *, xp=np) -> "xp.ndarray":
    """Image -> visibilities. phasesign = +1.
    image:      (npix,) real (or complex, for the adjoint test)
    baselines:  (nbl, 3) equatorial-rotated baselines b_pq(t), metres
    pix_vec:    (npix, 3) unit vectors
    freqs:      (nchan,) Hz
    returns:    vis (nbl, nchan) complex
    """

def dft_adjoint(vis, baselines, pix_vec, freqs, *, xp=np) -> "xp.ndarray":
    """Visibilities -> image. phasesign = -1 (exact Hermitian transpose of dft_forward).
    vis:        (nbl, nchan) complex
    returns:    image (npix,) complex   # caller takes Re / normalizes
    """

def dirty_map(vis, weights, baselines, pix_vec, freqs, *, xp=np) -> "xp.ndarray":
    """Convenience: Re{ dft_adjoint(weights * vis) } / sum(weights). (npix,) real.
    Implements the design-doc dirty-map equation directly."""
```

### 3.2 Frame rotation helper (host, swappable backend)

`C(t)` maps equatorial unit vectors to the topocentric ENU frame; the imager
needs `b_rot = b_enu @ C(t)ᵀ`. This is O(n_time) host bookkeeping (same split as
`rephase_to_dir`: tiny per-frame host work, big per-baseline/pixel device work).

```python
def equatorial_baselines(enu_baselines, time, site, *, backend="astropy") -> ndarray:
    """ENU baselines (nbl,3) -> equatorial-rotated b_pq(t) per timestamp.
    backend="astropy": transform the ENU triad into the equatorial frame
                       (reuses rephasing.py); the test oracle, ships first.
    backend="native":  GMST/ERA polynomial + R_tilt·R_z(LST), GPU-friendly,
                       validated against the astropy oracle to << pixel. (later)
    """
```

`C(t)` decomposes as a constant latitude tilt ∘ rotation about the polar axis by
LST. The native backend computes `LST(t)` from a GMST/ERA polynomial (linear in
time) and builds the 3×3 from `sin/cos` of LST and latitude. We **drop**
precession/nutation, polar motion, UT1−UTC and diurnal aberration (all ≪ the
~0.9° pixel at nside=64 over a streaming session). The backend is selectable
behind one interface; we ship `astropy` first as the oracle, develop `native`
against it.

### 3.3 `src/kremetart/operators/dft_healpix.py` — Holoscan operator

Thin wrapper binding `xp=cp`, following the `dft_lm.py` I/O contract.

- **Inputs:** `VISIBILITY` (unstopped residual `R_pq`), `WEIGHT`, ENU baselines
  (or ENU antenna positions to difference), `FREQ`, `time`.
- **Steps:** form `b_rot` via `equatorial_baselines(...)` (host) → push to device
  → `dirty_map(...)` on device.
- **Outputs:** `cube` (HEALPix map + placeholder axes), `time_out`, `freq_out`.

---

## 4. Math & conventions

Geometric delay (metres): `G_{pq,j} = b_pq · ŝ_j`, with `b_pq(t) = C(t)ᵀ b_enu`.

- **Forward** (model vis): `V_pq = Σ_j I_j exp(+2πi (ν/c) G_{pq,j})`.
- **Adjoint:** `Î_j = Σ_pq exp(−2πi (ν/c) G_{pq,j}) V_pq = Σ_pq conj(forward kernel) V_pq`.
- **Dirty map:** `I_j = (1/Σw) Re{ Σ_pq w_pq R_pq exp(−2πi (ν/c) G_{pq,j}) }
  = (1/Σw) Re{ dft_adjoint(w ⊙ R)_j }`.

No `(n−1)` w-term, no `1/n` Jacobian (equal-area grid). Implementation forms
`G = baselines @ pix_vecᵀ` once, then `exp(±2πi (ν/c) G)` per channel.

---

## 5. Input contract & relationship to `rephase_to_dir`

The imager consumes **unstopped** residual visibilities. The bare-delay kernel
matches the raw, fringe-unstopped topocentric phase of TART data:
`b_pq(t)·ŝ_jᵉᑫ = b_enu·ŝ_jᵗᵒᵖᵒ(t)`. Feeding fringe-stopped (rephased)
visibilities would reintroduce a `(ŝ_j − ŝ_0)` phase-centre term and the
tangent-plane `(n−1)` factor the equal-area design deliberately drops.

So the two are distinct, complementary tools:
- **`rephase_to_dir`** — fringe-stops to one phase centre, standard UVW → MSv4
  export / tart2ms validation. **Not** wired into imaging.
- **HEALPix DFT** — no fringe stopping, rotates baselines, bare delay over the
  full sphere → imaging/detection.

---

## 6. Polarisation & layout

TART is single-pol RHCP: one correlation per baseline, scalar residual `R_pq`.
The imager images the **scalar** residual directly — no Stokes machinery
(`s2c`/`c2s`, `stokes_expr_cupy`). A **trailing length-1 correlation axis** is
retained in the `cube` layout for forward-compatibility.

---

## 7. Memory & performance

At nside=64: `npix = 49152`, `nbl = 276`, `nchan = 1`. The complex kernel
`(nbl, npix)` ≈ 108 MB; forward/adjoint are single matmuls. **Plain, non-tiled
matmul** to start. Pixel-tiling (mirroring `dft_lm_tiled.py`'s spatial-tile +
channel-batch logic) is a deferred scaling extension, added when `npix·nbl`
exceeds the device budget (higher nside, more baselines, or multi-station
mosaicing). The API is designed so adding tiling requires no signature change.

---

## 8. Testing

CPU tests run with `xp=np` (no GPU in CI).

1. **Adjointness (dot-product) test — primary correctness guard.** With complex
   random `image` and `data`:
   `⟨forward(image), data⟩ == ⟨image, adjoint(data)⟩`
   (inner product `Σ conj(a)·b`). Holds to machine precision because forward
   (+1) and adjoint (−1) are exact Hermitian transposes. Definitive check on the
   `ν/c` / phase-sign convention.
2. **Grid sanity:** `make_pixel_grid` returns `12*nside²` unit vectors (norm ≈ 1),
   NESTED ordering.
3. **Dirty-map point-source:** forward a single-pixel unit source → `dirty_map`
   of that model peaks at the right pixel.
4. **`C(t)` consistency:** `equatorial_baselines(backend="native")` matches
   `backend="astropy")` to ≪ pixel (target < 0.05°) — added when the native
   backend lands.
5. **`xp` parity (GPU-gated):** numpy vs cupy results agree — opt-in, excluded
   from required CI.

---

## 9. Future extensions (out of scope this phase)

- **GPU-native `C(t)`** (`backend="native"`), validated against the astropy oracle.
- **Pixel-tiled** forward/adjoint for scaling.
- **Tikhonov map:** `Î = (AᴴWA + μΓ)⁻¹ AᴴWR` via CG, reusing `dft_forward`/
  `dft_adjoint` as `A`/`Aᴴ`. The forward/adjoint pair is built now precisely so
  this drops in later.
- **Primary beam + multi-station mosaicing:** per-pixel beam weight and stitching
  of all TART stations onto the common equatorial grid. Requires a primary-beam
  model (separate student project); imaging path designed to accommodate it.
- **Above-horizon gather** as a compute optimisation over the full allocated grid
  (`ŝ_j · ẑᵉᑫ(t)`), not a science mask.
