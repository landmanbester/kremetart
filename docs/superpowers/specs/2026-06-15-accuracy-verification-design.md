# Accuracy Verification (L1 + L2) — Design

**Date:** 2026-06-15
**Status:** Approved (brainstorming) — ready for implementation plan
**Scope:** Demonstrate, against an independent geodetic standard, that the kremetart geometry +
HEALPix imaging path recovers a known sky more accurately than the tart2ms geometry. Two layers:
**L1** (antenna positions vs an independent truth, no imaging) and **L2** (a sub-pixel point-source
simulation imaged with both geometries, measuring peak position and flux against truth). The
real-data visual movie + steady-source stability (**L3**) is a separate follow-up spec.

---

## 1. Purpose & the central principle

We claim our pipeline is *more accurate* than tart2ms. "More accurate" requires **ground truth**,
which real TART data lacks (the dominant sources are GNSS satellites with no local catalogue). So we
**simulate**: define a known sky, forward-model the visibilities, and measure each pipeline's
recovered sky against the known truth.

**No circularity.** The truth must come from a geometry that privileges *neither* pipeline:

- **Truth antenna ECEF** from **`pyproj` / PROJ** (surveying-grade WGS84), a code path used by
  neither our reader (hand-rolled `_geodetic_to_ecef`/`_enu_to_ecef`) nor tart2ms (spherical
  `astropy.offset_by` with a *mean* Earth radius).
- **Truth source directions** from `astropy` (ICRS, fixed celestial sky).
- The fringe model itself (`exp(2πi·delay)`) is unambiguous physics.

If our transform matches the PROJ truth (L1) and tart2ms's does not, then our imaging recovers the
source and tart2ms's is offset — a measurement against an external standard, not a self-assessment.
The same test would catch *us* being wrong, which is the point.

## 2. Geometry isolation

The established difference between the pipelines is the **antenna positions** (from earlier rephasing
work: `w` matched tart2ms to 1e-5 m; only `u/v` differed by ~1 cm via the mean-Earth-radius
`offset_by`). So we hold *everything else identical* and vary only the antenna ECEF:

- Same imager (`kremetart.utils.healpix_dft`), same frame rotation `C(t)`, same pixel grid, same
  truth visibilities.
- Three position sets: **truth** (pyproj), **ours** (`read_hdf_as_msv4` `ANTENNA_POSITION`),
  **tart2ms** (read from the reference MS `ANTENNA` table).

Because `disko` is *also* a gridless HEALPix DFT, re-imaging through the literal tart2ms→MS→disko
path would mostly reproduce this while confounding geometry with disko's imaging choices. Isolation
gives the clean statement: *with everything else identical, our positions put the source N arcmin
closer to truth.*

## 3. L1 — antenna-position truth comparison (no imaging)

1. Build **truth** antenna ECEF from the HDF ENU offsets via PROJ's topocentric↔geocentric
   conversion (`pyproj`, ellipsoid WGS84, origin at the site geodetic `lat/lon/alt` from the HDF
   config). This rotates *and* places the array independently of our code (the rotation is what
   matters for baselines, so a shared ENU-rotation shortcut is **not** acceptable — use PROJ end to
   end).
2. Read **ours** from `read_hdf_as_msv4(hdf).antenna_xds.ANTENNA_POSITION`.
3. Read **tart2ms** from the reference MS `ANTENNA` table (via `xarray-ms`).
4. Form baselines `b_ij = pos_i − pos_j` for each set and report, over all 276 baselines:
   per-baseline vector difference and baseline-length difference (max + RMS).

**Expected & asserted:** ours vs truth ≪ tart2ms vs truth; ours-vs-truth at the µm–sub-mm level;
tart2ms-vs-truth ~1 cm (max baseline-length difference).

## 4. L2 — sub-pixel point-source simulation

1. **Sky:** a general injector — N point sources at arbitrary continuous ICRS `(ra, dec)` with
   fluxes (Jy). The headline configuration places ~4 sources spanning **zenith angle** at the
   observation epoch (near-zenith → near-horizon), where the position error from a baseline error
   grows. Sources are deliberately **off pixel centres** (min angular distance to the nearest pixel
   centre asserted > a fraction of the pixel) so we exercise the true sub-pixel response.
2. **Truth visibilities (continuous, sub-pixel):**
   `V_pq(t) = Σ_s f_s · exp(2πi (ν/c) · b_pq^truth(t) · ŝ_s)`, where `b_pq^truth(t)` are the **pyproj**
   ECEF baselines rotated by the shared `C(t)` (`equatorial_baselines`) and `ŝ_s` is each source's
   ICRS unit vector. Mechanically this is `dft_forward` with the *source* unit vectors as directions
   and the fluxes as the "image". Noiseless.
3. **Recover** with the same imager (`image_frame`/`dirty_map`) twice — once with **our** ECEF, once
   with **tart2ms** ECEF — onto the same HEALPix grid (`nside` ≈ 256, chosen so the ~arcmin offset
   spans ~a pixel and the broad ~3° PSF is well oversampled, making peak-flux read-off clean).
4. **Metrics, per source:**
   - **Peak position offset (arcsec):** sub-pixel centroid in a neighbourhood of the recovered peak
     → recovered direction → angular separation from the true `ŝ_s`.
   - **Flux ratio (Jy/pixel):** recovered peak value (Jy, from the `1/Σw` normalisation) ÷ injected
     `f_s`.
   - **Analytic cross-check:** the position shift predicted by the per-baseline extra delay
     `(b^tart2ms − b^truth)·ŝ_s` (least-squares direction shift); assert the imaged tart2ms offset
     matches this prediction.

**Expected & asserted:** our offset ≈ 0 (≪ a pixel; set by L1 residual), tart2ms offset > ours and
≈ the analytic prediction, growing with zenith angle; our flux ratio ≈ 1.

## 5. Components & files

Verification code is **test-only** (pyproj goes in the test extra), reusing the shipped imager.

- `tests/verification/geometry_truth.py` — pyproj truth ENU→ECEF; source/`ŝ_s` setup from
  `(ra,dec)`; truth-visibility forward model; sub-pixel centroid + angular-offset + analytic-offset
  helpers. Pure functions, `numpy`/`astropy`/`pyproj`, no GPU.
- `tests/test_accuracy_l1_geometry.py` — L1 assertions.
- `tests/test_accuracy_l2_imaging.py` — L2 assertions.

Both test modules `importorskip("pyproj")`, `importorskip("xarray_ms")`, and skip if the reference
MS / HDF are absent (mirrors the existing rephasing tests).

## 6. Dependencies

Add `pyproj` (PROJ ≥ 6.3 for the topocentric conversion) to the **test** dependency group in
`pyproject.toml` (`[dependency-groups].test`). Not a runtime/`full` dependency.

## 7. Testing strategy

The "tests" *are* the verification: they assert quantitative superiority (our offset < tart2ms
offset, ours below a tight bound, tart2ms ≈ analytic prediction, flux ratio ≈ 1). They run on CPU
(`xp=np`). Reference data: the existing `tests/data/...nocal.ms` (tart2ms positions) and the test
HDF. A small fixed source set with a fixed seed keeps assertions deterministic.

## 8. Out of scope (this spec)

- **L3:** the real-data 9-frame visual movie and steady-source pixel-stability (separate spec).
- Installing/driving **disko** or the literal tart2ms→MS→disko imaging path.
- **Noise** / statistical robustness (the accuracy difference is deterministic geometry).
- Real-data **ground truth** via TLE/ephemeris fetching.
