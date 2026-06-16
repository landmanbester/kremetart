# `smoovie` — Common Phase Direction & Per-Slice Frames — Design

**Date:** 2026-06-16
**Status:** Approved (brainstorming) — ready for implementation plan
**Scope:** Rework `kremetart smoovie` so that (1) **every** sub-integration becomes a movie frame
(not one mid-slice per file), and (2) all frames are imaged into a single **common ICRS frame**
anchored by an explicit, reusable **common phase direction** computed from the global mid-time.
Amends `docs/superpowers/specs/2026-06-15-smoovie-design.md` (§4 data flow, §3 CLI signature, §5–6
tests); all other sections of that spec still hold.

---

## 1. Motivation

The current `frame_dirty_maps` images only the **mid sub-integration of each HDF file** (≈9 frames),
each rotated by *that file's own* `C(t)`. Two problems:

1. **Too few frames.** One frame per 1-minute file gives a choppy, near-static movie. We want a
   frame per sub-integration so motion (satellites, sidereal drift) is visible.
2. **No shared phase direction.** Nothing pins the projection to a common reference, so the observed
   patch is not consistently centered, and there is no mechanism to give *multiple TARTs* a common
   field center for the eventual full-sky **mosaic**.

## 2. Key decision — common frame is ICRS, anchored by a shared phase direction

The eventual goal is to mosaic **all** TARTs onto one full-sky map. TARTs at different sites have
different zeniths/horizons, so a **topocentric frame is site-specific and cannot be mosaicked**. The
only frame all TARTs share is the celestial one. Therefore:

- **Common frame = ICRS.** The HEALPix grid already lives there (`make_pixel_grid` declares ICRS).
- **Per-time rotation `C(t)` is kept** — `b_icrs(t) = R(t)ᵀ · b_itrs`. This is precisely the
  mechanism that registers every snapshot (and later every TART) onto the shared celestial grid: a
  fixed celestial source holds one pixel; for a single site the local zenith drifts only
  ~0.25°/min (~2.25° over a 9-minute run). **The imaging math in `image_frame` does not change.**
- **Common phase direction** = a *single shared ICRS direction* `(ra_deg, dec_deg)` that every frame
  — and eventually every TART — agrees on. It is the **zenith RA/Dec at the global mid-time** over
  all input files. Its roles:
  1. **Orient the all-sky projection** (`hp.mollview(rot=(ra_deg, dec_deg))`) so the observed patch
     sits stably at the center across the whole movie.
  2. **Be the reusable, overridable field center** handed to each TART when mosaicking.

  It does **not** modify the DFT: the gridless full-sky DFT needs no phase-center subtraction, and
  mosaicking is just summing weighted dirty maps on the same global grid. The phase direction is
  orientation + shared-field metadata only.

## 3. Components (`core/smoovie.py`)

### 3.1 `common_phase_direction(hdf_paths) -> tuple[float, float]`

New helper. Computes the single shared ICRS phase direction:

1. Read the first and last timestamps across **all** `hdf_paths` (open each partition, read
   `ds.time.values`); take the global midpoint `t_mid = 0.5 * (t_first + t_last)`.
2. Read the site `(lat, lon, alt)` from `ds.attrs["observation_info"]`
   (`site_latitude_deg` / `site_longitude_deg` / `site_altitude_m`).
3. `AltAz(az=0°, alt=90°, obstime=Time(t_mid, format="unix", scale="utc"),
   location=EarthLocation(lat, lon, height=alt)).transform_to(ICRS())` → `(ra_deg, dec_deg)`.

Deterministic, host-side, O(n_files). Reused unchanged by future multi-TART code (call once, pass
the same value to each TART).

### 3.2 `frame_dirty_maps(hdf_paths, nside, *, xp=np)` — reworked

Iterate **every sub-integration** across all files:

```
pix_vec = make_pixel_grid(nside, xp=xp)
maps, stamps = [], []
for path in hdf_paths:
    node  = _partition(read_hdf_as_msv4(path))
    times = node.ds.time.values            # (n_time,)
    bl    = itrs_baselines(node, xp)       # (nbl, 3)
    freqs = node.ds.frequency.values
    vis   = node.ds.VISIBILITY.values[..., 0]   # (n_time, nbl, nchan), drop pol
    wgt   = node.ds.WEIGHT.values[..., 0]
    for k in range(times.size):
        dmap = image_frame(vis[k:k+1], wgt[k:k+1], times[k:k+1], bl, pix_vec, freqs, xp=xp)
        maps.append(np.asarray(dmap)); stamps.append(_utc(times[k]))
return maps, stamps, pix_vec
```

One frame per sub-integration; existing `image_frame` (per-time `C(t)`, ICRS grid) unchanged.
Per-frame memory stays small (`nrow = nbl`, single time slice).

### 3.3 `render_frames(..., rot=(ra_deg, dec_deg))` — amended

Pass `rot=(ra_deg, dec_deg)` to `hp.mollview` so every frame is centered on the common phase
direction. Fixed `vmin/vmax` over all maps and per-frame UTC titles are unchanged.

### 3.4 `smoovie(...)` — amended wiring

```
hdf_paths = sorted(hdf_dir.glob("*.hdf"))
if phase_ra_deg is None or phase_dec_deg is None:
    phase_ra_deg, phase_dec_deg = common_phase_direction(hdf_paths)
maps, stamps, _ = frame_dirty_maps(hdf_paths, nside)
render_frames(maps, stamps, nside, cmap, td, rot=(phase_ra_deg, phase_dec_deg))
encode_movie(...)
```

If exactly one of `phase_ra_deg`/`phase_dec_deg` is supplied, raise `ValueError` (both or neither).

## 4. CLI signature additions (`cli/smoovie.py`)

Two new optional parameters (the override hooks for the mosaic path), following
`python-standards.md` §3 "Optional None" pattern:

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `phase_ra_deg` | `float \| None` | `None` | Common phase-direction RA (deg, ICRS). `None` → auto from global mid-time zenith. |
| `phase_dec_deg` | `float \| None` | `None` | Common phase-direction Dec (deg, ICRS). `None` → auto. |

All other parameters (`hdf_dir`, `movie`, `nside`, `fps`, `cmap`, `backend`,
`always_pull_images`) are unchanged. The wrapper keeps the standard lazy-import + container-fallback
shape; regenerate via `hip-cargo generate-function` so the round-trip stays byte-stable.

## 5. Testing

- **`tests/test_smoovie_core.py`** — update `test_frame_dirty_maps_*` to assert the frame count
  equals the **total sub-integration count** across the test files (sum of each file's `time` size),
  and that every map is finite. The ffmpeg smoke test is unchanged (skips without ffmpeg) but now
  exercises the many-frame path at small `nside`.
- **`tests/test_smoovie_stability.py`** — unchanged assertion (a fixed-ICRS source still holds one
  pixel; the per-time `C(t)` math is untouched).
- **New `common_phase_direction` test** (in `test_smoovie_core.py`): (a) determinism — same inputs
  give the same `(ra, dec)`; (b) the auto value equals an independent astropy AltAz→ICRS computation
  at the global mid-time within a small tolerance; (c) `smoovie` honors an explicit override (assert
  the override flows to `render_frames`, e.g. via monkeypatch capturing `rot`).
- **Round-trip (mandatory):** `tests/test_roundtrip.py::test_roundtrip_smoovie` already exists; it
  must still pass with the two new CLI params (regenerate the cab via the pre-commit hook).

## 6. Performance note

~540 frames at `nside=128` on CPU (`xp=np`) is slow (minutes) but acceptable for a one-off; the
core stays `xp`-injectable so the GPU pipeline is unaffected, and tests run at small `nside`.

## 7. Out of scope (unchanged from the original smoovie spec)

- Actual multi-TART mosaicking and primary-beam weighting (this design only adds the *shared phase
  direction mechanism* that mosaicking will consume).
- Below-horizon masking (full sphere is rendered).
- Native GPU rendering of the batch tool; tart2ms side-by-side comparison.
