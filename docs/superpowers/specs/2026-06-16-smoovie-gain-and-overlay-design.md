# `smoovie` — Gain Correction & Satellite-Track Overlay — Design

**Date:** 2026-06-16
**Status:** Approved (brainstorming) — ready for implementation plan
**Scope:** Add two opt-in capabilities to `kremetart smoovie` so we can *verify* the common-ICRS-frame
imaging: (1) **`--correct-gains`** applies the inverse of TART's per-file gain snapshot to the
visibilities before imaging, so frames show calibrated data; (2) **`--overlay-catalog`** fetches the
TART satellite catalogue and overlays each visible satellite's expected position (marker + trailing
track + name label) on every frame. Builds on
`docs/superpowers/specs/2026-06-16-smoovie-common-frame-design.md`; all sections of that spec still
hold.

---

## 1. Motivation

The common-frame rework images every sub-integration onto a fixed equatorial (ICRS) HEALPix grid
centred on a shared phase direction. We now need to confirm it is *correct*: that real sources land
where they should. Two pieces are missing.

1. **Uncorrected data.** `smoovie` images raw TART visibilities `V_pq = g_p · g_q* · V_pq^true`. The
   per-antenna gain errors smear and shift power, so sources do not sit at their true sky positions.
   Applying the inverse gain solution recovers calibrated data in which point sources (satellites)
   are sharp and correctly placed.
2. **No ground truth on the frame.** Even with corrected data, "correct placement" is unverifiable
   without an independent prediction. The TART catalogue gives each visible satellite's `(az, el)`
   per timestamp; overlaying those positions (converted to ICRS) on each frame lets us read off,
   by eye, whether imaged sources coincide with their predicted positions.

Both are **opt-in flags, default off**, so the existing hermetic, uncorrected behaviour is unchanged.

## 2. Key decisions

- **Gains = TART's own per-file snapshot.** Each HDF carries one gain solution per antenna,
  exposed by `read_hdf_as_msv4` as `gain_xds.GAIN` (`g = amp · e^{iφ}`, complex64), indexed by
  `antenna_name`. This is the only gain information available and is time-independent within a file.
- **Correction convention:** `V_pq^corr = V_pq / (g_p · conj(g_q))`, the inverse of the measurement
  equation `V_pq = g_p · g_q* · V_pq^true`, consistent with the `b = pos(ant1) − pos(ant2)` baseline
  sign convention used throughout the pipeline. **Weights are scaled by `|g_p · g_q|²`** (the
  consistent inverse-variance transform). Dead antennas (`g = 0`) are already weight-flagged; the
  correction guards against division by zero and leaves them at zero weight.
- **Catalogue source = TART API (network).** Reuse `tart_tools.api_handler.catalog_url(lon, lat,
  datestr=t)` — the exact path `read_hdf_as_xr` already uses — but keep **all** sources above the
  elevation cutoff, not just the brightest. The MSv4 reader stays hermetic; catalogue fetching is a
  separate, clearly network-dependent step invoked only when `--overlay-catalog` is set.
- **Coordinate consistency.** Satellite `(az, el)` is converted to ICRS `(ra, dec)` at each frame
  timestamp via `AltAz(...).icrs` with the site `EarthLocation` — the same astropy path as
  `rephasing._instantaneous_zenith`, so the overlay lives in the same ICRS frame that
  `make_pixel_grid` declares for the HEALPix grid. Any residual offset between an imaged source and
  its overlaid marker is precisely the discrepancy this verification exists to expose.

## 3. CLI signature additions (`cli/smoovie.py`)

Three new parameters, following `python-standards.md` §3. All are real cab inputs (no
`StimelaMeta(skip=True)`), threaded through the preflight dict, the `run_in_container` dict, and
`core.smoovie`. Regenerate the wrapper via `hip-cargo generate-function` so the round-trip stays
byte-stable.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `correct_gains` | `bool` | `False` | Apply inverse gains (`gain_xds.GAIN`) before imaging. |
| `overlay_catalog` | `bool` | `False` | Overlay TART catalogue satellite tracks. Requires network. |
| `catalog_elevation_deg` | `float` | `45.0` | Elevation cutoff for catalogue sources to overlay (matches `read_hdf_as_xr`'s `filter_elevation`). |

All existing parameters are unchanged.

## 4. Change 1 — gain correction

### 4.1 `utils/gains.py` (new)

A small, reusable, host/`xp`-agnostic helper — keeps the correction math out of `core/smoovie` and
makes it directly unit-testable, mirroring the utils-heavy layout.

```python
def apply_inverse_gains(vis, weight, gains, a1_idx, a2_idx, *, xp=np):
    """Correct vis/weight by the inverse per-antenna gain product.

    factor = gains[a1_idx] * conj(gains[a2_idx])     # (nbl,)
    ok     = abs(factor) > 0                          # dead antennas -> factor == 0
    vis_c  = where(ok, vis / factor, 0)
    wgt_c  = where(ok, weight * abs(factor)**2, 0)    # inverse-variance scaling
    return vis_c, wgt_c
```

- `gains`: `(n_ant,)` complex, ordered by `antenna_name` (straight from `gain_xds.GAIN`).
- `a1_idx`, `a2_idx`: `(nbl,)` integer antenna indices per baseline.
- Broadcasts cleanly over a leading time axis and a trailing channel axis.

### 4.2 `core/smoovie.frame_dirty_maps(..., correct_gains=False)` — amended

When `correct_gains` is true, per file (gains are per-file, applied once before the sub-integration
loop):

1. Read `gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values`.
2. Build `a1_idx`/`a2_idx` from `baseline_antenna1_name`/`baseline_antenna2_name` against
   `antenna_xds.antenna_name` — the **same mapping `itrs_baselines` already uses**.
3. `vis, wgt = apply_inverse_gains(vis, wgt, gains, a1_idx, a2_idx)`.

Then image each sub-integration as today. `image_frame` and the DFT math are untouched.

### 4.3 Validation oracle

The test data ships `vis_2026-06-09_08_11_43.476804.ms` (calibrated) alongside
`…_nocal.ms` (raw). `tart2ms` produces the calibrated MS by applying the same gain solution, so the
corrected visibilities should match the calibrated `.ms` DATA column for matching baselines — a
strong oracle for an opt-in test.

## 5. Change 2 — satellite overlay

### 5.1 `utils/satellites.py` (new)

Network + astropy only; no healpy/matplotlib (rendering stays in `core/smoovie`).

```python
def satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch):
    """Per-satellite ICRS tracks aligned 1:1 with the smoovie frame sequence.

    Returns dict[name] -> list[(frame_index, ra_deg, dec_deg, flux_jy)].
    """
```

- **Frame alignment invariant.** Iterates the *same* ordering as `frame_dirty_maps`
  (`for path in hdf_paths: for k in range(n_time)`), so global frame index `i` produced here matches
  dirty-map index `i` exactly. Documented and asserted in tests.
- **Fetch.** Per frame timestamp `t`: `api.catalog_url(lon, lat, datestr=t) + f"&elevation={elevation_deg}"`
  → list of sources, each `{name, az, el, jy, r}`. Reuses `tart_tools.api_handler` (same path as
  `read_hdf_as_xr`) but keeps **all** returned sources. `fetch` is an injectable callable so tests
  supply a canned catalogue with no network.
- **Conversion.** Per frame, batch-convert that frame's source `(az, el)` to ICRS `(ra, dec)` via
  `SkyCoord(AltAz(az, alt=el, obstime=Time(t), location=loc)).icrs`, accumulating per named
  satellite.

### 5.2 `core/smoovie.render_frames(..., tracks=None)` — amended

After `hp.mollview(..., rot=rot)` and `hp.graticule()`, when `tracks` is provided, for each satellite
visible in frame `i`:

- `hp.projscatter(ra, dec, lonlat=True, …)` — marker at the current-frame position.
- `hp.projplot(trail_ra, trail_dec, lonlat=True, …)` — faint trailing line of that satellite's
  positions from frames `≤ i` (the track so far).
- `hp.projtext(ra, dec, name, lonlat=True, …)` — name label at the current position.

`lonlat=True` takes degrees with `lon = RA`; healpy applies the active `rot`, so markers land in the
same projected ICRS frame as the imaged pixels.

### 5.3 `core/smoovie.smoovie(...)` — amended wiring

```
maps, stamps, _ = frame_dirty_maps(hdf_paths, nside, correct_gains=correct_gains)
tracks = satellite_tracks(hdf_paths, catalog_elevation_deg) if overlay_catalog else None
render_frames(maps, stamps, nside, cmap, td, rot=(phase_ra_deg, phase_dec_deg), tracks=tracks)
encode_movie(...)
```

## 6. Cost & container notes

- **Network cadence.** One catalogue query per sub-integration (hundreds of calls for a full
  multi-file run) — the same cadence `read_hdf_as_xr` already uses. No subsampling is added now
  (YAGNI); a `--catalog-stride` (query every k-th frame, interpolate) is a clean future addition if
  it proves too slow.
- **Container path.** `--overlay-catalog` needs `tart_tools` (already a `[full]` dep, imported by
  `read_tart_hdf`) and outbound network. Native runs need `[full]`; the container image already has
  the dep and is expected to allow network for the catalogue fetch.

## 7. Testing

- **`tests/test_gains.py` (new):** `apply_inverse_gains` on synthetic gains — correction formula,
  `|g|²` weight scaling, and the dead-antenna (`g=0`) guard (no `inf/nan`, weight stays 0).
- **`tests/test_smoovie_core.py`:** add `frame_dirty_maps(correct_gains=True)` finite-map smoke
  test; add an **opt-in** (env-gated) test asserting corrected vis match the calibrated `.ms` DATA
  for matching baselines.
- **`tests/test_satellites.py` (new):** with an injected fake `fetch` (no network) — frame-index
  alignment against the `frame_dirty_maps` ordering, and `(ra, dec)` matching an independent
  `AltAz→ICRS` computation within tolerance. A real-network end-to-end test is gated on an env var
  and excluded from required CI (per `testing-and-ci.md` §2).
- **`tests/test_smoovie_core.py` (render):** `render_frames(tracks=…)` at small `nside` with a canned
  track produces the PNG sequence; monkeypatch `hp.projscatter`/`projplot`/`projtext` to assert call
  counts per frame.
- **Round-trip (mandatory):** `tests/test_roundtrip.py::test_roundtrip_smoovie` regenerated for the
  three new CLI params; must stay byte-identical after `ruff format`.

## 8. Out of scope

- Multi-TART mosaicking and primary-beam weighting (unchanged from the common-frame spec).
- Below-horizon masking (full sphere still rendered).
- Catalogue subsampling/interpolation (`--catalog-stride`).
- Native (non-astropy) az/el→ICRS conversion; GPU rendering.
- Solving for gains — we only *apply* TART's existing solution.
