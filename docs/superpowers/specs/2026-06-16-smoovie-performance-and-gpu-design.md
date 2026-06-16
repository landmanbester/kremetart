# `smoovie` — Performance: Catalog Caching, Profiling & a GPU Holoscan Path — Design

**Date:** 2026-06-16
**Status:** Approved (brainstorming) — ready for implementation plan (Phase 1 first)
**Scope:** Make `kremetart smoovie` fast and re-runnable. **Phase 1** (implement now, CPU path): cache
the TART catalogue to a sidecar so re-runs are network-free, and add a profiling harness to locate the
real bottlenecks. **Phase 2** (architecture captured here; its implementation plan written *after*
Phase 1 profiling): restructure `smoovie` into a Holoscan GPU application mirroring
`core/stream_msv4.py`, with a pure-GPU `HealpixDFTOperator`, all host work in a prepare step + reader,
an xarray/zarr I/O pattern, and movie encoding after `app.run()`.

Builds on the two prior smoovie specs (`2026-06-16-smoovie-common-frame-design.md`,
`2026-06-16-smoovie-gain-and-overlay-design.md`); all of their behaviour still holds.

---

## 1. Motivation

`kremetart smoovie --hdf-dir tests/data --movie /tmp/tart.mp4 --fps 4 --correct-gains
--overlay-catalog` is "incredibly slow". The default `nside=128` images **every** sub-integration
(9 files × 60 = ~540 frames). The prime suspect is the imaging DFT: `image_frame` materialises an
`exp(i·phase)` kernel of shape `(nbl=276, nchan=1, npix≈196k)` — ~870 MB of complex128 **per frame, on
CPU, ×540**. Secondary costs: ~540 sequential catalogue API calls (network), ~540 astropy `C(t)`
transforms, and ~540 matplotlib `mollview` renders. We need (a) to stop paying the network cost on
every run and (b) hard numbers on where time actually goes, before investing in the GPU rewrite.

The longer-term direction is fixed: `smoovie` should become a Holoscan app like `stream_msv4`, with the
`HealpixDFTOperator` (`operators/dft_healpix.py`) running **completely on the GPU** and all host work
(reading, gain correction, `C(t)`) confined to the reader / a prepare step, using the same xarray/zarr
pattern. Movie creation (the ffmpeg call) happens **outside** `app.run()` in the core implementation.

## 2. Key decisions (settled during brainstorming)

1. **HDF → GPU via a host "prepare" step → MSv4 zarr.** A host preprocessing step converts the HDFs
   into one MSv4 zarr, applying gain correction and precomputing per-frame rotated baselines `b_rot(t)`.
   The (lightly adapted) zarr reader streams it to the GPU. Isolates all host/astropy/gain work; the
   zarr is reusable across runs; the GPU operator stays pure.
2. **`C(t)` precomputed on the host; `b_rot(t)` streamed.** `b_rot(t) = R(t)ᵀ · b_itrs` is computed
   with astropy (vectorised over all frames) in the prepare step, so `HealpixDFTOperator.compute()` is
   pure cupy (just the DFT). Keeps current astropy accuracy; no GPU-native `C(t)` polynomial.
3. **Catalogue cache = sidecar file** keyed by `(lat, lon, elevation_deg, datestr)`. Separate from the
   zarr, inspectable, portable.
4. **Sequencing: cache + profile first**, then finalise Phase 2 depth from the measured numbers.

## 3. Phase 1 — caching + profiling (CPU path, implement now)

### 3.1 Catalogue cache (`utils/satellites.py`)

`satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch, cache_path=None)` gains an
optional `cache_path`:

- The cache is a JSON object mapping a composite string key
  `f"{lat:.6f}:{lon:.6f}:{elevation_deg}:{datestr}"` → the catalogue source list (the list of
  `{name, az, el, jy, r}` dicts the API returns).
- On entry, load `cache_path` if it exists. Per frame: key present → use cached sources (no fetch);
  absent → call `fetch(...)`, store under the key. Write the (updated) cache back to `cache_path` at the
  end if anything changed.
- The injectable `fetch` seam is unchanged, so tests still run without the network. A cache hit must
  **not** call `fetch` (verified by call-count in tests).
- JSON (not parquet): the payload is small lists of dicts; JSON is simple and inspectable.

`core/smoovie.smoovie` gains `catalog_cache: str | None = None`; when `overlay_catalog` is set it passes
`cache_path` (defaulting to `<movie>.catalog.json` when the option is unset) into `satellite_tracks`.

### 3.2 Profiling harness (`core/smoovie.py`)

- A tiny stage-timer (context manager) records wall-clock for each major stage:
  `common_phase_direction`, `frame_dirty_maps` (imaging), `satellite_tracks` (catalogue),
  `render_frames`, `encode_movie`. On completion it prints a summary table: per-stage seconds, % of
  total, and per-frame ms where applicable.
- Gated by a `profile: bool = False` parameter (CLI `--profile`). A `nframes: int | None = None`
  parameter (CLI `--nframes`) caps the number of frames imaged/rendered so a representative slice can be
  measured quickly instead of all ~540. `nframes` limits the frame loop only; it is a profiling/preview
  aid, documented as such.
- The harness writes to stdout (and is a no-op unless `profile=True`), so it adds no overhead to normal
  runs. Optionally, when `profile=True`, also emit a `cProfile` dump of the imaging call for hotspot
  detail.

### 3.3 Phase 1 CLI additions (`cli/smoovie.py`)

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `catalog_cache` | `str \| None` | `None` | Sidecar cache path; `None` → `<movie>.catalog.json`. |
| `profile` | `bool` | `False` | Print per-stage timing summary. |
| `nframes` | `int \| None` | `None` | Cap frames imaged/rendered (profiling/preview aid). |

Threaded through the preflight/core/container dicts and the cab, regenerated via the pre-commit hook;
the round-trip test must stay byte-stable.

### 3.4 Phase 1 deliverable

Run the user's command with `--profile` once (first run populates the cache; a second confirms it is
network-free) and capture the per-stage table. **These numbers set Phase 2's priorities** — in
particular whether rendering must be optimised alongside the DFT.

## 4. Phase 2 — GPU Holoscan `smoovie` (architecture)

> Implementation plan to be written **after** Phase 1 profiling. Captured here so the direction is not
> lost and the Phase 1 code is shaped to fit it.

### 4.1 Data flow (mirrors `core/stream_msv4.py`)

```
HDFs ─prepare(host)─► MSv4 zarr  ─reader─► [GPU] HealpixDFTOperator ─► writer ─► healpix zarr
        gains applied,            stream per-frame                     (TIME, npix)
        b_rot(t) precomputed      (vis_corr, wgt, b_rot, time)         dirty maps + meta
                                                                              │
              after app.run() (host): load zarr ─► mollview + cached overlay ─► ffmpeg ─► mp4
```

### 4.2 Components

- **Prepare step (host, new — e.g. `core/smoovie_prepare.py` or `utils/`):** HDF sequence → one MSv4
  zarr. Applies inverse gains (reuse `utils.gains.apply_inverse_gains`) and precomputes per-frame
  `b_rot(t)` via the existing astropy `equatorial_baselines`, vectorised over all frames. Writes a zarr
  carrying corrected `VISIBILITY`, `WEIGHT`, `b_rot` (`(n_time, nbl, 3)`), `time`, `frequency`, plus the
  phase-direction metadata. Reusable across runs (a form of caching for the heavy host work).
- **Reader operator:** adapt `operators/io.py::XarrayZarrReaderOperator` to stream per-frame
  `(VISIBILITY, WEIGHT, b_rot, time)` to the GPU (drop UVW/FLAG it does not need; add `b_rot`).
- **`HealpixDFTOperator` → pure GPU (`operators/dft_healpix.py`):** the key refactor. Split
  `utils/healpix_dft.image_frame` into (a) host `equatorial_baselines` (moves to the prepare step) and
  (b) a device-only `image_frame_prerotated(vis, wgt, b_rot, pix_vec, freqs, *, xp)` (flatten
  `(time, baseline)` → rows, adjoint-DFT onto the fixed grid). The operator consumes streamed `b_rot`
  and calls `image_frame_prerotated` with `xp=cupy`, so `compute()` is pure cupy — no astropy round-trip
  inside the operator. The CPU `smoovie` keeps working via the same split (it computes `b_rot` then calls
  the prerotated function).
- **Writer operator:** a healpix variant of `ResultWriterOperator` writing a `(TIME, npix)` dirty-map
  zarr (dask-scaffold + `region="auto"`, as today) plus per-frame metadata (timestamp, phase center).
- **Post-`app.run()` (host, in `core/smoovie`):** load the healpix zarr, render Mollweide frames with the
  cached satellite overlay (`render_frames` / `_overlay_tracks` unchanged), then `encode_movie`. Movie
  creation stays outside the Holoscan app.
- **`smoovie()` core (Phase 2 shape):** `prepare(HDFs → zarr)` → build & `app.run()` (GPU imaging →
  healpix zarr) → render + overlay → ffmpeg. The Holoscan app uses the empty `tests/data/config.yaml`
  (an empty Holoscan config is valid) via `app.config(...)`.

### 4.3 `healpix_dft` refactor (shared by CPU and GPU)

`image_frame(vis, wgt, times, itrs_baselines, pix_vec, freqs, ...)` is decomposed:

```
b_rot = equatorial_baselines(itrs_baselines, times, ...)          # host (astropy)
return image_frame_prerotated(vis, wgt, b_rot, pix_vec, freqs, xp=xp)   # device-pure
```

`image_frame` keeps its current signature/behaviour (now implemented via the two pieces), so existing
callers and tests are unaffected. `image_frame_prerotated` is the new pure-`xp` entry point the GPU
operator and the prepare-fed CPU path use.

## 5. Stumbling blocks / risks

- **Rendering may rival the DFT.** ~540 `mollview` saves at `nside=128` could be a co-dominant cost.
  Phase 1 profiling decides whether Phase 2 must also optimise rendering (lower dpi, a faster projection
  path) rather than only the DFT. Do not assume the DFT is the only bottleneck until measured.
- **GPU operators are import-time GPU-only** (`cupy`/`holoscan` at module top), so their tests are
  gated on a GPU being present and will not run in CPU CI. The pure DFT logic
  (`image_frame_prerotated`) stays `xp`-injectable and CPU-tested. A GPU is available in the dev
  environment for local validation.
- **`config.yaml`:** an empty `tests/data/config.yaml` is provided and reused; an empty Holoscan config
  is valid.
- **GPU memory:** ~870 MB kernel/frame at `nside=128`. Per-frame streaming is the default; multi-frame
  batching is a later tuning knob, not in initial scope.
- **Behaviour preservation:** the restructure must not change the imaged result. An equivalence test
  (GPU-app dirty maps ≈ current CPU `frame_dirty_maps` at small `nside`) guards this.

## 6. Testing

**Phase 1 (CPU, no network):**
- Cache round-trip: first `satellite_tracks` call with a `cache_path` writes the file; a second call with
  the same `cache_path` and a `fetch` that asserts it is **never called** returns identical tracks.
- Cache keying: different `(lat, lon, elevation_deg, datestr)` produce distinct entries; a changed
  `elevation_deg` is a cache miss.
- Profiling harness smoke: `smoovie(..., profile=True, nframes=2)` runs, prints a stage summary, and
  produces a valid (short) movie; `nframes` truncates the frame count.
- Round-trip: `test_roundtrip.py::test_roundtrip_smoovie` regenerated for the three new CLI params.

**Phase 2 (its own plan; GPU-gated where noted):**
- `image_frame_prerotated` equals `image_frame` (CPU, `xp=np`) on the test data at small `nside`.
- Prepare step produces a valid MSv4 zarr with corrected `VISIBILITY` and `b_rot` of the right shape.
- `HealpixDFTOperator` (gated on `cupy`) emits the expected dirty-map shape.
- End-to-end: GPU-app dirty maps ≈ CPU `frame_dirty_maps` baseline at small `nside` (behaviour
  preservation), and the post-app render+ffmpeg produces a playable mp4.

## 7. Out of scope

- Multi-frame GPU batching and any GPU-native `C(t)` polynomial (host astropy `C(t)` is retained).
- Multi-TART mosaicking, primary-beam weighting, below-horizon masking (unchanged from prior specs).
- Changing the gain-correction or overlay semantics (only their performance / plumbing changes).
- Streaming directly from HDF without the prepare-step zarr (explicitly rejected in §2).
