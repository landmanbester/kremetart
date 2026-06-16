# Smoovie Performance Phase 1 — Catalog Caching & Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kremetart smoovie` re-runnable without the network (cache the TART catalogue to a time-indexed xarray/zarr dataset) and measurable (a per-stage profiling harness + a `--nframes` cap), so the next step can target the real bottleneck with evidence.

**Architecture:** `satellite_tracks` gains a `cache_path` and persists the raw catalogue as a `(time, source)` xarray `Dataset` (zarr), reusing cached frames and fetching only misses. `core/smoovie` gains a lightweight stage timer, a `--profile` summary, and a `--nframes` cap threaded into `frame_dirty_maps` and `satellite_tracks`. Three new CLI params are added and the cab regenerated. This is Phase 1 of `docs/superpowers/specs/2026-06-16-smoovie-performance-and-gpu-design.md`; Phase 2 (GPU Holoscan) gets its own plan after we read the profiling numbers.

**Tech Stack:** Python 3.10+, NumPy, xarray + zarr 3.x, astropy (AltAz→ICRS), Typer + hip-cargo (CLI→cab), pytest.

---

## Background the engineer needs

- **Run tests:** `uv run pytest <path> -v`. **After every code change:** `uv run ruff format . && uv run ruff check . --fix`. Commit only when a task says to; the pre-commit hook re-runs ruff + a cab generator and may modify files (re-stage and re-commit if so). Branch is `actually_make_movie` (commit here; do not switch branches).
- **Files in play:**
  - `src/kremetart/utils/satellites.py` — `satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch)` builds per-satellite ICRS tracks; `_frame_times_and_site(hdf_paths)` returns `(times_unix, lat, lon, alt)`; `_tart_api_fetch` is the injectable network call. Frame ordering iterates `for path in hdf_paths: for k in range(n_time)`.
  - `src/kremetart/core/smoovie.py` — `frame_dirty_maps(hdf_paths, nside, *, correct_gains=False, xp=np)` (images one frame per sub-integration), `render_frames`, `encode_movie`, and `smoovie(...)` (the orchestration). Module top imports: `from __future__ import annotations`, `import datetime`, `from pathlib import Path`, `import numpy as np`.
  - `src/kremetart/cli/smoovie.py` — thin Typer wrapper; params currently end `...phase_dec_deg, correct_gains, overlay_catalog, catalog_elevation_deg, movie, backend, always_pull_images`. Three dicts (preflight / core call / `run_in_container`) each list args; new inputs go **before** `movie=movie,`.
  - `src/kremetart/cabs/smoovie.yml` — generated; never hand-edit.
  - `tests/test_satellites.py`, `tests/test_smoovie_core.py`, `tests/test_roundtrip.py`.
- **Catalogue source dict** (one per visible satellite per frame): `{"name": str, "az": deg, "el": deg, "jy": flux, "r": height_m}`.
- **Verified fact (don't re-investigate):** writing a `(time, source)` xarray `Dataset` with an **object-dtype** `source_name` string array to zarr 3.x via `ds.to_zarr(path, mode="w")` and reading it back with `xr.open_zarr(path)` round-trips correctly. It emits a harmless `UnstableSpecificationWarning` (forward-compat note, not an error) — do **not** treat it as a failure; the test suite does not error on warnings.
- **Cache identity:** a cache is valid for a given `(site_latitude_deg, site_longitude_deg, elevation_deg)` (stored as dataset attrs). A change in any of these is a miss → refetch.
- **`time` axis:** the cache's `time` coordinate is **unix seconds** (same convention as `main.time.values` and the Phase-2 MSv4 zarr), with a secondary `datestr` coord (the ISO string the API was queried with) used for cache lookup.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/kremetart/utils/satellites.py` | `_load_catalog_cache` / `_save_catalog_cache` helpers; `satellite_tracks` gains `cache_path` + `nframes`. | Modify (Task 1) |
| `src/kremetart/core/smoovie.py` | `nframes` in `frame_dirty_maps`; `_stage_timer` + `_print_profile`; `smoovie` gains `profile`/`nframes`/`catalog_cache` wiring. | Modify (Tasks 2, 3) |
| `src/kremetart/cli/smoovie.py` | three new CLI params threaded through the three dicts. | Modify (Task 4) |
| `src/kremetart/cabs/smoovie.yml` | regenerated cab. | Regenerated (Task 4) |
| `tests/test_satellites.py` | cache round-trip, schema, miss-on-elevation, nframes-cap. | Modify (Task 1) |
| `tests/test_smoovie_core.py` | nframes cap in imaging; profiling harness; cache-path wiring. | Modify (Tasks 2, 3) |

---

## Task 1: catalogue cache as a time-indexed zarr dataset

**Files:**
- Modify: `src/kremetart/utils/satellites.py`
- Test: `tests/test_satellites.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_satellites.py`:

```python
def test_satellite_tracks_caches_and_reuses(tmp_path):
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:1]
    cache = tmp_path / "cat.zarr"
    calls = {"n": 0}

    def fetch(lon, lat, datestr, elevation_deg):
        calls["n"] += 1
        return [{"name": "SAT-A", "az": 30.0, "el": 60.0, "jy": 1.0, "r": 7.0e6}]

    first = satellite_tracks(paths, 45.0, fetch=fetch, cache_path=str(cache))
    assert calls["n"] > 0 and cache.exists()

    def fetch_forbidden(lon, lat, datestr, elevation_deg):
        raise AssertionError("cache hit must not fetch")

    second = satellite_tracks(paths, 45.0, fetch=fetch_forbidden, cache_path=str(cache))
    assert first == second  # identical tracks, no network on the second run


def test_catalog_cache_schema(tmp_path):
    xr = pytest.importorskip("xarray")
    from kremetart.utils.satellites import _frame_times_and_site, satellite_tracks

    paths = _hdfs()[:1]
    cache = tmp_path / "cat.zarr"

    def fetch(lon, lat, datestr, elevation_deg):
        return [
            {"name": "SAT-A", "az": 30.0, "el": 60.0, "jy": 1.0, "r": 7.0e6},
            {"name": "SAT-B", "az": 10.0, "el": 50.0, "jy": 0.5, "r": 8.0e6},
        ]

    satellite_tracks(paths, 45.0, fetch=fetch, cache_path=str(cache))
    ds = xr.open_zarr(str(cache))
    assert set(ds.dims) == {"time", "source"}
    for v in ("source_name", "source_elevation_deg", "source_azimuth_deg", "source_flux_jy", "source_height_m"):
        assert v in ds.data_vars
    assert ds.attrs["elevation_deg"] == 45.0
    times, *_ = _frame_times_and_site(paths)
    np.testing.assert_allclose(ds.time.values, times)


def test_catalog_cache_miss_on_elevation_change(tmp_path):
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:1]
    cache = tmp_path / "cat.zarr"
    calls = {"n": 0}

    def fetch(lon, lat, datestr, elevation_deg):
        calls["n"] += 1
        return [{"name": "SAT-A", "az": 30.0, "el": 60.0, "jy": 1.0, "r": 7.0e6}]

    satellite_tracks(paths, 45.0, fetch=fetch, cache_path=str(cache))
    after_first = calls["n"]
    satellite_tracks(paths, 30.0, fetch=fetch, cache_path=str(cache))  # different elevation -> miss
    assert calls["n"] > after_first


def test_satellite_tracks_nframes_caps(tmp_path):
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:1]
    calls = {"n": 0}

    def fetch(lon, lat, datestr, elevation_deg):
        calls["n"] += 1
        return [{"name": "SAT-A", "az": 30.0, "el": 60.0, "jy": 1.0, "r": 7.0e6}]

    satellite_tracks(paths, 45.0, fetch=fetch, nframes=2)
    assert calls["n"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_satellites.py -v`
Expected: the four new tests FAIL with `TypeError: satellite_tracks() got an unexpected keyword argument 'cache_path'` / `'nframes'`.

- [ ] **Step 3: Add the cache helpers**

In `src/kremetart/utils/satellites.py`, add these two module-level functions **above** `def satellite_tracks(`:

```python
def _load_catalog_cache(path, lat, lon, elevation_deg):
    """Return ``{datestr -> source-list}`` from a cached catalogue zarr, or ``None`` on miss.

    A cache is reusable only if its site/elevation attrs match the request; otherwise ``None``
    (forces a refetch). Padding slots (``source_name == ""``) are dropped on read.
    """
    import os

    if not os.path.exists(path):
        return None
    import xarray as xr

    ds = xr.open_zarr(path)
    a = ds.attrs
    if not (
        np.isclose(a.get("site_latitude_deg", np.nan), lat)
        and np.isclose(a.get("site_longitude_deg", np.nan), lon)
        and np.isclose(a.get("elevation_deg", np.nan), elevation_deg)
    ):
        return None
    names = ds.source_name.values
    el = ds.source_elevation_deg.values
    az = ds.source_azimuth_deg.values
    jy = ds.source_flux_jy.values
    r = ds.source_height_m.values
    out: dict[str, list] = {}
    for ti, datestr in enumerate(ds.datestr.values):
        sources = []
        for si in range(names.shape[1]):
            name = str(names[ti, si])
            if name == "":
                continue  # padding slot
            sources.append(
                {
                    "name": name,
                    "el": float(el[ti, si]),
                    "az": float(az[ti, si]),
                    "jy": float(jy[ti, si]),
                    "r": float(r[ti, si]),
                }
            )
        out[str(datestr)] = sources
    return out


def _save_catalog_cache(path, datestrs, times_unix, per_frame, lat, lon, elevation_deg):
    """Write per-frame catalogue source lists to a ``(time, source)`` zarr ``Dataset``.

    ``source`` is padded to the max source count over all frames (``""`` / ``NaN`` for empty slots).
    ``source_name`` is stored as an object-dtype string array (verified to round-trip through zarr).
    """
    import os
    import shutil

    import xarray as xr

    nt = len(per_frame)
    nsrc = max((len(s) for s in per_frame), default=0)
    name = np.full((nt, nsrc), "", dtype=object)
    el = np.full((nt, nsrc), np.nan)
    az = np.full((nt, nsrc), np.nan)
    jy = np.full((nt, nsrc), np.nan)
    r = np.full((nt, nsrc), np.nan)
    for ti, sources in enumerate(per_frame):
        for si, s in enumerate(sources):
            name[ti, si] = str(s["name"])
            el[ti, si] = float(s["el"])
            az[ti, si] = float(s["az"])
            jy[ti, si] = float(s["jy"])
            r[ti, si] = float(s["r"])
    ds = xr.Dataset(
        data_vars={
            "source_name": (("time", "source"), name),
            "source_elevation_deg": (("time", "source"), el),
            "source_azimuth_deg": (("time", "source"), az),
            "source_flux_jy": (("time", "source"), jy),
            "source_height_m": (("time", "source"), r),
        },
        coords={
            "time": ("time", np.asarray(times_unix, dtype=np.float64)),
            "datestr": ("time", np.asarray(datestrs)),
        },
        attrs={
            "site_latitude_deg": float(lat),
            "site_longitude_deg": float(lon),
            "elevation_deg": float(elevation_deg),
        },
    )
    if os.path.exists(path):
        shutil.rmtree(path)  # zarr is a directory; overwrite cleanly
    ds.to_zarr(path, mode="w")
```

- [ ] **Step 4: Rewrite `satellite_tracks` to be cache-aware**

In `src/kremetart/utils/satellites.py`, change the signature line:

```python
def satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch):
```

to:

```python
def satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch, cache_path=None, nframes=None):
```

Add to its docstring `Args:` block:

```
        cache_path: optional zarr path; cached frames are reused and only misses are fetched, then
            the (time, source) dataset is rewritten. ``None`` disables caching.
        nframes: optional cap on the number of leading frames processed (profiling/preview aid).
```

Then replace the function body (everything from `times_unix, lat, lon, alt = _frame_times_and_site(hdf_paths)` through `return tracks`) with:

```python
    times_unix, lat, lon, alt = _frame_times_and_site(hdf_paths)
    if nframes is not None:
        times_unix = times_unix[:nframes]
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)

    datestrs = [
        datetime.datetime.fromtimestamp(float(t), tz=datetime.timezone.utc).isoformat() for t in times_unix
    ]

    # Cache-aware per-frame source lists: reuse cached frames, fetch only the misses.
    cached = _load_catalog_cache(cache_path, lat, lon, elevation_deg) if cache_path else None
    per_frame, fetched_any = [], False
    for datestr in datestrs:
        if cached is not None and datestr in cached:
            per_frame.append(cached[datestr])
        else:
            per_frame.append(fetch(lon, lat, datestr, elevation_deg))
            fetched_any = True
    if cache_path and (cached is None or fetched_any):
        _save_catalog_cache(cache_path, datestrs, times_unix, per_frame, lat, lon, elevation_deg)

    # Convert az/el -> ICRS per frame and group into per-satellite tracks.
    tracks: dict[str, list] = {}
    for i, (t, sources) in enumerate(zip(times_unix, per_frame)):
        if not sources:
            continue
        az = np.array([float(s["az"]) for s in sources])
        el = np.array([float(s["el"]) for s in sources])
        obstime = Time(float(t), format="unix", scale="utc")
        icrs = SkyCoord(AltAz(az=az * u.deg, alt=el * u.deg, obstime=obstime, location=loc)).icrs
        for src, ra, dec in zip(sources, icrs.ra.deg, icrs.dec.deg):
            tracks.setdefault(src["name"], []).append((i, float(ra), float(dec), float(src["jy"])))
    return tracks
```

- [ ] **Step 5: Run to verify the new tests pass and the old ones still pass**

Run: `uv run pytest tests/test_satellites.py -v`
Expected: all pass (the 3 pre-existing `satellite_tracks` tests + the 4 new ones), or skip if test data is absent.

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/satellites.py tests/test_satellites.py
git commit -m "feat: cache TART catalogue to a time-indexed zarr dataset"
```

---

## Task 2: `nframes` cap in `frame_dirty_maps`

**Files:**
- Modify: `src/kremetart/core/smoovie.py`
- Test: `tests/test_smoovie_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoovie_core.py`:

```python
def test_frame_dirty_maps_nframes_caps():
    paths = _hdfs()  # multiple files; nframes caps the total frames produced
    maps, stamps, pix = frame_dirty_maps(paths, 16, nframes=3)
    assert len(maps) == len(stamps) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_smoovie_core.py::test_frame_dirty_maps_nframes_caps -v`
Expected: FAIL with `TypeError: frame_dirty_maps() got an unexpected keyword argument 'nframes'`.

- [ ] **Step 3: Add `nframes` to `frame_dirty_maps`**

In `src/kremetart/core/smoovie.py`, change the signature:

```python
def frame_dirty_maps(hdf_paths, nside: int, *, correct_gains: bool = False, xp=np):
```

to:

```python
def frame_dirty_maps(hdf_paths, nside: int, *, correct_gains: bool = False, nframes: int | None = None, xp=np):
```

Add to its docstring `Args:` block (next to the `correct_gains:` line):

```
        nframes: optional cap on the total number of frames produced (profiling/preview aid).
```

Then, in the body, add a cap check at the **top of the outer loop** and **top of the inner loop**. The outer loop currently begins:

```python
    for path in hdf_paths:
        node = _partition(read_hdf_as_msv4(path))
```

Change it to:

```python
    for path in hdf_paths:
        if nframes is not None and len(maps) >= nframes:
            break
        node = _partition(read_hdf_as_msv4(path))
```

The inner loop currently begins:

```python
        for k in range(times.size):
            dmap = image_frame(vis[k : k + 1], wgt[k : k + 1], times[k : k + 1], bl, pix_vec, freqs, xp=xp)
```

Change it to:

```python
        for k in range(times.size):
            if nframes is not None and len(maps) >= nframes:
                break
            dmap = image_frame(vis[k : k + 1], wgt[k : k + 1], times[k : k + 1], bl, pix_vec, freqs, xp=xp)
```

- [ ] **Step 4: Run to verify it passes (and existing tests still pass)**

Run: `uv run pytest tests/test_smoovie_core.py::test_frame_dirty_maps_nframes_caps tests/test_smoovie_core.py::test_frame_dirty_maps_one_frame_per_subintegration -v`
Expected: both PASS (or skip if data absent).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/core/smoovie.py tests/test_smoovie_core.py
git commit -m "feat: add nframes cap to frame_dirty_maps"
```

---

## Task 3: profiling harness + `smoovie` wiring (`profile`, `nframes`, `catalog_cache`)

**Files:**
- Modify: `src/kremetart/core/smoovie.py`
- Test: `tests/test_smoovie_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_smoovie_core.py`:

```python
def test_print_profile_outputs(capsys):
    from kremetart.core.smoovie import _print_profile

    _print_profile([("imaging", 2.0), ("render", 1.0)], nframes=4)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out and "render" in out and "TOTAL" in out


def test_stage_timer_records():
    from kremetart.core.smoovie import _stage_timer

    timings = []
    with _stage_timer("stage_a", timings):
        pass
    assert len(timings) == 1 and timings[0][0] == "stage_a" and timings[0][1] >= 0.0


def test_smoovie_profile_prints(tmp_path, monkeypatch, capsys):
    import kremetart.core.smoovie as sm

    _hdfs()
    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "render_frames", lambda *a, **k: [Path("frame_0000.png")])
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, profile=True)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out and "render" in out


def test_smoovie_nframes_flows_to_imaging(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm

    _hdfs()
    captured = {}

    def fake_fdm(paths, nside, **k):
        captured.update(k)
        return ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))

    monkeypatch.setattr(sm, "frame_dirty_maps", fake_fdm)
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "render_frames", lambda *a, **k: [Path("frame_0000.png")])
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, nframes=7)
    assert captured.get("nframes") == 7


def test_smoovie_default_catalog_cache_path(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm
    import kremetart.utils.satellites as sat

    _hdfs()
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "render_frames", lambda *a, **k: [Path("frame_0000.png")])
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    def fake_tracks(paths, elevation_deg, **k):
        captured.update(k)
        return {}

    monkeypatch.setattr(sat, "satellite_tracks", fake_tracks)

    movie = tmp_path / "m.mp4"
    sm.smoovie(hdf_dir=_DATA, movie=movie, nside=1, overlay_catalog=True, nframes=3)
    assert captured["cache_path"] == str(movie) + ".catalog.zarr"
    assert captured["nframes"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_smoovie_core.py -k "profile or stage_timer or nframes_flows or default_catalog_cache" -v`
Expected: FAIL — `ImportError: cannot import name '_print_profile'` / `_stage_timer`, and `smoovie() got an unexpected keyword argument 'profile'`.

- [ ] **Step 3: Add the timer + summary helpers**

In `src/kremetart/core/smoovie.py`, add `import contextlib` and `import time` to the module top imports (next to `import datetime`). Then add these two functions immediately **above** `def smoovie(`:

```python
@contextlib.contextmanager
def _stage_timer(name, timings):
    """Record wall-clock seconds for a named stage into ``timings`` (a list of ``(name, seconds)``)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings.append((name, time.perf_counter() - t0))


def _print_profile(timings, nframes):
    """Print a per-stage timing summary table to stdout."""
    total = sum(dt for _, dt in timings) or 1.0
    print("\n=== smoovie profile ===")
    print(f"{'stage':<18}{'seconds':>10}{'%total':>9}{'ms/frame':>11}")
    for name, dt in timings:
        per_frame = f"{1000.0 * dt / nframes:.1f}" if nframes else "-"
        print(f"{name:<18}{dt:>10.3f}{100.0 * dt / total:>8.1f}%{per_frame:>11}")
    print(f"{'TOTAL':<18}{total:>10.3f}{100.0:>8.1f}%")
```

- [ ] **Step 4: Add the new params to `smoovie` and wire the stages**

In `src/kremetart/core/smoovie.py`, change the `smoovie` signature from:

```python
def smoovie(
    hdf_dir,
    movie,
    nside: int = 128,
    fps: int = 2,
    cmap: str = "inferno",
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    correct_gains: bool = False,
    overlay_catalog: bool = False,
    catalog_elevation_deg: float = 45.0,
):
```

to:

```python
def smoovie(
    hdf_dir,
    movie,
    nside: int = 128,
    fps: int = 2,
    cmap: str = "inferno",
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    correct_gains: bool = False,
    overlay_catalog: bool = False,
    catalog_elevation_deg: float = 45.0,
    catalog_cache: str | None = None,
    profile: bool = False,
    nframes: int | None = None,
):
```

Add to the docstring (after the `overlay_catalog` sentence):

```
    ``catalog_cache`` is the zarr path for the cached catalogue (``None`` -> ``<movie>.catalog.zarr``).
    ``profile`` prints a per-stage timing summary; ``nframes`` caps the frames imaged/rendered (a
    profiling/preview aid).
```

Then replace the body from `maps, stamps, _ = frame_dirty_maps(...)` through `return movie` with:

```python
    timings: list[tuple[str, float]] = []

    with _stage_timer("phase_direction", timings):
        if phase_ra_deg is None:
            phase_ra_deg, phase_dec_deg = common_phase_direction(hdf_paths)

    with _stage_timer("imaging", timings):
        maps, stamps, _ = frame_dirty_maps(hdf_paths, nside, correct_gains=correct_gains, nframes=nframes)

    tracks = None
    if overlay_catalog:
        from kremetart.utils.satellites import satellite_tracks

        cache_path = catalog_cache if catalog_cache is not None else str(movie) + ".catalog.zarr"
        with _stage_timer("catalog", timings):
            tracks = satellite_tracks(hdf_paths, catalog_elevation_deg, cache_path=cache_path, nframes=nframes)

    with tempfile.TemporaryDirectory() as td:
        with _stage_timer("render", timings):
            pngs = render_frames(maps, stamps, nside, cmap, Path(td), rot=(phase_ra_deg, phase_dec_deg), tracks=tracks)
        with _stage_timer("encode", timings):
            encode_movie(pngs, movie, fps)

    if profile:
        _print_profile(timings, len(maps))
    return movie
```

Note: the pre-existing `phase_ra_deg`/`phase_dec_deg` validation, `hdf_dir`/`movie` `Path` conversion, the `hdf_paths` glob, and the empty-glob `FileNotFoundError` stay exactly as they are above this block; only the trailing orchestration (from `maps, stamps, _ = ...`) is replaced.

- [ ] **Step 5: Run the new and existing smoovie tests**

Run: `uv run pytest tests/test_smoovie_core.py -k "profile or stage_timer or nframes or default_catalog_cache or overlay or auto_phase or honors" -v`
Expected: all PASS (new profiling/wiring tests + the existing monkeypatched `smoovie` tests, which absorb the new kwargs).

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/core/smoovie.py tests/test_smoovie_core.py
git commit -m "feat: add per-stage profiling and nframes/catalog-cache wiring to smoovie"
```

---

## Task 4: CLI params + cab regeneration + round-trip

**Files:**
- Modify: `src/kremetart/cli/smoovie.py`
- Regenerated (do NOT hand-edit): `src/kremetart/cabs/smoovie.yml`
- Test: `tests/test_roundtrip.py::test_roundtrip_smoovie` (exists — must stay byte-stable)

> **Round-trip rule:** `cli/smoovie.py` must match exactly what `hip-cargo generate-function` emits, or the round-trip test fails on a byte diff. The new blocks mirror existing canonical blocks (`catalog_elevation_deg` for `float`, `phase_ra_deg` for `Optional[float]`, `correct_gains` for `bool`). The cab is regenerated by the tool, never hand-edited.

- [ ] **Step 1: Add the three parameters to the signature**

In `src/kremetart/cli/smoovie.py`, find the `catalog_elevation_deg` block (it ends with `] = 45.0,`) immediately followed by the `movie:` block. Insert these three blocks **between** `catalog_elevation_deg`'s `] = 45.0,` line and `    movie: Annotated[`:

```python
    catalog_cache: Annotated[
        str | None,
        typer.Option(
            help="Catalog cache zarr path; defaults to <movie>.catalog.zarr.",
        ),
    ] = None,
    profile: Annotated[
        bool,
        typer.Option(
            help="Print a per-stage timing summary.",
        ),
    ] = False,
    nframes: Annotated[
        int | None,
        typer.Option(
            help="Cap the number of frames imaged/rendered (profiling/preview aid).",
        ),
    ] = None,
```

- [ ] **Step 2: Thread through the preflight dict**

In the `preflight_remote_must_exist(smoovie, dict(...))` call, the arg list has `catalog_elevation_deg=catalog_elevation_deg,` immediately before `movie=movie,` (indented 20 spaces). Insert the three keys before `movie=movie,`:

```python
                    catalog_cache=catalog_cache,
                    profile=profile,
                    nframes=nframes,
```

- [ ] **Step 3: Thread through the core call**

In the `smoovie_core(...)` call, the arg list has `catalog_elevation_deg=catalog_elevation_deg,` immediately before `movie=movie,` (indented 16 spaces). Insert the three keys before `movie=movie,`:

```python
                catalog_cache=catalog_cache,
                profile=profile,
                nframes=nframes,
```

- [ ] **Step 4: Thread through the `run_in_container` dict**

In the `run_in_container(smoovie, dict(...), ...)` call, the arg list has `catalog_elevation_deg=catalog_elevation_deg,` immediately before `movie=movie,` (indented 12 spaces). Insert the three keys before `movie=movie,`:

```python
            catalog_cache=catalog_cache,
            profile=profile,
            nframes=nframes,
```

- [ ] **Step 5: Format, regenerate the cab, run the round-trip**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run hip-cargo generate-cabs --module src/kremetart/cli/smoovie.py --output-dir src/kremetart/cabs
uv run pytest tests/test_roundtrip.py::test_roundtrip_smoovie -v
```

Expected: the cab gains `catalog-cache` (`dtype: Optional[str]`), `profile` (`dtype: bool, default: false`), and `nframes` (`dtype: Optional[int]`) under `inputs:`, retains its `image:` field, and the round-trip test PASSES (regenerated source byte-identical to `cli/smoovie.py`). If the round-trip fails on a line diff, make `cli/smoovie.py` match the **Generated** form printed in the failure message (that is the canonical shape), then re-run.

- [ ] **Step 6: Commit**

```bash
git add src/kremetart/cli/smoovie.py src/kremetart/cabs/smoovie.yml
git commit -m "feat: expose catalog-cache, profile, and nframes on smoovie CLI"
```

---

## Final verification

- [ ] **Targeted suites green**

Run: `uv run pytest tests/test_satellites.py tests/test_roundtrip.py -v`
Expected: all pass (or skip if test data absent).

- [ ] **Lint/format clean**

Run: `uv run ruff format --check . && uv run ruff check .`
Expected: no changes needed, no errors.

- [ ] **Capture the profiling numbers (the Phase 1 deliverable)**

This is the point of Phase 1 — it needs the TART catalogue API (network) on the first run, `ffmpeg`, and `matplotlib`. Run:

```bash
# first run: populates the catalogue cache (network), prints the stage table
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/tart.mp4 --fps 4 \
    --correct-gains --overlay-catalog --profile --backend native
# second run: confirm it is network-free (delete-cache check)
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/tart2.mp4 --fps 4 \
    --correct-gains --overlay-catalog --profile --catalog-cache /tmp/tart.mp4.catalog.zarr --backend native
```

Expected: first run produces `/tmp/tart.mp4` + `/tmp/tart.mp4.catalog.zarr` and prints the per-stage table; second run reuses the cache (no catalogue network calls) and prints its own table. Record the per-stage seconds / %total / ms-per-frame — **these numbers decide Phase 2's priorities** (e.g. whether `render` rivals `imaging`). A `--nframes 30` run is a fast way to get a representative slice without all ~540 frames.

---

## Self-Review (completed during planning)

**Spec coverage (vs `2026-06-16-smoovie-performance-and-gpu-design.md`):** §3.1 catalogue cache (schema, `time` coord in unix seconds, raw az/el, attrs identity, reuse-without-fetch, derived tracks unchanged) → Task 1. §3.2 profiling harness (`_stage_timer`, `_print_profile`, `profile`, `nframes`) → Tasks 2–3. §3.3 CLI params (`catalog_cache`, `profile`, `nframes`) + round-trip → Task 4. §3.4 deliverable (run with `--profile`, confirm network-free re-run, capture numbers) → Final verification. Phase 2 (§4) is intentionally a separate future plan.

**Placeholder scan:** none — every code/edit step shows the actual code; every command states its expected result.

**Type/name consistency:** `_load_catalog_cache(path, lat, lon, elevation_deg)` and `_save_catalog_cache(path, datestrs, times_unix, per_frame, lat, lon, elevation_deg)` defined and called identically in Task 1. `satellite_tracks(..., cache_path=None, nframes=None)` (Task 1) is called with `cache_path=`/`nframes=` from `smoovie` (Task 3) and exercised by Task 1 tests. `frame_dirty_maps(..., nframes=None, xp=np)` (Task 2) is called with `nframes=nframes` from `smoovie` (Task 3). `_stage_timer(name, timings)` / `_print_profile(timings, nframes)` (Task 3) match their tests. CLI param names `catalog_cache`/`profile`/`nframes` (Task 4) match the `smoovie` core signature (Task 3). Cache `time` coordinate is unix seconds in both writer (`_save_catalog_cache`) and the schema test.
