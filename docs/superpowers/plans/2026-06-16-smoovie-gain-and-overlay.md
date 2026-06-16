# Smoovie Gain Correction & Satellite-Track Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in capabilities to `kremetart smoovie` — applying the inverse TART gain solution before imaging (`--correct-gains`), and overlaying expected satellite trajectories from the TART catalogue on every frame (`--overlay-catalog`) — so we can verify that imaged sources land where the catalogue predicts.

**Architecture:** A new `utils/gains.py` holds the pure inverse-gain correction (`V_pq^corr = V_pq / (g_p·conj(g_q))`, weights scaled by `|g_p·g_q|²`). A new `utils/satellites.py` fetches the TART catalogue (network, isolated from the hermetic MSv4 reader) and projects each satellite's `(az, el)` to ICRS `(ra, dec)` tracks aligned 1:1 with the frame sequence. `core/smoovie.py` wires both into `frame_dirty_maps` (correction) and `render_frames` (overlay via healpy `projscatter`/`projplot`/`projtext`). The `cli/smoovie.py` wrapper grows three params and is regenerated so the cab round-trip stays byte-stable.

**Tech Stack:** Python 3.10+, NumPy, astropy (AltAz→ICRS), healpy (HEALPix + Mollweide projection), matplotlib (Agg), `tart_tools` (catalogue API), `xarray`/`xarray_ms` (MSv4/MS reading), Typer + hip-cargo (CLI→cab), pytest.

---

## Background the engineer needs

- **Reading data.** `kremetart.utils.read_tart_hdf.read_hdf_as_msv4(path)` returns an `xarray.DataTree`. `kremetart.core.smoovie._partition(dt)` returns the single partition node. From a node:
  - `node.ds` is the main dataset. `node.ds.VISIBILITY.values` is `(n_time, nbl, nchan, npol)`; smoovie drops the single pol with `[..., 0]`. `node.ds.WEIGHT.values` matches. `node.ds.time.values` is `(n_time,)` float64 **unix seconds**. `node.ds.frequency.values` is `(nchan,)` Hz.
  - `node.ds.baseline_antenna1_name.values` / `baseline_antenna2_name.values` are `(nbl,)` arrays of antenna name strings (e.g. `"ant00"`).
  - `node["antenna_xds"].to_dataset(inherit=False).antenna_name.values` is the ordered antenna-name array.
  - `node["gain_xds"].to_dataset(inherit=False).GAIN.values` is `(n_ant,)` complex64 — TART's per-file gain snapshot, ordered by `antenna_name`. Dead antennas have `GAIN == 0` and are already weight-flagged.
  - `node.ds.attrs["observation_info"]` has `site_latitude_deg`, `site_longitude_deg`, `site_altitude_m`.
- **Frame ordering.** `frame_dirty_maps` produces one frame per sub-integration, iterating `for path in hdf_paths: for k in range(n_time)`. Any per-frame data (satellite tracks) MUST use the identical ordering so frame index `i` lines up.
- **Antenna→baseline mapping.** Build it exactly like `kremetart.utils.rephasing.itrs_baselines`: `index = {name: i for i, name in enumerate(antenna_name)}`, then `a1 = [index[n] for n in baseline_antenna1_name]`, `a2` likewise. The 24-antenna array has `nbl = 276`.
- **Timestamps.** HDF stores ISO strings like `'2026-06-09T08:10:42.993515+00:00'`; the MSv4 reader converts to unix seconds. The catalogue API wants an ISO datestr; convert back with `datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()`.
- **Catalogue API.** `tart_tools.api_handler.APIhandler("").catalog_url(lon, lat, datestr=iso) + f"&elevation={deg}"`, then `.get_url(url)` returns a **list of dicts**, each `{"name", "az", "el", "jy", "r"}` (az/el in degrees). `read_hdf_as_xr` already uses this path.
- **Test data.** `tests/data/` holds nine `vis_*.hdf` (60 sub-integrations each), a calibrated `vis_2026-06-09_08_11_43.476804.ms`, and `..._nocal.ms`. Tests skip if data is absent (`conftest.py` downloads it best-effort). Read an MS with `xarray.open_datatree(str(ms), engine="xarray-ms:msv2")`.
- **Mandatory after every code change:** `uv run ruff format . && uv run ruff check . --fix`.
- **Run tests with:** `uv run pytest <path> -v`.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/kremetart/utils/gains.py` | Pure inverse-gain correction of vis/weights. No I/O. | Create (Task 1) |
| `src/kremetart/utils/satellites.py` | Fetch TART catalogue (network) + project to per-satellite ICRS tracks aligned to frames. | Create (Task 3) |
| `src/kremetart/core/smoovie.py` | Wire correction into `frame_dirty_maps`; overlay into `render_frames`; new `smoovie` params. | Modify (Tasks 2, 4, 5) |
| `src/kremetart/cli/smoovie.py` | Three new CLI params, threaded through preflight/core/container dicts. | Modify (Task 6) |
| `src/kremetart/cabs/smoovie.yml` | Regenerated cab (never hand-edited). | Regenerated (Task 6) |
| `tests/test_gains.py` | Unit tests for `apply_inverse_gains`. | Create (Task 1) |
| `tests/test_satellites.py` | Track-assembly + alignment + az/el→ICRS tests (injected fetch, no network). | Create (Task 3) |
| `tests/test_smoovie_core.py` | Correction integration test, overlay render test, smoovie-wiring tests. | Modify (Tasks 2, 4, 5) |

---

## Task 1: `utils/gains.py` — inverse gain correction

**Files:**
- Create: `src/kremetart/utils/gains.py`
- Test: `tests/test_gains.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gains.py`:

```python
"""Unit tests for inverse per-antenna gain correction."""

import numpy as np

from kremetart.utils.gains import apply_inverse_gains


def test_apply_inverse_gains_divides_by_gain_product():
    # Antennas 0 and 1, one baseline (0, 1). V_corr = V / (g0 * conj(g1)).
    g0 = 2.0 + 0.0j
    g1 = 0.0 + 1.0j  # |g1| == 1
    gains = np.array([g0, g1], dtype=np.complex64)
    a1 = np.array([0])
    a2 = np.array([1])
    vis = np.array([[[4.0 + 0.0j]]], dtype=np.complex64)  # (n_time=1, nbl=1, nchan=1)
    wgt = np.array([[[1.0]]], dtype=np.float64)

    vis_c, wgt_c = apply_inverse_gains(vis, wgt, gains, a1, a2)

    factor = g0 * np.conj(g1)  # 2 * (-i) = -2i, |factor| == 2
    np.testing.assert_allclose(vis_c[0, 0, 0], (4.0 + 0.0j) / factor, rtol=1e-5)
    np.testing.assert_allclose(wgt_c[0, 0, 0], 1.0 * abs(factor) ** 2, rtol=1e-5)


def test_apply_inverse_gains_guards_dead_antenna():
    # Antenna 0 is dead (gain 0): the baseline must come out zeroed, not inf/nan.
    gains = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64)
    a1 = np.array([0])
    a2 = np.array([1])
    vis = np.array([[[3.0 + 1.0j]]], dtype=np.complex64)
    wgt = np.array([[[1.0]]], dtype=np.float64)

    vis_c, wgt_c = apply_inverse_gains(vis, wgt, gains, a1, a2)

    assert np.all(np.isfinite(vis_c))
    assert vis_c[0, 0, 0] == 0
    assert wgt_c[0, 0, 0] == 0


def test_apply_inverse_gains_broadcasts_over_time_and_channel():
    gains = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64)
    a1 = np.array([0, 0])  # two baselines
    a2 = np.array([1, 1])
    vis = np.ones((3, 2, 4), dtype=np.complex64)  # (n_time, nbl, nchan)
    wgt = np.ones((3, 2, 4), dtype=np.float64)

    vis_c, wgt_c = apply_inverse_gains(vis, wgt, gains, a1, a2)

    factor = 1.0 * np.conj(2.0)  # 2.0
    assert vis_c.shape == (3, 2, 4)
    np.testing.assert_allclose(vis_c, 1.0 / factor)
    np.testing.assert_allclose(wgt_c, abs(factor) ** 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gains.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kremetart.utils.gains'`.

- [ ] **Step 3: Write the implementation**

Create `src/kremetart/utils/gains.py`:

```python
"""Apply the inverse of a per-antenna gain solution to visibilities.

The TART measurement equation for a baseline ``(p, q)`` is
``V_pq = g_p · conj(g_q) · V_pq^true`` with per-antenna complex gains ``g``. Correcting
(calibrating) the data divides that factor out; the matching inverse-variance weight
transform scales the weight by ``|g_p · g_q|**2``. Dead antennas carry ``g == 0`` and are
already weight-flagged; the correction guards the division and keeps them at zero weight.

Pure array math, ``xp``-injectable (``numpy`` on CPU, ``cupy`` on GPU) like the rest of the
imaging pipeline. No I/O.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def apply_inverse_gains(vis, weight, gains, a1_idx, a2_idx, *, xp: ModuleType = np):
    """Correct ``vis``/``weight`` by the inverse per-antenna gain product.

    Args:
        vis: ``(n_time, nbl, nchan)`` complex visibilities (baseline on the middle axis).
        weight: ``(n_time, nbl, nchan)`` real weights, same layout as ``vis``.
        gains: ``(n_ant,)`` complex per-antenna gains, ordered by antenna index.
        a1_idx: ``(nbl,)`` integer antenna index of the first antenna of each baseline.
        a2_idx: ``(nbl,)`` integer antenna index of the second antenna of each baseline.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(vis_corr, weight_corr)`` with the same shapes as the inputs. Baselines touching a
        dead antenna (gain ``0``) come out as ``0`` vis and ``0`` weight (never ``inf``/``nan``).
    """
    vis = xp.asarray(vis)
    weight = xp.asarray(weight)
    gains = xp.asarray(gains)

    factor = gains[a1_idx] * xp.conj(gains[a2_idx])  # (nbl,)
    ok = xp.abs(factor) > 0  # dead antennas -> factor == 0
    safe = xp.where(ok, factor, 1.0 + 0.0j)  # avoid divide-by-zero before masking

    factor_b = factor[None, :, None]
    ok_b = ok[None, :, None]
    safe_b = safe[None, :, None]

    vis_corr = xp.where(ok_b, vis / safe_b, 0.0)
    weight_corr = xp.where(ok_b, weight * xp.abs(factor_b) ** 2, 0.0)
    return vis_corr, weight_corr
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gains.py -v`
Expected: 3 passed.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/gains.py tests/test_gains.py
git commit -m "feat: add inverse per-antenna gain correction helper"
```

---

## Task 2: wire gain correction into `frame_dirty_maps`

**Files:**
- Modify: `src/kremetart/core/smoovie.py` (add `_correct_file_gains`; add `correct_gains` param to `frame_dirty_maps`)
- Test: `tests/test_smoovie_core.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_smoovie_core.py` (after `test_frame_dirty_maps_one_frame_per_subintegration`):

```python
def test_correct_file_gains_real_data():
    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    node = _partition(read_hdf_as_msv4(_hdfs()[0]))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]

    vis_c, wgt_c = _correct_file_gains(node, vis, wgt)

    assert vis_c.shape == vis.shape
    assert np.all(np.isfinite(vis_c)) and np.all(np.isfinite(wgt_c))
    # The correction must actually change non-trivial gains.
    assert not np.allclose(vis_c, vis)
    # Dead antennas (gain 0) -> zero-weight, zero-vis baselines (no inf/nan).
    gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values
    if np.any(gains == 0):
        assert np.any(wgt_c == 0)


def test_frame_dirty_maps_correct_gains_finite():
    paths = _hdfs()[:1]
    nside = 16
    maps, stamps, pix = frame_dirty_maps(paths, nside, correct_gains=True)
    npix = 12 * nside * nside
    assert len(maps) == len(stamps) > 0
    for m in maps:
        assert m.shape == (npix,)
        assert np.all(np.isfinite(m))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_smoovie_core.py::test_correct_file_gains_real_data tests/test_smoovie_core.py::test_frame_dirty_maps_correct_gains_finite -v`
Expected: FAIL — `ImportError: cannot import name '_correct_file_gains'` and `frame_dirty_maps() got an unexpected keyword argument 'correct_gains'`.

- [ ] **Step 3: Add `_correct_file_gains` helper**

In `src/kremetart/core/smoovie.py`, add this function immediately **above** `def frame_dirty_maps(`:

```python
def _correct_file_gains(node, vis, wgt, *, xp=np):
    """Divide a file's vis/weight by the per-antenna gain product (``gain_xds.GAIN``).

    Maps each baseline to its two antenna gains the same way :func:`itrs_baselines` maps
    antennas, then delegates to :func:`kremetart.utils.gains.apply_inverse_gains`. The gain
    snapshot is per-file (time-independent), so this runs once before the sub-integration loop.
    """
    from kremetart.utils.gains import apply_inverse_gains

    antenna = node["antenna_xds"].to_dataset(inherit=False)
    index = {name: i for i, name in enumerate(antenna.antenna_name.values)}
    a1 = np.array([index[n] for n in node.ds.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in node.ds.baseline_antenna2_name.values])
    gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values
    return apply_inverse_gains(vis, wgt, gains, a1, a2, xp=xp)
```

- [ ] **Step 4: Add `correct_gains` to `frame_dirty_maps`**

In `src/kremetart/core/smoovie.py`, change the signature line:

```python
def frame_dirty_maps(hdf_paths, nside: int, *, xp=np):
```

to:

```python
def frame_dirty_maps(hdf_paths, nside: int, *, correct_gains: bool = False, xp=np):
```

Then, inside the `for path in hdf_paths:` loop, immediately **after** the line
`freqs = np.asarray(main.frequency.values)` and **before** `for k in range(times.size):`, insert:

```python
        if correct_gains:
            vis, wgt = _correct_file_gains(node, vis, wgt, xp=xp)
```

Also update the `frame_dirty_maps` docstring's `Args:` block to add:

```
        correct_gains: divide vis/weights by the per-antenna gain product before imaging.
```

- [ ] **Step 5: Run to verify the new tests pass and nothing regressed**

Run: `uv run pytest tests/test_smoovie_core.py -v`
Expected: all pass (including the two new tests and the existing ones). If data is absent they skip — acceptable.

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/core/smoovie.py tests/test_smoovie_core.py
git commit -m "feat: apply inverse gains in smoovie frame_dirty_maps"
```

---

## Task 3: `utils/satellites.py` — catalogue fetch + ICRS tracks

**Files:**
- Create: `src/kremetart/utils/satellites.py`
- Test: `tests/test_satellites.py`

The network fetch is injected via a `fetch` callable so tests run with a canned catalogue (no network). The `satellite_tracks` ordering must match `frame_dirty_maps` exactly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_satellites.py`:

```python
"""Tests for satellite track assembly (catalogue fetch is injected; no network)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")
pytest.importorskip("xarray_ms")  # read_hdf_as_msv4 path uses MSv4 machinery

_DATA = Path(__file__).parent / "data"


def _hdfs():
    paths = sorted(_DATA.glob("*.hdf"))
    if not paths:
        pytest.skip("no test HDFs present")
    return paths


def test_satellite_tracks_align_with_frame_order():
    from kremetart.core.smoovie import _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:2]
    expected_frames = sum(int(_partition(read_hdf_as_msv4(p)).ds.time.size) for p in paths)

    calls = {"n": 0}

    def fake_fetch(lon, lat, datestr, elevation_deg):
        calls["n"] += 1
        return [{"name": "SAT-A", "az": 0.0, "el": 90.0, "jy": 1.0, "r": 7.0e6}]

    tracks = satellite_tracks(paths, 45.0, fetch=fake_fetch)

    assert calls["n"] == expected_frames  # one query per frame, in frame order
    assert set(tracks) == {"SAT-A"}
    points = tracks["SAT-A"]
    assert len(points) == expected_frames
    assert [p[0] for p in points] == list(range(expected_frames))  # frame indices 0..N-1


def test_satellite_tracks_radec_matches_astropy():
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    from kremetart.core.smoovie import _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:1]
    main = _partition(read_hdf_as_msv4(paths[0])).ds
    info = main.attrs["observation_info"]
    t0 = float(np.asarray(main.time.values)[0])

    def fake_fetch(lon, lat, datestr, elevation_deg):
        return [{"name": "SAT-A", "az": 30.0, "el": 60.0, "jy": 1.0, "r": 7.0e6}]

    tracks = satellite_tracks(paths, 45.0, fetch=fake_fetch)

    loc = EarthLocation(
        lat=info["site_latitude_deg"] * u.deg,
        lon=info["site_longitude_deg"] * u.deg,
        height=info["site_altitude_m"] * u.m,
    )
    ref = SkyCoord(
        AltAz(az=30.0 * u.deg, alt=60.0 * u.deg, obstime=Time(t0, format="unix", scale="utc"), location=loc)
    ).icrs

    frame, ra, dec, jy = tracks["SAT-A"][0]
    assert frame == 0
    assert abs(ra - float(ref.ra.deg)) < 1e-6
    assert abs(dec - float(ref.dec.deg)) < 1e-6
    assert jy == 1.0


def test_satellite_tracks_skips_empty_frames():
    from kremetart.utils.satellites import satellite_tracks

    paths = _hdfs()[:1]

    def fake_fetch(lon, lat, datestr, elevation_deg):
        return []  # nothing above the cutoff

    tracks = satellite_tracks(paths, 89.0, fetch=fake_fetch)
    assert tracks == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_satellites.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kremetart.utils.satellites'`.

- [ ] **Step 3: Write the implementation**

Create `src/kremetart/utils/satellites.py`:

```python
"""Fetch TART catalogue satellite positions and project them into ICRS tracks.

The TART catalogue API returns, for a site ``(lon, lat)`` and a UTC datestr, the list of
sources above an elevation cutoff, each a dict with ``name``/``az``/``el``/``jy``/``r``.
:func:`satellite_tracks` queries it once per smoovie frame (one per sub-integration, in the
same order as :func:`kremetart.core.smoovie.frame_dirty_maps`), converts each ``(az, el)`` to
ICRS ``(ra, dec)`` at that timestamp, and groups the results by satellite name so the renderer
can draw a per-satellite track.

Network access lives here, isolated from the hermetic MSv4 reader. The ``fetch`` callable is
injectable so tests can supply a canned catalogue.
"""

from __future__ import annotations

import datetime

import numpy as np


def _tart_api_fetch(lon, lat, datestr, elevation_deg):
    """Return the catalogue source list for a site at a UTC datestr (network).

    Args:
        lon: site longitude (deg).
        lat: site latitude (deg).
        datestr: ISO-8601 UTC timestamp string.
        elevation_deg: elevation cutoff (deg).

    Returns:
        A list of source dicts, each with ``name``/``az``/``el``/``jy``/``r``.

    Raises:
        RuntimeError: if the catalogue cannot be fetched after several retries.
    """
    from tart_tools import api_handler

    api = api_handler.APIhandler("")
    url = api.catalog_url(lon, lat, datestr=datestr) + f"&elevation={elevation_deg}"
    nretry = 5
    for retry in range(nretry):
        try:
            return api.get_url(url)
        except Exception as exc:  # noqa: BLE001 -- retry any transient API/network error
            print(f"Error fetching catalog (attempt {retry + 1}/{nretry}): {exc}")
    raise RuntimeError(f"Failed to fetch catalog after {nretry} attempts.")


def _frame_times_and_site(hdf_paths):
    """Per-frame unix timestamps and the shared site, in frame_dirty_maps order.

    Returns:
        ``(times_unix, lat_deg, lon_deg, alt_m)`` where ``times_unix`` is a ``(n_frame,)`` array
        covering every sub-integration of every file, in order.

    Raises:
        ValueError: if ``hdf_paths`` is empty.
    """
    from kremetart.core.smoovie import _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    times_unix: list[float] = []
    info = None
    for path in hdf_paths:
        main = _partition(read_hdf_as_msv4(path)).ds
        times_unix.extend(float(t) for t in np.asarray(main.time.values))
        if info is None:
            info = main.attrs["observation_info"]
    if info is None:
        raise ValueError("no HDF files provided")
    return (
        np.asarray(times_unix),
        info["site_latitude_deg"],
        info["site_longitude_deg"],
        info["site_altitude_m"],
    )


def satellite_tracks(hdf_paths, elevation_deg, *, fetch=_tart_api_fetch):
    """Per-satellite ICRS tracks aligned 1:1 with the smoovie frame sequence.

    Iterates the same ordering as :func:`kremetart.core.smoovie.frame_dirty_maps`, so the global
    frame index produced here matches the dirty-map index exactly.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (same order as ``frame_dirty_maps``).
        elevation_deg: elevation cutoff (deg) for catalogue sources.
        fetch: ``callable(lon, lat, datestr, elevation_deg) -> list[dict]``; injectable so tests
            avoid the network. Defaults to :func:`_tart_api_fetch`.

    Returns:
        ``dict`` mapping satellite name -> list of ``(frame_index, ra_deg, dec_deg, flux_jy)``,
        sorted by ``frame_index``. Satellites absent from a frame simply have no point there.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    times_unix, lat, lon, alt = _frame_times_and_site(hdf_paths)
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)

    tracks: dict[str, list] = {}
    for i, t in enumerate(times_unix):
        datestr = datetime.datetime.fromtimestamp(float(t), tz=datetime.timezone.utc).isoformat()
        sources = fetch(lon, lat, datestr, elevation_deg)
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

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest tests/test_satellites.py -v`
Expected: 3 passed (or skipped if test data absent).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/satellites.py tests/test_satellites.py
git commit -m "feat: build ICRS satellite tracks from the TART catalogue"
```

---

## Task 4: overlay tracks in `render_frames`

**Files:**
- Modify: `src/kremetart/core/smoovie.py` (add `_overlay_tracks`; add `tracks` param to `render_frames`)
- Test: `tests/test_smoovie_core.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_smoovie_core.py`:

```python
def test_render_frames_overlays_tracks(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import healpy as hp

    import kremetart.core.smoovie as sm

    calls = {"scatter": 0, "plot": 0, "text": 0}
    monkeypatch.setattr(hp, "projscatter", lambda *a, **k: calls.__setitem__("scatter", calls["scatter"] + 1))
    monkeypatch.setattr(hp, "projplot", lambda *a, **k: calls.__setitem__("plot", calls["plot"] + 1))
    monkeypatch.setattr(hp, "projtext", lambda *a, **k: calls.__setitem__("text", calls["text"] + 1))

    nside = 8
    npix = 12 * nside * nside
    maps = [np.arange(npix, dtype=float), np.arange(npix, dtype=float) + 1.0]
    stamps = ["t0 UTC", "t1 UTC"]
    # SAT-A is present in both frames; the trailing line only appears once there are >1 past points.
    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.0), (1, 12.0, -19.0, 1.0)]}

    pngs = sm.render_frames(maps, stamps, nside, "inferno", tmp_path, rot=(0.0, -30.0), tracks=tracks)

    assert len(pngs) == 2
    assert calls["scatter"] == 2  # one marker per frame
    assert calls["text"] == 2  # one label per frame
    assert calls["plot"] == 1  # trail drawn only on frame 1 (needs >1 past point)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_smoovie_core.py::test_render_frames_overlays_tracks -v`
Expected: FAIL — `render_frames() got an unexpected keyword argument 'tracks'`.

- [ ] **Step 3: Add the `_overlay_tracks` helper**

In `src/kremetart/core/smoovie.py`, add this function immediately **above** `def render_frames(`:

```python
def _overlay_tracks(hp, tracks, frame_index):
    """Draw each satellite present in ``frame_index``: trailing line, current marker, name label.

    ``tracks`` maps name -> list of ``(frame_index, ra_deg, dec_deg, flux_jy)``. Coordinates are
    plotted with ``lonlat=True`` (degrees, ``lon == RA``) so healpy applies the active Mollweide
    ``rot`` and the overlay lands in the same projected ICRS frame as the imaged pixels.
    """
    for name, points in tracks.items():
        trail = [(ra, dec) for (f, ra, dec, _jy) in points if f <= frame_index]
        current = [(ra, dec) for (f, ra, dec, _jy) in points if f == frame_index]
        if not current:
            continue  # satellite not above the cutoff in this frame
        if len(trail) > 1:
            hp.projplot(
                [ra for ra, _ in trail],
                [dec for _, dec in trail],
                lonlat=True,
                color="cyan",
                linewidth=0.7,
                alpha=0.6,
            )
        ra0, dec0 = current[0]
        hp.projscatter([ra0], [dec0], lonlat=True, color="cyan", marker="x", s=30)
        hp.projtext(ra0, dec0, name, lonlat=True, color="cyan", fontsize=6)
```

- [ ] **Step 4: Add the `tracks` param to `render_frames`**

In `src/kremetart/core/smoovie.py`, change the `render_frames` signature:

```python
def render_frames(
    maps, timestamps, nside: int, cmap: str, outdir, *, rot: tuple[float, float] | None = None, nest: bool = True
):
```

to:

```python
def render_frames(
    maps,
    timestamps,
    nside: int,
    cmap: str,
    outdir,
    *,
    rot: tuple[float, float] | None = None,
    nest: bool = True,
    tracks=None,
):
```

Then, inside the `for i, (m, ts) in enumerate(zip(maps, timestamps)):` loop, immediately **after**
the `hp.graticule()` line and **before** `out = outdir / f"frame_{i:04d}.png"`, insert:

```python
        if tracks:
            _overlay_tracks(hp, tracks, i)
```

Update the `render_frames` docstring to note: ``tracks`` (if given) overlays per-satellite ICRS
trajectories (trailing line + current marker + name label) on each frame.

- [ ] **Step 5: Run to verify the test passes**

Run: `uv run pytest tests/test_smoovie_core.py::test_render_frames_overlays_tracks -v`
Expected: PASS.

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/core/smoovie.py tests/test_smoovie_core.py
git commit -m "feat: overlay satellite tracks in smoovie render_frames"
```

---

## Task 5: wire new options into `smoovie(...)`

**Files:**
- Modify: `src/kremetart/core/smoovie.py` (`smoovie` signature + body)
- Test: `tests/test_smoovie_core.py` (new wiring test; update two existing `fake_render` signatures)

> **Why the existing tests change:** `smoovie` will now call `render_frames(..., tracks=tracks)`.
> The two existing tests (`test_smoovie_honors_explicit_phase_direction`,
> `test_smoovie_auto_phase_direction_used`) monkeypatch `render_frames` with a `fake_render` whose
> signature lacks `tracks`, so they would raise `TypeError` unless updated.

- [ ] **Step 1: Update the two existing `fake_render` signatures**

In `tests/test_smoovie_core.py`, both occurrences currently read:

```python
    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True):
        captured["rot"] = rot
        return [Path("frame_0000.png")]
```

Change **both** to add `tracks=None`:

```python
    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["rot"] = rot
        return [Path("frame_0000.png")]
```

- [ ] **Step 2: Write the failing wiring test**

Add to `tests/test_smoovie_core.py`:

```python
def test_smoovie_overlay_passes_tracks(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm
    import kremetart.utils.satellites as sat

    _hdfs()  # need a non-empty glob; heavy steps are monkeypatched out
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))
    monkeypatch.setattr(sat, "satellite_tracks", lambda paths, elev, **k: {"SAT": [(0, 1.0, 2.0, 1.0)]})

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm, "render_frames", fake_render)

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1, overlay_catalog=True, catalog_elevation_deg=30.0)
    assert captured["tracks"] == {"SAT": [(0, 1.0, 2.0, 1.0)]}


def test_smoovie_no_overlay_passes_none_tracks(tmp_path, monkeypatch):
    import kremetart.core.smoovie as sm

    _hdfs()
    captured = {}

    monkeypatch.setattr(
        sm, "frame_dirty_maps", lambda paths, nside, **k: ([np.zeros(12)], ["t UTC"], np.zeros((12, 3)))
    )
    monkeypatch.setattr(sm, "common_phase_direction", lambda paths: (0.0, 0.0))
    monkeypatch.setattr(sm, "encode_movie", lambda pngs, movie, fps: Path(movie))

    def fake_render(maps, stamps, nside, cmap, outdir, *, rot=None, nest=True, tracks=None):
        captured["tracks"] = tracks
        return [Path("frame_0000.png")]

    monkeypatch.setattr(sm, "render_frames", fake_render)

    sm.smoovie(hdf_dir=_DATA, movie=tmp_path / "m.mp4", nside=1)
    assert captured["tracks"] is None
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_smoovie_core.py::test_smoovie_overlay_passes_tracks -v`
Expected: FAIL — `smoovie() got an unexpected keyword argument 'overlay_catalog'`.

- [ ] **Step 4: Update the `smoovie` signature and body**

In `src/kremetart/core/smoovie.py`, change the signature:

```python
def smoovie(
    hdf_dir,
    movie,
    nside: int = 128,
    fps: int = 2,
    cmap: str = "inferno",
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
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
):
```

In the body, change:

```python
    maps, stamps, _ = frame_dirty_maps(hdf_paths, nside)
    with tempfile.TemporaryDirectory() as td:
        pngs = render_frames(maps, stamps, nside, cmap, Path(td), rot=(phase_ra_deg, phase_dec_deg))
        encode_movie(pngs, movie, fps)
    return movie
```

to:

```python
    maps, stamps, _ = frame_dirty_maps(hdf_paths, nside, correct_gains=correct_gains)
    tracks = None
    if overlay_catalog:
        from kremetart.utils.satellites import satellite_tracks

        tracks = satellite_tracks(hdf_paths, catalog_elevation_deg)
    with tempfile.TemporaryDirectory() as td:
        pngs = render_frames(maps, stamps, nside, cmap, Path(td), rot=(phase_ra_deg, phase_dec_deg), tracks=tracks)
        encode_movie(pngs, movie, fps)
    return movie
```

Also extend the `smoovie` docstring to document the three new parameters (`correct_gains`,
`overlay_catalog`, `catalog_elevation_deg`), e.g.:

```
    ``correct_gains`` divides the visibilities by the per-antenna gain product (TART's own
    solution) before imaging. ``overlay_catalog`` overlays each catalogue satellite above
    ``catalog_elevation_deg`` (degrees) as a trailing track + marker + label on every frame; it
    requires network access to the TART catalogue API.
```

- [ ] **Step 5: Run the full smoovie core suite**

Run: `uv run pytest tests/test_smoovie_core.py -v`
Expected: all pass (new wiring tests + the two updated monkeypatch tests + the rest).

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/core/smoovie.py tests/test_smoovie_core.py
git commit -m "feat: thread gain-correction and catalog-overlay options through smoovie"
```

---

## Task 6: CLI params + cab regeneration + round-trip

**Files:**
- Modify: `src/kremetart/cli/smoovie.py` (3 new params; thread through 3 dicts)
- Regenerated (do NOT hand-edit): `src/kremetart/cabs/smoovie.yml`
- Test: `tests/test_roundtrip.py::test_roundtrip_smoovie` (already exists — must still pass)

> **Round-trip rule:** `cli/smoovie.py` must be in the exact canonical shape `hip-cargo
> generate-function` emits, or `test_roundtrip_smoovie` fails on a byte diff. The new blocks below
> mirror the existing canonical param blocks (e.g. `phase_ra_deg` for float, `always_pull_images`
> for bool) so they round-trip. The cab is regenerated by the pre-commit hook, never hand-edited.

- [ ] **Step 1: Add the three parameters to the signature**

In `src/kremetart/cli/smoovie.py`, the signature currently has the `phase_dec_deg` block ending with
`] = None,` immediately followed by the `movie:` block. Insert these three blocks **between**
`phase_dec_deg`'s `] = None,` line and the `movie:` block:

```python
    correct_gains: Annotated[
        bool,
        typer.Option(
            help="Apply the inverse per-antenna gain solution before imaging.",
        ),
    ] = False,
    overlay_catalog: Annotated[
        bool,
        typer.Option(
            help="Overlay TART catalog satellite tracks on each frame (requires network).",
        ),
    ] = False,
    catalog_elevation_deg: Annotated[
        float,
        typer.Option(
            help="Elevation cutoff (deg) for catalog sources to overlay.",
        ),
    ] = 45.0,
```

- [ ] **Step 2: Thread the params through the preflight dict**

In `src/kremetart/cli/smoovie.py`, the `preflight_remote_must_exist(smoovie, dict(...))` call lists
`hdf_dir=..., nside=..., fps=..., cmap=..., phase_ra_deg=..., phase_dec_deg=..., movie=...`. Add the
three new keys (before `movie=movie,` to keep inputs grouped):

```python
                    correct_gains=correct_gains,
                    overlay_catalog=overlay_catalog,
                    catalog_elevation_deg=catalog_elevation_deg,
```

- [ ] **Step 3: Thread the params through the core call**

In the `smoovie_core(...)` call, the argument list currently ends `phase_dec_deg=phase_dec_deg,
movie=movie,`. Add the three keys before `movie=movie,`:

```python
                correct_gains=correct_gains,
                overlay_catalog=overlay_catalog,
                catalog_elevation_deg=catalog_elevation_deg,
```

- [ ] **Step 4: Thread the params through the `run_in_container` dict**

In the `run_in_container(smoovie, dict(...), ...)` call, add the three keys before `movie=movie,`:

```python
            correct_gains=correct_gains,
            overlay_catalog=overlay_catalog,
            catalog_elevation_deg=catalog_elevation_deg,
```

- [ ] **Step 5: Format and regenerate the cab**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run hip-cargo generate-cabs --module src/kremetart/cli/smoovie.py --output-dir src/kremetart/cabs
```

Confirm `src/kremetart/cabs/smoovie.yml` now lists `correct-gains`, `overlay-catalog`, and
`catalog-elevation-deg` under `inputs:` and still has an `image:` field (run inside the project venv
so package metadata resolves). **Do not hand-edit the YAML.**

- [ ] **Step 6: Run the round-trip test**

Run: `uv run pytest tests/test_roundtrip.py::test_roundtrip_smoovie -v`
Expected: PASS (regenerated source byte-identical to `cli/smoovie.py`).

If it fails on a line diff, the failure message prints the exact differing line. Make
`cli/smoovie.py` match the **Generated** form shown (that is the canonical shape), then re-run.

- [ ] **Step 7: Commit**

```bash
git add src/kremetart/cli/smoovie.py src/kremetart/cabs/smoovie.yml
git commit -m "feat: expose gain-correction and catalog-overlay on smoovie CLI"
```

---

## Task 7: opt-in `.ms` oracle — corrected vis vs calibrated MS

**Files:**
- Test: `tests/test_smoovie_core.py`

This cross-checks our correction against `tart2ms`'s own calibrated Measurement Set. It is **opt-in**
(env-gated) because it couples to an external tool's calibration convention; a failure may indicate a
conjugate/scale convention difference to reconcile with the domain owner, not a bug in our wiring.
Excluded from required CI per `testing-and-ci.md` §2.

- [ ] **Step 1: Add the env-gated test**

Add to the top of `tests/test_smoovie_core.py` (with the other imports):

```python
import os
```

Then add this test:

```python
@pytest.mark.skipif(
    os.environ.get("KREMETART_MS_ORACLE") != "1",
    reason="opt-in: cross-checks tart2ms calibration convention (set KREMETART_MS_ORACLE=1)",
)
def test_corrected_vis_matches_calibrated_ms():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xarray_ms")  # registers the "xarray-ms:msv2" engine

    from kremetart.core.smoovie import _correct_file_gains, _partition
    from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

    hdf = _DATA / "vis_2026-06-09_08_11_43.476804.hdf"
    ms = _DATA / "vis_2026-06-09_08_11_43.476804.ms"  # calibrated (NOT _nocal)
    if not (hdf.exists() and ms.exists()):
        pytest.skip("HDF or calibrated MS not present")

    node = _partition(read_hdf_as_msv4(hdf))
    main = node.ds
    vis = np.asarray(main.VISIBILITY.values)[..., 0]
    wgt = np.asarray(main.WEIGHT.values)[..., 0]
    vis_c, _ = _correct_file_gains(node, vis, wgt)

    ref = _partition(xr.open_datatree(str(ms), engine="xarray-ms:msv2"))
    ref_vis = np.asarray(ref.ds.VISIBILITY.values)[..., 0]

    # Compare only weighted baselines (dead antennas are zeroed on our side). The MSv4 reader and
    # tart2ms share baseline ordering (see test_rephasing.py), so this is an element-wise compare.
    mask = wgt > 0
    np.testing.assert_allclose(vis_c[mask], ref_vis[mask], rtol=1e-2, atol=1e-3)
```

- [ ] **Step 2: Run it opt-in to verify convention (manual)**

Run: `KREMETART_MS_ORACLE=1 uv run pytest tests/test_smoovie_core.py::test_corrected_vis_matches_calibrated_ms -v`
Expected: PASS if our convention matches `tart2ms`. If it fails, capture the max residual and the
ratio `vis_c / ref_vis` over the masked baselines (a constant phase/scale points at a conjugate or
normalisation convention difference) and raise it with the domain owner before changing the formula.

- [ ] **Step 3: Confirm default suite skips it**

Run: `uv run pytest tests/test_smoovie_core.py -v`
Expected: `test_corrected_vis_matches_calibrated_ms` shows SKIPPED; everything else passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoovie_core.py
git commit -m "test: add opt-in oracle comparing corrected vis to calibrated MS"
```

---

## Final verification

- [ ] **Run the full test suite**

Run: `uv run pytest -v`
Expected: all pass or skip (no failures). Data-dependent tests skip cleanly if `tests/data/` is absent.

- [ ] **Lint/format clean**

Run: `uv run ruff format --check . && uv run ruff check .`
Expected: no changes needed, no errors.

- [ ] **Smoke-test both features end-to-end (manual, needs data + ffmpeg + network)**

Run:
```bash
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/corrected.mp4 \
    --nside 64 --correct-gains --overlay-catalog --catalog-elevation-deg 45 --backend native
```
Expected: `/tmp/corrected.mp4` is produced; satellites sit under their overlaid markers.

---

## Self-Review (completed during planning)

**Spec coverage:** Every spec section maps to a task — §3 gain helper → Task 1; §4.2 `frame_dirty_maps`
wiring → Task 2; §5.1 `satellite_tracks` → Task 3; §5.2 `render_frames` overlay → Task 4; §5.3
`smoovie` wiring → Task 5; §3 CLI params + round-trip → Task 6; §4.3/§7 `.ms` oracle → Task 7; §7 unit
tests distributed across Tasks 1, 3. §6 cost/container note needs no code.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output.

**Type/name consistency:** `apply_inverse_gains(vis, weight, gains, a1_idx, a2_idx, *, xp)` is defined
in Task 1 and called identically in Tasks 2 and 7. `_correct_file_gains(node, vis, wgt, *, xp)` defined
in Task 2, reused in Task 7. `satellite_tracks(hdf_paths, elevation_deg, *, fetch)` defined in Task 3,
called in Task 5. `render_frames(..., tracks=None)` defined in Task 4, called in Task 5. Track tuple
shape `(frame_index, ra_deg, dec_deg, flux_jy)` is consistent across Tasks 3, 4, 5. CLI param names
(`correct_gains`, `overlay_catalog`, `catalog_elevation_deg`) match the core signature in Task 5.
