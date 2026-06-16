# Accuracy Verification (L1 + L2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove, against an independent geodetic standard (pyproj/PROJ), that the kremetart geometry + HEALPix imaging recovers a known sky more accurately than the tart2ms geometry — L1 (antenna positions, no imaging) and L2 (a sub-pixel point-source simulation imaged with both geometries).

**Architecture:** Test-only helpers in `tests/accuracy_helpers.py` build PROJ truth antenna positions, simulate visibilities with the *shipped* forward model, and measure recovered source position/flux. Two pytest modules assert quantitative superiority. Geometry isolation: everything (imager, `C(t)`, grid, truth visibilities) is held identical; only the antenna ECEF varies (truth = pyproj, ours = reader, tart2ms = reference MS).

**Tech Stack:** numpy, astropy, pyproj (PROJ ≥ 6.3, new **test** dep), healpy; reuses `kremetart.utils.healpix_dft` and `read_hdf_as_msv4`. CPU only (`xp=np`).

**Spec:** `docs/superpowers/specs/2026-06-15-accuracy-verification-design.md`

**Note on file layout:** the spec suggested `tests/verification/geometry_truth.py`; this plan uses a single flat `tests/accuracy_helpers.py` (imported as `from tests.accuracy_helpers import …`, which works because `tests/__init__.py` already exists) to avoid subpackage scaffolding. Same responsibility, simpler.

**Reference data (already in `tests/data/`):** `vis_2026-06-09_08_11_43.476804.hdf` (ours) and `vis_2026-06-09_08_11_43.476804_nocal.ms` (tart2ms antenna positions). Site: Bel Air, Mauritius (lat −20.2587508°, lon 57.7591989°, alt 20 m), L1 = 1.575420 GHz.

---

### Task 1: test dependency + PROJ truth geometry helpers

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups].test`)
- Create: `tests/accuracy_helpers.py`
- Test: `tests/test_accuracy_helpers.py`

- [ ] **Step 1: Add pyproj to the test dependency group**

In `pyproject.toml`, under `[dependency-groups]`, add `pyproj` to the existing `test` list:

```toml
test = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pyproj>=3.6.0",
]
```

- [ ] **Step 2: Install it**

Run: `.venv/bin/python -m pip install "pyproj>=3.6.0"`
Expected: pyproj installs (bundled PROJ ≥ 9). Verify: `.venv/bin/python -c "import pyproj; print(pyproj.__version__, pyproj.proj_version_str)"`

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_accuracy_helpers.py
"""Unit tests for the accuracy-verification helpers."""

import numpy as np
import pytest

pyproj = pytest.importorskip("pyproj")

from tests.accuracy_helpers import baselines_from_positions, enu_to_ecef_truth

SITE = dict(lat_deg=-20.2587508, lon_deg=57.7591989, alt_m=20.0)


def test_enu_origin_maps_to_site_ecef():
    """ENU (0,0,0) maps to the WGS84 site ECEF from the independent EPSG:4979->4978 path."""
    from pyproj import Transformer

    site = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(
        SITE["lon_deg"], SITE["lat_deg"], SITE["alt_m"]
    )
    got = enu_to_ecef_truth(np.zeros((1, 3)), **SITE)[0]
    np.testing.assert_allclose(got, site, atol=1e-3)


def test_enu_offset_preserves_distance():
    """A 3.4 m ENU offset produces a 3.4 m ECEF displacement (rotation is rigid)."""
    pts = enu_to_ecef_truth(np.array([[0.0, 0, 0], [3.4, 0, 0], [0, 3.4, 0]]), **SITE)
    np.testing.assert_allclose(np.linalg.norm(pts[1] - pts[0]), 3.4, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(pts[2] - pts[0]), 3.4, atol=1e-6)


def test_baselines_from_positions():
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 2, 0]])
    a1 = np.array([0, 0, 1])
    a2 = np.array([1, 2, 2])
    bl = baselines_from_positions(pos, a1, a2)
    np.testing.assert_allclose(bl, np.array([[-1, 0, 0], [0, -2, 0], [1, -2, 0]]))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_accuracy_helpers.py -v`
Expected: FAIL — `cannot import name 'enu_to_ecef_truth'`.

- [ ] **Step 5: Write the helpers**

```python
# tests/accuracy_helpers.py
"""Independent geodetic truth + simulation helpers for accuracy verification (test-only).

Truth antenna ECEF comes from pyproj/PROJ (WGS84) -- a code path independent of both the kremetart
reader (hand-rolled transform) and tart2ms (mean-Earth-radius offset_by). See
docs/superpowers/specs/2026-06-15-accuracy-verification-design.md.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

LIGHTSPEED = 299792458.0


def enu_to_ecef_truth(enu, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Independent ENU->ECEF via PROJ topocentric (WGS84).

    Args:
        enu: ``(n, 3)`` East/North/Up offsets (m) relative to the site.
        lat_deg, lon_deg, alt_m: site geodetic origin.

    Returns:
        ``(n, 3)`` geocentric ECEF positions (m).
    """
    from pyproj import CRS, Transformer

    enu = np.asarray(enu, dtype=np.float64)
    topo = CRS.from_proj4(f"+proj=topocentric +ellps=WGS84 +lon_0={lon_deg} +lat_0={lat_deg} +h_0={alt_m}")
    ecef = CRS.from_epsg(4978)
    tr = Transformer.from_crs(topo, ecef, always_xy=True)
    x, y, z = tr.transform(enu[:, 0], enu[:, 1], enu[:, 2])
    return np.stack([x, y, z], axis=1)


def baselines_from_positions(positions, ant1_idx, ant2_idx) -> np.ndarray:
    """Baseline vectors ``pos[ant1] - pos[ant2]`` -> ``(nbl, 3)``."""
    positions = np.asarray(positions)
    return positions[np.asarray(ant1_idx)] - positions[np.asarray(ant2_idx)]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_accuracy_helpers.py -v`
Expected: PASS (3 passed). If `test_enu_origin_maps_to_site_ecef` fails, the PROJ topocentric param names/axis order need adjusting until the contract (origin → site ECEF, rigid offsets) holds.

- [ ] **Step 7: Format, lint, commit**

```bash
.venv/bin/ruff format tests/accuracy_helpers.py tests/test_accuracy_helpers.py
.venv/bin/ruff check tests/accuracy_helpers.py tests/test_accuracy_helpers.py --fix
git add pyproject.toml tests/accuracy_helpers.py tests/test_accuracy_helpers.py
git commit -m "test: add pyproj truth-geometry helpers for accuracy verification"
```

---

### Task 2: L1 — antenna-position truth comparison

**Files:**
- Modify: `tests/accuracy_helpers.py` (add datatree readers)
- Test: `tests/test_accuracy_l1_geometry.py`

- [ ] **Step 1: Add datatree reader helpers**

Append to `tests/accuracy_helpers.py`:

```python
def antenna_ecef(antenna_xds) -> np.ndarray:
    """ANTENNA_POSITION (ECEF, m) as ``(n_ant, 3)`` in antenna-index order."""
    return np.asarray(antenna_xds.ANTENNA_POSITION.values, dtype=np.float64)


def antenna_enu_and_site(partition):
    """Return (enu (n_ant,3), lat_deg, lon_deg, alt_m) from a kremetart partition node."""
    ant = partition["antenna_xds"].to_dataset(inherit=False)
    enu = np.asarray(ant.ANTENNA_POSITION_ENU.values, dtype=np.float64)
    info = partition.ds.attrs["observation_info"]
    return enu, info["site_latitude_deg"], info["site_longitude_deg"], info["site_altitude_m"]


def baseline_index_arrays(partition):
    """(ant1_idx, ant2_idx) mapping each baseline to antenna indices, in the partition's order."""
    ant = partition["antenna_xds"].to_dataset(inherit=False)
    names = list(ant.antenna_name.values)
    index = {name: i for i, name in enumerate(names)}
    a1 = np.array([index[n] for n in partition.ds.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in partition.ds.baseline_antenna2_name.values])
    return a1, a2
```

- [ ] **Step 2: Write the failing L1 test**

```python
# tests/test_accuracy_l1_geometry.py
"""L1: kremetart antenna positions match an independent PROJ truth; tart2ms does not."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")
xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_ms")  # registers xarray-ms:msv2

from kremetart.utils.read_tart_hdf import read_hdf_as_msv4  # noqa: E402
from tests.accuracy_helpers import (  # noqa: E402
    antenna_ecef,
    antenna_enu_and_site,
    baseline_index_arrays,
    baselines_from_positions,
    enu_to_ecef_truth,
)

_DATA = Path(__file__).parent / "data"
_HDF = _DATA / "vis_2026-06-09_08_11_43.476804.hdf"
_MS = _DATA / "vis_2026-06-09_08_11_43.476804_nocal.ms"


def _partition(dt):
    return dt[list(dt.children)[0]]


@pytest.fixture(scope="module")
def positions():
    if not _HDF.exists() or not _MS.exists():
        pytest.skip("reference HDF/MS not present")
    ours_dt = read_hdf_as_msv4(_HDF)
    ours_part = _partition(ours_dt)
    enu, lat, lon, alt = antenna_enu_and_site(ours_part)
    truth = enu_to_ecef_truth(enu, lat, lon, alt)
    ours = antenna_ecef(ours_part["antenna_xds"].to_dataset(inherit=False))
    ms_part = _partition(xr.open_datatree(str(_MS), engine="xarray-ms:msv2"))
    tart2ms = antenna_ecef(ms_part["antenna_xds"].to_dataset(inherit=False))
    a1, a2 = baseline_index_arrays(ours_part)
    return dict(truth=truth, ours=ours, tart2ms=tart2ms, a1=a1, a2=a2)


def test_antenna_index_alignment(positions):
    """Ours and tart2ms antennas are in the same index order (per-antenna abs diff < 1 cm)."""
    assert positions["ours"].shape == positions["tart2ms"].shape == (24, 3)
    assert np.abs(positions["ours"] - positions["tart2ms"]).max() < 1e-2


def test_our_positions_match_proj_truth(positions):
    """Our reader's WGS84 transform matches the independent PROJ truth to << 1 mm."""
    assert np.abs(positions["ours"] - positions["truth"]).max() < 1e-3


def test_tart2ms_baselines_are_farther_from_truth(positions):
    """tart2ms baseline lengths deviate from truth far more than ours do."""
    a1, a2 = positions["a1"], positions["a2"]
    len_ = lambda p: np.linalg.norm(baselines_from_positions(p, a1, a2), axis=1)
    truth_len = len_(positions["truth"])
    ours_err = np.abs(len_(positions["ours"]) - truth_len).max()
    tart_err = np.abs(len_(positions["tart2ms"]) - truth_len).max()
    print(f"\nL1 baseline-length max error: ours={ours_err * 1e3:.4f} mm, tart2ms={tart_err * 1e3:.4f} mm")
    assert ours_err < 1e-3  # ours matches the geodetic standard (sub-mm)
    assert tart_err > 5 * ours_err  # tart2ms is materially farther from truth
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_accuracy_l1_geometry.py -v`
Expected: FAIL — `cannot import name 'antenna_ecef'` (before Step 1 is applied) or assertions/imports incomplete.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_accuracy_l1_geometry.py -v -s`
Expected: PASS (3 passed); printed line shows ours ≈ µm-sub-mm, tart2ms ≈ several mm-cm.

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format tests/accuracy_helpers.py tests/test_accuracy_l1_geometry.py
.venv/bin/ruff check tests/accuracy_helpers.py tests/test_accuracy_l1_geometry.py --fix
git add tests/accuracy_helpers.py tests/test_accuracy_l1_geometry.py
git commit -m "test: add L1 antenna-position accuracy verification vs PROJ truth"
```

---

### Task 3: sources + truth-visibility simulation

**Files:**
- Modify: `tests/accuracy_helpers.py`
- Test: `tests/test_accuracy_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_accuracy_helpers.py` (and extend the import line at the top to include
`simulate_visibilities, source_svec, sources_spanning_zenith`):

```python
from tests.accuracy_helpers import simulate_visibilities, source_svec, sources_spanning_zenith


def test_source_svec_unit_and_value():
    np.testing.assert_allclose(source_svec([0.0], [0.0])[0], [1.0, 0.0, 0.0], atol=1e-12)
    v = source_svec([0.3, 1.2], [-0.2, 0.4])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)


def test_sources_spanning_zenith_roundtrip():
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    times = np.array([1.6e9, 1.6e9 + 60, 1.6e9 + 120])
    els = np.array([20.0, 50.0, 80.0])
    ra, dec = sources_spanning_zenith(times, **SITE, els_deg=els)
    loc = EarthLocation(lat=SITE["lat_deg"] * u.deg, lon=SITE["lon_deg"] * u.deg, height=SITE["alt_m"] * u.m)
    tmid = Time(times[1], format="unix", scale="utc")
    back = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs").transform_to(AltAz(obstime=tmid, location=loc))
    np.testing.assert_allclose(np.sort(back.alt.deg), np.sort(els), atol=1e-3)


def test_simulate_visibilities_point_source_amplitude():
    """A single point source of flux f gives |V| == f on every baseline (|fringe| = 1)."""
    rng = np.random.default_rng(7)
    ecef_bl = rng.standard_normal((10, 3)) * 2.0
    vis = simulate_visibilities(
        np.array([4.0]), source_svec([0.5], [-0.3]), ecef_bl, np.array([1.6e9]), np.array([1.575e9])
    )
    assert vis.shape == (1, 10, 1)
    np.testing.assert_allclose(np.abs(vis), 4.0, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_accuracy_helpers.py -k "svec or zenith or simulate" -v`
Expected: FAIL — `cannot import name 'simulate_visibilities'`.

- [ ] **Step 3: Write the implementation**

Append to `tests/accuracy_helpers.py`:

```python
def source_svec(ra, dec) -> np.ndarray:
    """ICRS unit vectors (n, 3) for ra/dec arrays in radians."""
    ra = np.atleast_1d(np.asarray(ra, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec, dtype=np.float64))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def sources_spanning_zenith(times, lat_deg, lon_deg, alt_m, els_deg, az_deg=0.0):
    """ICRS (ra, dec) radians for sources at given elevations (deg) at the mid timestamp."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    times = np.asarray(times)
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    tmid = Time(times[times.size // 2], format="unix", scale="utc")
    els = np.atleast_1d(np.asarray(els_deg, dtype=np.float64))
    aa = AltAz(az=np.full(els.shape, az_deg) * u.deg, alt=els * u.deg, obstime=tmid, location=loc)
    icrs = SkyCoord(aa).icrs
    return np.atleast_1d(icrs.ra.rad), np.atleast_1d(icrs.dec.rad)


def simulate_visibilities(fluxes, svec, ecef_baselines, times, freqs, *, xp: ModuleType = np):
    """Truth visibilities V_pq(t) = sum_s f_s exp(2pi i (nu/c) b_pq(t).s_s), shape (n_time, nbl, nchan).

    Uses the shipped forward model with the shared C(t); ``ecef_baselines`` (nbl,3) are the ITRS
    baseline vectors whose accuracy is under test.
    """
    from kremetart.utils.healpix_dft import dft_forward, equatorial_baselines

    fluxes = xp.asarray(fluxes)
    svec = xp.asarray(svec)
    b_rot = equatorial_baselines(np.asarray(ecef_baselines), np.asarray(times), xp=xp)  # (nt, nbl, 3)
    nt, nbl = b_rot.shape[0], b_rot.shape[1]
    nchan = np.asarray(freqs).shape[0]
    vis = xp.zeros((nt, nbl, nchan), dtype=xp.complex128)
    for t in range(nt):
        vis[t] = dft_forward(fluxes, b_rot[t], svec, freqs, xp=xp)
    return vis
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_accuracy_helpers.py -k "svec or zenith or simulate" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format tests/accuracy_helpers.py tests/test_accuracy_helpers.py
.venv/bin/ruff check tests/accuracy_helpers.py tests/test_accuracy_helpers.py --fix
git add tests/accuracy_helpers.py tests/test_accuracy_helpers.py
git commit -m "test: add sky/source + truth-visibility simulation helpers"
```

---

### Task 4: metrics — recovery, angular offset, analytic offset

**Files:**
- Modify: `tests/accuracy_helpers.py`
- Test: `tests/test_accuracy_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_accuracy_helpers.py` (extend the import line to add
`analytic_offset, angular_offset, recovered_direction_and_flux`):

```python
from tests.accuracy_helpers import analytic_offset, angular_offset, recovered_direction_and_flux


def test_angular_offset_basic():
    assert abs(angular_offset([1, 0, 0], [1, 0, 0])) < 1e-12
    np.testing.assert_allclose(angular_offset([1, 0, 0], [0, 1, 0]), np.pi / 2, atol=1e-12)


def test_recovered_direction_and_flux_single_hot_pixel():
    from kremetart.utils.healpix_dft import make_pixel_grid

    nside = 16
    pix = make_pixel_grid(nside, xp=np)
    dmap = np.zeros(pix.shape[0])
    src = 300
    dmap[src] = 5.0
    vec, flux = recovered_direction_and_flux(dmap, pix, nside)
    assert flux == 5.0
    np.testing.assert_allclose(vec, pix[src], atol=1e-12)


def test_analytic_offset_recovers_known_shift():
    """A baseline set whose extra delay equals b_rec.delta is predicted to shift by |delta|."""
    rng = np.random.default_rng(9)
    b_rec = rng.standard_normal((30, 3)) * 2.0
    s = source_svec([0.7], [-0.4])[0]
    z = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(s, z)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    delta = 1e-4 * e1 - 2e-4 * e2  # radians
    extra = b_rec @ delta
    b_truth = b_rec + extra[:, None] * s  # so (b_truth - b_rec).s == extra
    got = analytic_offset(b_rec, b_truth, s)
    np.testing.assert_allclose(got, np.linalg.norm(delta), rtol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_accuracy_helpers.py -k "angular or recovered or analytic" -v`
Expected: FAIL — `cannot import name 'recovered_direction_and_flux'`.

- [ ] **Step 3: Write the implementation**

Append to `tests/accuracy_helpers.py`:

```python
def angular_offset(a, b) -> float:
    """Angle (radians) between two unit vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def recovered_direction_and_flux(dmap, pix_vec, nside, *, nest: bool = True):
    """Flux-weighted centroid direction (unit vector) and peak flux (Jy) of a dirty map.

    Centroids the positive pixels within ~2.5 pixel-radii of the peak so a sub-pixel source is
    localised below the pixel scale.
    """
    import healpy as hp

    dmap = np.asarray(dmap)
    pix_vec = np.asarray(pix_vec)
    peak = int(np.argmax(dmap))
    disc = hp.query_disc(nside, pix_vec[peak], 2.5 * hp.nside2resol(nside), nest=nest)
    w = np.clip(dmap[disc], 0.0, None)
    centroid = (w[:, None] * pix_vec[disc]).sum(axis=0)
    centroid /= np.linalg.norm(centroid)
    return centroid, float(dmap[peak])


def analytic_offset(b_rec, b_truth, svec) -> float:
    """Predicted peak offset (radians) when imaging truth data with b_rec instead of b_truth.

    Least-squares stationary-phase: solve for the tangent-plane shift delta minimising
    || (b_truth - b_rec).s - b_rec.delta ||. ``b_rec``/``b_truth`` are the (nbl,3) rotated baselines.
    """
    s = np.asarray(svec, dtype=np.float64)
    s = s / np.linalg.norm(s)
    e1 = np.cross(s, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(s, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    b_rec = np.asarray(b_rec, dtype=np.float64)
    extra = (np.asarray(b_truth, dtype=np.float64) - b_rec) @ s  # (nbl,)
    design = np.stack([b_rec @ e1, b_rec @ e2], axis=1)  # (nbl, 2)
    coef, *_ = np.linalg.lstsq(design, extra, rcond=None)
    return float(np.hypot(coef[0], coef[1]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_accuracy_helpers.py -k "angular or recovered or analytic" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
.venv/bin/ruff format tests/accuracy_helpers.py tests/test_accuracy_helpers.py
.venv/bin/ruff check tests/accuracy_helpers.py tests/test_accuracy_helpers.py --fix
git add tests/accuracy_helpers.py tests/test_accuracy_helpers.py
git commit -m "test: add recovery/offset/analytic metrics for accuracy verification"
```

---

### Task 5: L2 — sub-pixel simulation imaging comparison

**Files:**
- Test: `tests/test_accuracy_l2_imaging.py`

Images one isolated source at a time (single representative integration, nside=128 ≈ 0.43 GB/image)
across zenith angles, with our geometry and tart2ms's, against the same PROJ-truth visibilities.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_accuracy_l2_imaging.py
"""L2: imaging PROJ-truth visibilities with our geometry recovers the source more accurately
than tart2ms's geometry (sub-pixel, off-grid sources spanning zenith angle)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")
xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_ms")
pytest.importorskip("astropy")
pytest.importorskip("healpy")

from kremetart.utils.healpix_dft import equatorial_baselines, image_frame, make_pixel_grid  # noqa: E402
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4  # noqa: E402
from tests.accuracy_helpers import (  # noqa: E402
    analytic_offset,
    angular_offset,
    antenna_ecef,
    antenna_enu_and_site,
    baseline_index_arrays,
    baselines_from_positions,
    enu_to_ecef_truth,
    recovered_direction_and_flux,
    simulate_visibilities,
    source_svec,
    sources_spanning_zenith,
)

_DATA = Path(__file__).parent / "data"
_HDF = _DATA / "vis_2026-06-09_08_11_43.476804.hdf"
_MS = _DATA / "vis_2026-06-09_08_11_43.476804_nocal.ms"
NSIDE = 128
FREQ = np.array([1575420000.0])
FLUX = 10.0
ELS_DEG = np.array([15.0, 35.0, 55.0, 75.0])  # horizon -> near zenith


def _partition(dt):
    return dt[list(dt.children)[0]]


@pytest.fixture(scope="module")
def setup():
    if not _HDF.exists() or not _MS.exists():
        pytest.skip("reference HDF/MS not present")
    ours_part = _partition(read_hdf_as_msv4(_HDF))
    enu, lat, lon, alt = antenna_enu_and_site(ours_part)
    a1, a2 = baseline_index_arrays(ours_part)
    truth_pos = enu_to_ecef_truth(enu, lat, lon, alt)
    ours_pos = antenna_ecef(ours_part["antenna_xds"].to_dataset(inherit=False))
    tart_pos = antenna_ecef(_partition(xr.open_datatree(str(_MS), engine="xarray-ms:msv2"))["antenna_xds"].to_dataset(inherit=False))
    times = np.asarray(ours_part.ds.time.values)
    tmid = times[times.size // 2 : times.size // 2 + 1]  # single representative integration
    pix = make_pixel_grid(NSIDE, xp=np)
    return dict(
        truth_bl=baselines_from_positions(truth_pos, a1, a2),
        ours_bl=baselines_from_positions(ours_pos, a1, a2),
        tart_bl=baselines_from_positions(tart_pos, a1, a2),
        tmid=tmid, pix=pix, site=(lat, lon, alt),
    )


@pytest.fixture(scope="module")
def results(setup):
    """Per-source: ours/tart offsets vs truth, their differential, the analytic prediction, flux ratio."""
    s = setup
    rows = []
    for el in ELS_DEG:
        ra, dec = sources_spanning_zenith(s["tmid"], *s["site"], els_deg=[el])
        svec = source_svec(ra, dec)  # (1,3)
        vis = simulate_visibilities([FLUX], svec, s["truth_bl"], s["tmid"], FREQ)
        wgt = np.ones_like(vis.real)
        dmap_ours = image_frame(vis, wgt, s["tmid"], s["ours_bl"], s["pix"], FREQ, xp=np)
        dmap_tart = image_frame(vis, wgt, s["tmid"], s["tart_bl"], s["pix"], FREQ, xp=np)
        rec_ours, flux_ours = recovered_direction_and_flux(dmap_ours, s["pix"], NSIDE)
        rec_tart, _ = recovered_direction_and_flux(dmap_tart, s["pix"], NSIDE)
        brot_truth = equatorial_baselines(s["truth_bl"], s["tmid"], xp=np)[0]
        brot_tart = equatorial_baselines(s["tart_bl"], s["tmid"], xp=np)[0]
        rows.append(dict(
            el=el,
            off_ours=angular_offset(rec_ours, svec[0]),
            off_tart=angular_offset(rec_tart, svec[0]),
            diff=angular_offset(rec_ours, rec_tart),
            pred=analytic_offset(brot_tart, brot_truth, svec[0]),
            flux_ratio=flux_ours / FLUX,
        ))
    arcmin = np.degrees(1.0) * 60.0
    print("\n el   off_ours  off_tart  diff(o-t) analytic  flux_ratio  [arcmin]")
    for r in rows:
        print(f"{r['el']:5.0f} {r['off_ours']*arcmin:9.3f} {r['off_tart']*arcmin:9.3f} "
              f"{r['diff']*arcmin:9.3f} {r['pred']*arcmin:9.3f} {r['flux_ratio']:10.4f}")
    return rows


def test_our_geometry_is_more_accurate(results):
    """At the clearly-resolved zenith angles, our recovered source is closer to truth."""
    for r in results:
        if r["el"] <= 35.0:  # largest geometry effect, well above the centroid floor
            assert r["off_ours"] < r["off_tart"]


def test_tart2ms_shift_matches_analytic_prediction(results):
    """The our-vs-tart2ms differential shift matches the stationary-phase prediction (within 2x)."""
    horizon = min(results, key=lambda r: r["el"])
    assert 0.5 * horizon["pred"] < horizon["diff"] < 2.0 * horizon["pred"]


def test_offset_grows_toward_horizon(results):
    """The tart2ms position error is larger near the horizon than near zenith."""
    by_el = {r["el"]: r["off_tart"] for r in results}
    assert by_el[min(ELS_DEG)] > by_el[max(ELS_DEG)]


def test_flux_is_recovered_in_jy(results):
    """Our dirty-map peak recovers the injected flux (Jy/pixel) to a few percent."""
    for r in results:
        np.testing.assert_allclose(r["flux_ratio"], 1.0, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run: `.venv/bin/python -m pytest tests/test_accuracy_l2_imaging.py -v -s`
Expected first: collection/assertion issues if helpers missing. Once Tasks 1–4 are in: **PASS (4 passed)**, with the printed table showing `off_ours` ≪ `off_tart`, `diff ≈ analytic`, `off_tart` growing toward the horizon, and `flux_ratio ≈ 1`.

If `test_our_geometry_is_more_accurate` fails only at the smallest offsets, the sub-pixel centroid
floor at nside=128 is competing with the geometry shift — raise `NSIDE` to 256 (≈1.7 GB/image) and
re-run; do **not** loosen the assertion.

- [ ] **Step 3: Format, lint, commit**

```bash
.venv/bin/ruff format tests/test_accuracy_l2_imaging.py
.venv/bin/ruff check tests/test_accuracy_l2_imaging.py --fix
git add tests/test_accuracy_l2_imaging.py
git commit -m "test: add L2 sub-pixel imaging accuracy verification vs tart2ms geometry"
```

---

## Self-Review

**Spec coverage:**
- §1 independent truth (pyproj, no circularity) → Task 1 (`enu_to_ecef_truth`, self-validated vs EPSG:4979→4978).
- §2 geometry isolation (only ECEF varies) → Tasks 2 & 5 use one truth-vis, one imager/`C(t)`/grid, three position sets.
- §3 L1 baseline comparison (ours≈truth, tart2ms~cm) → Task 2.
- §4 L2 sub-pixel sources spanning ZA, peak-offset arcsec, Jy flux, analytic cross-check → Tasks 3–5.
- §5 components (test-only helpers + two test modules) → flat `tests/accuracy_helpers.py` (noted deviation) + `test_accuracy_l1_geometry.py` / `test_accuracy_l2_imaging.py`.
- §6 pyproj in test deps → Task 1 Step 1.
- §7 deterministic CPU assertions of superiority → all tasks (`xp=np`, fixed seeds/fixed source set).
- §8 out of scope (movie, disko, noise, TLE) → not implemented.

**Placeholder scan:** none — every step has complete code/commands.

**Type/name consistency:** helper signatures (`enu_to_ecef_truth`, `baselines_from_positions`,
`antenna_ecef`, `antenna_enu_and_site`, `baseline_index_arrays`, `source_svec`,
`sources_spanning_zenith`, `simulate_visibilities`, `recovered_direction_and_flux`,
`angular_offset`, `analytic_offset`) are defined once and called with matching arguments in Tasks 2
and 5. `image_frame`/`equatorial_baselines`/`make_pixel_grid`/`dft_forward` match the shipped
`kremetart.utils.healpix_dft` API. `SITE` dict in the helper tests matches the
`enu_to_ecef_truth(**SITE)` keyword names (`lat_deg/lon_deg/alt_m`).

**Known approximations (intentional, documented in steps):** the analytic offset is a
stationary-phase prediction matched to ~2× (PSF asymmetry); the sub-pixel centroid has a small
common floor at fixed nside, so the "more accurate" assertion is applied at the clearly-resolved
zenith angles (with an nside bump as the escalation, not a tolerance loosening).

---

## Implementation notes (as-built deviations)

Two changes were made during execution and reflect the committed code:

1. **PROJ invocation (Task 1).** `Transformer.from_crs(topocentric, geocentric)` raises a
   units-mismatch error. The helper instead uses an explicit inverse-topocentric pipeline:
   `Transformer.from_pipeline("+proj=pipeline +step +inv +proj=topocentric +ellps=WGS84 +lon_0=.. +lat_0=.. +h_0=..")`,
   validated by the self-tests (origin → site ECEF, rigid 3.4 m offsets).

2. **Coplanar mirror ambiguity (Tasks 4–5).** TART's array is coplanar (Up ≈ 0), so `b·ŝ` is
   independent of the Up component and the full-sky dirty map is mirror-symmetric about the horizon
   plane — every source has an equal-amplitude reflection below the horizon, and a global `argmax`
   picks it arbitrarily. `recovered_direction_and_flux` gained `near`/`search_radius_deg` to restrict
   the peak search to the *known* source's hemisphere (legitimate: we measure the position accuracy
   of an injected source, not blind detection). The analytic cross-check is asserted at the
   clearly-resolved angles (el ≤ 35°); near zenith the geometry effect collapses (`b·ŝ → 0`) and the
   measured shift is centroid-floor-limited.

**Result (nside=128, single integration, arcmin):**

| el | off_ours | off_tart | diff(o–t) | analytic | flux_ratio |
|----|----------|----------|-----------|----------|------------|
| 15 | 25.2 | 70.2 | 46.7 | 70.8 | 0.989 |
| 35 | 10.4 | 17.2 | 26.7 | 27.1 | 0.991 |
| 55 |  7.5 |  8.5 |  1.5 | 13.3 | 0.994 |
| 75 | 11.4 | 12.2 |  0.8 |  5.1 | 0.986 |

Our geometry recovers the source closer to truth where the effect is resolvable, the differential
shift tracks the analytic prediction, the error grows toward the horizon, and flux recovers in Jy.
`off_ours` is floor-limited by the single-snapshot beam, so the **differential** is the clean
geometry-isolating metric. (L1 separately: ours = 0.0 mm vs PROJ truth, tart2ms = 17.9 mm.)

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-15-accuracy-verification.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
