"""L2: imaging PROJ-truth visibilities with our geometry recovers the source more accurately
than tart2ms's geometry (sub-pixel, off-grid sources spanning zenith angle)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from kremetart.utils import partition_datatree
from kremetart.utils.healpix_dft import equatorial_baselines, image_frame, make_pixel_grid
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
from tests.accuracy_helpers import (
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


@pytest.fixture(scope="module")
def setup():
    if not _HDF.exists() or not _MS.exists():
        pytest.skip("reference HDF/MS not present")
    ours_part = partition_datatree(read_hdf_as_msv4(_HDF))
    enu, lat, lon, alt = antenna_enu_and_site(ours_part)
    a1, a2 = baseline_index_arrays(ours_part)
    truth_pos = enu_to_ecef_truth(enu, lat, lon, alt)
    ours_pos = antenna_ecef(ours_part["antenna_xds"].to_dataset(inherit=False))
    tart_pos = antenna_ecef(
        partition_datatree(xr.open_datatree(str(_MS), engine="xarray-ms:msv2"))["antenna_xds"].to_dataset(inherit=False)
    )
    times = np.asarray(ours_part.ds.time.values)
    tmid = times[times.size // 2 : times.size // 2 + 1]  # single representative integration
    pix = make_pixel_grid(NSIDE, xp=np)
    return dict(
        truth_bl=baselines_from_positions(truth_pos, a1, a2),
        ours_bl=baselines_from_positions(ours_pos, a1, a2),
        tart_bl=baselines_from_positions(tart_pos, a1, a2),
        tmid=tmid,
        pix=pix,
        site=(lat, lon, alt),
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
        # Coplanar array -> mirror-symmetric map; search the known source's hemisphere only.
        rec_ours, flux_ours = recovered_direction_and_flux(
            dmap_ours, s["pix"], NSIDE, near=svec[0], search_radius_deg=10.0
        )
        rec_tart, _ = recovered_direction_and_flux(dmap_tart, s["pix"], NSIDE, near=svec[0], search_radius_deg=10.0)
        brot_truth = equatorial_baselines(s["truth_bl"], s["tmid"], xp=np)[0]
        brot_tart = equatorial_baselines(s["tart_bl"], s["tmid"], xp=np)[0]
        rows.append(
            dict(
                el=el,
                off_ours=angular_offset(rec_ours, svec[0]),
                off_tart=angular_offset(rec_tart, svec[0]),
                diff=angular_offset(rec_ours, rec_tart),
                pred=analytic_offset(brot_tart, brot_truth, svec[0]),
                flux_ratio=flux_ours / FLUX,
            )
        )
    arcmin = np.degrees(1.0) * 60.0
    print("\n el   off_ours  off_tart  diff(o-t) analytic  flux_ratio  [arcmin]")
    for r in rows:
        print(
            f"{r['el']:5.0f} {r['off_ours'] * arcmin:9.3f} {r['off_tart'] * arcmin:9.3f} "
            f"{r['diff'] * arcmin:9.3f} {r['pred'] * arcmin:9.3f} {r['flux_ratio']:10.4f}"
        )
    return rows


def test_our_geometry_is_more_accurate(results):
    """At the clearly-resolved zenith angles, our recovered source is closer to truth."""
    for r in results:
        if r["el"] <= 35.0:  # largest geometry effect, well above the centroid floor
            assert r["off_ours"] < r["off_tart"]


def test_tart2ms_shift_matches_analytic_prediction(results):
    """The our-vs-tart2ms differential shift matches the stationary-phase prediction (within 2x)
    at the clearly-resolved (lower-elevation) angles; near zenith the shift is floor-limited."""
    for r in results:
        if r["el"] <= 35.0:
            assert 0.5 * r["pred"] < r["diff"] < 2.0 * r["pred"]


def test_offset_grows_toward_horizon(results):
    """The tart2ms position error is larger near the horizon than near zenith."""
    by_el = {r["el"]: r["off_tart"] for r in results}
    assert by_el[min(ELS_DEG)] > by_el[max(ELS_DEG)]


def test_flux_is_recovered_in_jy(results):
    """Our dirty-map peak recovers the injected flux (Jy/pixel) to a few percent."""
    for r in results:
        np.testing.assert_allclose(r["flux_ratio"], 1.0, atol=0.05)
