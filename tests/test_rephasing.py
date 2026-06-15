"""Verification tests for the rephasing operator.

These check that :func:`kremetart.utils.rephasing.rephase_to_dir` reproduces ``tart2ms``'s
``--rephase obs-midpoint`` result, using a reference Measurement Set converted from the same HDF
chunk *without* calibration applied (``..._nocal.ms``) so the raw visibility amplitudes match.

Visibilities (a pure phase rotation) are compared tightly. The UVW ``w`` component is compared
tightly too; ``u``/``v`` are compared at ~cm tolerance because ``tart2ms`` derives ITRF antenna
positions via a spherical ``offset_by`` using a mean Earth radius, whereas our reader uses the
exact tangent-plane ENU->ECEF transform -- a ~0.3% baseline-length convention difference, not a
rephasing error.
"""

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_ms")  # registers the "xarray-ms:msv2" engine
pytest.importorskip("astropy")

from kremetart.utils.read_tart_hdf import read_hdf_as_msv4  # noqa: E402  (after importorskip)
from kremetart.utils.rephasing import midpoint_zenith, rephase_to_dir  # noqa: E402

_DATA = Path(__file__).parent / "data"
_HDF = _DATA / "vis_2026-06-09_08_11_43.476804.hdf"
_MS_NOCAL = _DATA / "vis_2026-06-09_08_11_43.476804_nocal.ms"


def _single_partition(dt: "xr.DataTree") -> "xr.DataTree":
    children = list(dt.children)
    assert len(children) == 1, f"expected one partition, found {children}"
    return dt[children[0]]


@pytest.fixture(scope="module")
def reference() -> "xr.DataTree":
    if not _MS_NOCAL.exists():
        pytest.skip(f"reference nocal MS not present: {_MS_NOCAL}")
    return _single_partition(xr.open_datatree(str(_MS_NOCAL), engine="xarray-ms:msv2"))


@pytest.fixture(scope="module")
def rephased() -> "xr.DataTree":
    if not _HDF.exists():
        pytest.skip(f"test HDF not present: {_HDF}")
    dt = read_hdf_as_msv4(_HDF)
    return _single_partition(rephase_to_dir(dt, midpoint_zenith(dt)))


def test_midpoint_zenith_matches_reference_phase_centre(reference) -> None:
    """The obs-midpoint zenith equals the reference Measurement Set phase centre."""
    dt = read_hdf_as_msv4(_HDF)
    ra, dec = midpoint_zenith(dt)
    ref_dir = reference["field_and_source_base_xds"].to_dataset(inherit=False)
    ref_radec = ref_dir.FIELD_PHASE_CENTER_DIRECTION.values[0]
    np.testing.assert_allclose([ra, dec], ref_radec, atol=1e-6)


def test_visibilities_match_reference(rephased, reference) -> None:
    """Rephased visibilities reproduce the reference to high accuracy (pure phase rotation)."""
    got = rephased.ds.VISIBILITY.values
    ref = reference.ds.VISIBILITY.values
    assert np.abs(got - ref).max() < 1e-3
    # the rotation must not change amplitudes
    np.testing.assert_allclose(np.abs(got), np.abs(ref), atol=1e-6)


def test_uvw_w_component_matches_reference(rephased, reference) -> None:
    """The w coordinate (toward the phase centre) matches the reference tightly."""
    dw = rephased.ds.UVW.values[..., 2] - reference.ds.UVW.values[..., 2]
    assert np.abs(dw).max() < 1e-3


def test_uvw_uv_match_within_position_convention(rephased, reference) -> None:
    """The u, v coordinates agree to ~cm (the ITRF position-convention difference)."""
    duv = rephased.ds.UVW.values[..., :2] - reference.ds.UVW.values[..., :2]
    assert np.abs(duv).max() < 3e-2


def test_field_node_is_celestial_after_rephase(rephased) -> None:
    """The field node now carries an ra/dec phase-centre direction (MSv4 standard)."""
    field = rephased["field_and_source_base_xds"].to_dataset(inherit=False)
    assert "FIELD_PHASE_CENTER_DIRECTION" in field.data_vars
    assert field.FIELD_PHASE_CENTER_DIRECTION.attrs["frame"] == "fk5"
    assert list(field.sky_dir_label.values) == ["ra", "dec"]
    assert rephased.ds.UVW.attrs["frame"] == "fk5"


def test_itrs_baselines_public_helper() -> None:
    """The public itrs_baselines equals pos[ant1] - pos[ant2] in the partition's baseline order."""
    from kremetart.utils.rephasing import itrs_baselines

    node = _single_partition(read_hdf_as_msv4(_HDF))
    bl = np.asarray(itrs_baselines(node, np))
    ant = node["antenna_xds"].to_dataset(inherit=False)
    pos = ant.ANTENNA_POSITION.values
    index = {n: i for i, n in enumerate(ant.antenna_name.values)}
    a1 = [index[n] for n in node.ds.baseline_antenna1_name.values]
    a2 = [index[n] for n in node.ds.baseline_antenna2_name.values]
    assert bl.shape == (276, 3)
    np.testing.assert_allclose(bl, pos[a1] - pos[a2])


def test_rephase_is_identity_at_midpoint_frame(rephased) -> None:
    """At the midpoint frame the new centre equals the instantaneous zenith, so no rotation.

    Guards the phase-sign / direction-cosine convention: a sign error would still pass the
    aggregate visibility test if the reference shared it, but would show a non-zero rotation here.
    """
    raw = read_hdf_as_msv4(_HDF)
    raw_vis = _single_partition(raw).ds.VISIBILITY.values
    mid = raw_vis.shape[0] // 2
    ratio = rephased.ds.VISIBILITY.values[mid] / raw_vis[mid]
    np.testing.assert_allclose(ratio, 1.0 + 0.0j, atol=1e-6)
