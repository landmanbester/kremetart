"""Spec-compliance tests for ``read_hdf_as_msv4``.

These assert that the MSv4 ``DataTree`` produced by
:func:`kremetart.utils.read_tart_hdf.read_hdf_as_msv4` is structurally compliant with the
canonical MSv4 layout, by comparing it against a reference Measurement Set that was converted
from the same HDF chunk with ``tart2ms`` and opened through ``xarray-ms``::

    tart2ms --hdf <file>.hdf --ms <file>.ms --rephase obs-midpoint --single-field \\
        --write-model-catalog
    xr.open_datatree("<file>.ms", engine="xarray-ms:msv2")

Only *structure* is compared (node hierarchy, dimensions, coordinate/variable names, dtypes and
schema attributes), never data values. The reference MS was rephased to a phase centre at the
observation midpoint, so the visibilities, the UVW vectors and the field phase-centre direction
legitimately differ from our zenith-referenced output until rephasing is implemented. Those
rephasing-dependent fields are deliberately excluded from the comparison, as is the documented
dimensionless-vs-Jy ``VISIBILITY`` units choice.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from kremetart.utils.read_tart_hdf import read_hdf_as_msv4

_DATA = Path(__file__).parent / "data"
_HDF = _DATA / "vis_2026-06-09_08_11_43.476804.hdf"
_MS = _DATA / "vis_2026-06-09_08_11_43.476804.ms"


def _single_partition(dt: "xr.DataTree") -> "xr.DataTree":
    """Return the sole partition node beneath an MSv4 DataTree root."""
    children = list(dt.children)
    assert len(children) == 1, f"expected exactly one partition, found {children}"
    return dt[children[0]]


@pytest.fixture(scope="module")
def reference() -> "xr.DataTree":
    """The canonical MSv4 partition node from the tart2ms-converted Measurement Set."""
    if not _MS.exists():
        pytest.skip(f"reference MS not present: {_MS}")
    return _single_partition(xr.open_datatree(str(_MS), engine="xarray-ms:msv2"))


@pytest.fixture(scope="module")
def generated() -> "xr.DataTree":
    """The partition node produced by read_hdf_as_msv4 from the raw HDF chunk."""
    if not _HDF.exists():
        pytest.skip(f"test HDF not present: {_HDF}")
    return _single_partition(read_hdf_as_msv4(_HDF))


def test_partition_has_required_subnodes(reference, generated) -> None:
    """Both trees expose the MSv4 antenna and field-and-source sub-nodes."""
    for node in (reference, generated):
        assert "antenna_xds" in node.children
        assert "field_and_source_base_xds" in node.children


def test_main_dimensions_match(reference, generated) -> None:
    """The main visibility xds has identical dimension sizes."""
    assert dict(generated.ds.sizes) == dict(reference.ds.sizes)


def test_main_coords_are_superset(reference, generated) -> None:
    """Every reference main coordinate is present in our output."""
    ref = set(reference.to_dataset(inherit=False).coords)
    gen = set(generated.to_dataset(inherit=False).coords)
    assert ref <= gen, f"missing MSv4 main coords: {ref - gen}"


def test_main_data_vars_are_superset(reference, generated) -> None:
    """Every reference main data variable is present in our output."""
    ref = set(reference.ds.data_vars)
    gen = set(generated.ds.data_vars)
    assert ref <= gen, f"missing MSv4 main data vars: {ref - gen}"


def test_schema_attributes(reference, generated) -> None:
    """Both carry the MSv4 schema attributes and a consistent 'base' data group."""
    for node in (reference, generated):
        attrs = node.ds.attrs
        assert attrs["schema_version"] == "4.0.0"
        assert attrs["type"] == "visibility"
        base = attrs["data_groups"]["base"]
        assert base["correlated_data"] == "VISIBILITY"
        assert base["flag"] == "FLAG"
        assert base["weight"] == "WEIGHT"
        assert base["uvw"] == "UVW"


def test_core_array_dtypes_and_dims(reference, generated) -> None:
    """VISIBILITY/FLAG/WEIGHT dtypes and the VISIBILITY/UVW dimension order agree."""
    g, r = generated.ds, reference.ds
    assert g.VISIBILITY.dtype == r.VISIBILITY.dtype == np.complex64
    assert g.FLAG.dtype == r.FLAG.dtype == np.uint8
    assert g.WEIGHT.dtype == r.WEIGHT.dtype == np.float32
    assert g.VISIBILITY.dims == r.VISIBILITY.dims
    assert g.UVW.dims == r.UVW.dims


def test_time_coordinate_convention(reference, generated) -> None:
    """The time coordinate is float64 unix seconds (UTC) in both trees."""
    for node in (reference, generated):
        t = node.ds.time
        assert t.dtype == np.float64
        assert t.attrs["type"] == "time"
        assert t.attrs["units"] == "s"
        assert t.attrs["format"] == "unix"
        assert t.attrs["scale"] == "utc"


def test_polarization_and_frequency(reference, generated) -> None:
    """Single RHCP (RR) polarization and the L1 frequency axis match."""
    assert list(generated.ds.polarization.values) == list(reference.ds.polarization.values)
    np.testing.assert_allclose(generated.ds.frequency.values, reference.ds.frequency.values)


def test_antenna_xds_structure(reference, generated) -> None:
    """The antenna sub-node matches in shape, names and the ITRS position frame."""
    r = reference["antenna_xds"].to_dataset(inherit=False)
    g = generated["antenna_xds"].to_dataset(inherit=False)
    assert g.sizes["antenna_name"] == r.sizes["antenna_name"]
    assert g.sizes["cartesian_pos_label"] == r.sizes["cartesian_pos_label"] == 3
    assert set(r.coords) <= set(g.coords), f"missing antenna coords: {set(r.coords) - set(g.coords)}"
    assert set(r.data_vars) <= set(g.data_vars), f"missing antenna vars: {set(r.data_vars) - set(g.data_vars)}"
    for ds in (g, r):
        attrs = ds.ANTENNA_POSITION.attrs
        assert attrs["frame"] == "ITRS"
        assert attrs["coordinate_system"] == "geocentric"
        assert attrs["units"] == "m"


def test_field_and_source_node_present(reference, generated) -> None:
    """The field-and-source node exists and is keyed by field_name.

    The phase-centre *representation* differs until rephasing is added (our zenith el/az vs the
    reference's rephased ra/dec), so only the node's existence and field keying are asserted.
    """
    for node in (reference, generated):
        fs = node["field_and_source_base_xds"].to_dataset(inherit=False)
        assert "field_name" in fs.dims
