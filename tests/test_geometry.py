"""Host-side sky geometry (:mod:`kremetart.utils.geometry`)."""

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time

from kremetart.utils import partition_datatree
from kremetart.utils.geometry import common_phase_direction, equatorial_baselines


def test_equatorial_baselines_matches_itrs_unit_vectors():
    """b_rot . s_icrs equals b_itrs . s_itrs(t) with s_itrs(t) from astropy's full ICRS->ITRS transform."""
    rng = np.random.default_rng(3)
    itrs_bl = rng.standard_normal((7, 3)) * 3.0
    times = np.array([1.6e9, 1.6e9 + 600.0, 1.6e9 + 3600.0])
    ra, dec = 1.2, -0.35
    s_icrs = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    b_rot = equatorial_baselines(itrs_bl, times, xp=np)  # (n_time, nbl, 3)
    delay_imager = b_rot @ s_icrs  # (n_time, nbl)
    itrs = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs").transform_to(
        ITRS(obstime=Time(times, format="unix", scale="utc"))
    )
    s_itrs = np.stack([itrs.x.value, itrs.y.value, itrs.z.value], axis=1)  # (n_time, 3)
    delay_ref = np.einsum("bi,ti->tb", itrs_bl, s_itrs)
    # The imager's C(t) is a pure Earth-orientation rotation, so it reproduces the full astropy
    # ICRS->ITRS source transform up to stellar aberration (the non-rotational ICRS<->GCRS part,
    # ~20 arcsec): a delay residual ~3e-4 of the baseline length, hundreds of times below the
    # ~0.9 deg HEALPix pixel. A real bug (wrong axis/transpose/sign) would be O(baseline) metres.
    np.testing.assert_allclose(delay_imager, delay_ref, rtol=3e-3, atol=1e-3)


def test_zenith_icrs_vectors_unit_and_points_up():
    """Boresight vectors are unit-norm and transform back to altitude ~90 deg at the site."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    from kremetart.utils.geometry import zenith_icrs_vectors

    lat, lon, alt = -23.7, 133.9, 100.0  # MWA-ish southern site
    times = np.array([1.6e9, 1.6e9 + 1800.0])
    vecs = zenith_icrs_vectors(times, lat, lon, alt)
    assert vecs.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-12)

    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
    for t, v in zip(times, vecs):
        dec = np.arcsin(v[2])
        ra = np.arctan2(v[1], v[0])
        aa = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs").transform_to(
            AltAz(obstime=Time(t, format="unix", scale="utc"), location=loc)
        )
        assert aa.alt.deg > 89.9


def test_common_phase_direction_dec_matches_latitude(hdf_paths):
    ra, dec = common_phase_direction(hdf_paths)
    info = partition_datatree(xr.open_datatree(hdf_paths[0], engine="tart")).ds.attrs["observation_info"]
    lat = info["site_latitude_deg"]
    # The declination of the local zenith equals the observer's geodetic latitude, up to the
    # geodetic-vs-geocentric difference (~0.2 deg). Independent physical check, not a re-derivation.
    assert abs(dec - lat) < 0.3
    assert 0.0 <= ra < 360.0
    # Deterministic.
    assert (ra, dec) == common_phase_direction(hdf_paths)


def test_common_phase_direction_empty_raises():
    with pytest.raises(ValueError, match="no HDF files"):
        common_phase_direction([])
