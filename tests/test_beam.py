"""Unit tests for the analytic Airy primary beam (CPU, xp=numpy)."""

import numpy as np
from scipy.special import j1

from kremetart.utils.beam import GROUND_PLANE_DIAMETER, airy_power_beam
from kremetart.utils.healpix_dft import make_pixel_grid

LIGHTSPEED = 299792458.0


def test_shape_and_peak_at_boresight():
    """Output is (nchan, npix) and the beam peaks at exactly 1 on the boresight pixel."""
    pix = make_pixel_grid(8, xp=np)
    freqs = np.array([1.227e9, 1.575e9])  # GPS L2, L1
    boresight = pix[123]  # point at a real pixel so cos(theta) == 1 there
    beam = airy_power_beam(pix, boresight, freqs, xp=np)
    assert beam.shape == (freqs.size, pix.shape[0])
    np.testing.assert_allclose(beam[:, 123], 1.0, atol=1e-12)
    assert beam.max() <= 1.0 + 1e-12


def test_matches_explicit_airy_formula():
    """Beam equals [2 J1(x)/x]^2 with x = pi D sin(theta) / lambda, for above-horizon pixels."""
    pix = make_pixel_grid(8, xp=np)
    freqs = np.array([1.575e9])
    boresight = np.array([0.0, 0.0, 1.0])  # zenith
    beam = airy_power_beam(pix, boresight, freqs, xp=np)

    mu = np.clip(pix @ boresight, -1.0, 1.0)
    sinth = np.sqrt(1.0 - mu**2)
    x = np.pi * GROUND_PLANE_DIAMETER * (freqs[0] / LIGHTSPEED) * sinth
    expected = np.where(x == 0.0, 1.0, 2.0 * j1(np.where(x == 0.0, 1.0, x)) / np.where(x == 0.0, 1.0, x)) ** 2
    expected = np.where(mu >= 0.0, expected, 0.0)
    np.testing.assert_allclose(beam[0], expected, rtol=1e-12, atol=1e-12)


def test_below_horizon_is_zero():
    """Pixels in the back hemisphere (cos(theta) < 0) are exactly zero."""
    pix = make_pixel_grid(8, xp=np)
    freqs = np.array([1.575e9])
    boresight = np.array([0.0, 0.0, 1.0])  # zenith -> back hemisphere is z < 0
    beam = airy_power_beam(pix, boresight, freqs, xp=np)
    below = pix[:, 2] < 0.0
    assert np.all(beam[0, below] == 0.0)
    assert np.all(beam[0, ~below] > 0.0)  # broad beam: no null in the visible hemisphere at L1


def test_lower_frequency_gives_wider_beam():
    """Longer wavelength -> wider beam -> higher response at a fixed off-axis angle."""
    pix = make_pixel_grid(16, xp=np)
    boresight = np.array([0.0, 0.0, 1.0])
    freqs = np.array([1.176e9, 1.575e9])  # L5 (low) then L1 (high)
    beam = airy_power_beam(pix, boresight, freqs, xp=np)
    # An off-axis but above-horizon pixel: lower freq must have the larger (or equal) response.
    off_axis = (pix[:, 2] > 0.2) & (pix[:, 2] < 0.6)
    assert np.all(beam[0, off_axis] >= beam[1, off_axis] - 1e-12)
    assert beam[0, off_axis].mean() > beam[1, off_axis].mean()


def test_boresight_normalised_internally():
    """A non-unit boresight gives the same beam as its normalised counterpart."""
    pix = make_pixel_grid(8, xp=np)
    freqs = np.array([1.575e9])
    unit = np.array([0.0, 0.0, 1.0])
    scaled = 7.3 * unit
    np.testing.assert_allclose(
        airy_power_beam(pix, scaled, freqs, xp=np),
        airy_power_beam(pix, unit, freqs, xp=np),
        rtol=1e-12,
        atol=1e-12,
    )
