"""Unit tests for the unit-flux ENU sky model."""

import numpy as np

from kremetart.utils.skymodel import LIGHTSPEED, enu_direction_cosines, model_visibilities


def test_enu_direction_cosines_cardinal_points():
    # az from North toward East; el is altitude above horizon.
    # zenith (el=90): straight up.
    np.testing.assert_allclose(enu_direction_cosines(0.0, np.pi / 2), [0.0, 0.0, 1.0], atol=1e-12)
    # az=90deg, el=0: due East on the horizon.
    np.testing.assert_allclose(enu_direction_cosines(np.pi / 2, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
    # az=0, el=0: due North on the horizon.
    np.testing.assert_allclose(enu_direction_cosines(0.0, 0.0), [0.0, 1.0, 0.0], atol=1e-12)


def test_enu_direction_cosines_unit_norm_and_shape():
    az = np.array([0.1, 1.0, 2.0, 3.0])
    el = np.array([0.2, 0.5, 0.9, 1.2])
    s = enu_direction_cosines(az, el)
    assert s.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-12)


def test_model_visibilities_single_source_analytic_fringe():
    bl = np.array([[10.0, 0.0, 0.0]])  # one 10 m baseline along East
    s = enu_direction_cosines(np.array([np.pi / 2]), np.array([0.0]))  # due East -> (1,0,0)
    freqs = np.array([1.5e9])
    mvis = model_visibilities(s, bl, freqs)
    expected = np.exp(2j * np.pi * (1.5e9 / LIGHTSPEED) * 10.0)  # b.s = 10
    assert mvis.shape == (1, 1)
    np.testing.assert_allclose(mvis[0, 0], expected, rtol=1e-10)


def test_model_visibilities_is_flux_one_superposition():
    rng = np.random.default_rng(0)
    bl = rng.uniform(-5, 5, size=(7, 3))
    az = rng.uniform(0, 2 * np.pi, 4)
    el = rng.uniform(0.1, 1.5, 4)
    s = enu_direction_cosines(az, el)
    freqs = np.array([1.575e9])
    mvis = model_visibilities(s, bl, freqs)
    # equals the per-source sum with unit weights
    per_src = np.stack([model_visibilities(s[i : i + 1], bl, freqs)[:, 0] for i in range(4)], axis=0).sum(axis=0)
    np.testing.assert_allclose(mvis[:, 0], per_src, rtol=1e-10)
