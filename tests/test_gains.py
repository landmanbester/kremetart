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
