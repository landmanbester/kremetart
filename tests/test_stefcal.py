"""Unit tests for the acquisition StEFCal solver (simulation round-trip gate)."""

import numpy as np
import pytest

from kremetart.utils.skymodel import enu_direction_cosines, model_visibilities
from kremetart.utils.stefcal import referenced_phases, stefcal_solve

_FREQS = np.array([1.575e9])
_N_ANT = 24


def _layout(seed=0):
    """A planar 24-antenna ENU layout and its baseline antenna-index arrays."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(_N_ANT, 3))
    pos[:, 2] = 0.0  # TART elements are coplanar (Up = 0)
    a1, a2 = np.triu_indices(_N_ANT, k=1)
    bl = pos[a1] - pos[a2]
    return a1, a2, bl


def _sky(seed=1, nsrc=30):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0.0, 2.0 * np.pi, nsrc)
    el = rng.uniform(np.radians(20.0), np.radians(90.0), nsrc)
    return enu_direction_cosines(az, el)


def _true_gains(seed=2):
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.5, 2.0, _N_ANT)
    phase = rng.uniform(-np.pi, np.pi, _N_ANT)
    return amp * np.exp(1j * phase)


def _synth(g, model, a1, a2, noise=0.0, seed=3):
    vis = (g[a1] * model[:, 0] * np.conj(g[a2]))[:, None]
    if noise:
        rng = np.random.default_rng(seed)
        vis = vis + noise * (rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape))
    return vis


def test_stefcal_recovers_gains_noiseless():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    g_hat, info = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)
    assert info["converged"]
    g_true_ref = g_true / g_true[0]  # same gauge (ref antenna 0 -> 1)
    np.testing.assert_allclose(g_hat, g_true_ref, atol=1e-6)
    np.testing.assert_allclose(g_hat[0], 1.0 + 0.0j, atol=1e-12)


def test_referenced_phases_match_truth_up_to_gauge():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    g_hat, _ = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)
    got = referenced_phases(g_hat, 0)
    want = np.angle(g_true) - np.angle(g_true[0])
    # compare as unit complex to sidestep 2pi wrapping
    np.testing.assert_allclose(np.exp(1j * got), np.exp(1j * want), atol=1e-6)


def test_gauge_invariant_to_global_complex_scale():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    c = 0.7 * np.exp(1j * 1.1)  # arbitrary global complex factor
    vis_a = _synth(g_true, model, a1, a2)
    vis_b = _synth(c * g_true, model, a1, a2)  # V scales by |c|^2; phases referenced-invariant
    pa = referenced_phases(stefcal_solve(vis_a, model, a1, a2, _N_ANT, ref_ant=0)[0], 0)
    pb = referenced_phases(stefcal_solve(vis_b, model, a1, a2, _N_ANT, ref_ant=0)[0], 0)
    np.testing.assert_allclose(np.exp(1j * pa), np.exp(1j * pb), atol=1e-6)


def test_stefcal_flags_dead_antenna():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    dead = 5
    touch = (a1 == dead) | (a2 == dead)
    weight = np.ones((a1.size, 1))
    weight[touch] = 0.0
    vis[touch] = 0.0  # reader zeroes dead baselines too
    g_hat, info = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0, weight=weight)
    assert np.isnan(g_hat[dead])
    live = np.arange(_N_ANT) != dead
    assert np.all(np.isfinite(g_hat[live]))
    np.testing.assert_allclose(g_hat[live], (g_true / g_true[0])[live], atol=1e-6)


def test_stefcal_recovers_with_noise():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2, noise=1e-3)
    g_hat, _ = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)
    np.testing.assert_allclose(g_hat, g_true / g_true[0], atol=5e-3)


def test_stefcal_ref_dead_raises():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    vis = _synth(_true_gains(), model, a1, a2)
    touch = (a1 == 0) | (a2 == 0)
    weight = np.ones((a1.size, 1))
    weight[touch] = 0.0
    with pytest.raises(ValueError):
        stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0, weight=weight)
