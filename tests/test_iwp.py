"""CPU tests for the per-pixel IWP-Kalman recursion (utils.iwp), xp=numpy.

The GPU operator (operators/iwp_kalman.py) wraps these same functions with xp=cupy; here we pin
the q=1 closed forms (eq. AQ of the design note), the Joseph-form covariance staying symmetric
PSD over many steps, and the whitening property (Task 2).
"""

import numpy as np

from kremetart.utils.iwp import iwp_transition, kalman_predict, kalman_update


def test_iwp_transition_closed_form():
    a, q = iwp_transition(2.0, 3.0)
    # a = [[1, dt],[0, 1]]
    np.testing.assert_allclose(a, [[1.0, 2.0], [0.0, 1.0]])
    # q = sigma2 * [[dt^3/3, dt^2/2],[dt^2/2, dt]] with sigma2=3, dt=2 -> [[8,6],[6,6]]
    np.testing.assert_allclose(q, [[8.0, 6.0], [6.0, 6.0]])


def test_kalman_update_keeps_covariance_symmetric_psd():
    rng = np.random.default_rng(0)
    npix = 5
    x = rng.standard_normal((npix, 2))
    cov = np.broadcast_to(np.eye(2) * 10.0, (npix, 2, 2)).copy()
    a, q = iwp_transition(1.5, 0.5)
    for _ in range(50):
        x, cov = kalman_predict(x, cov, a, q)
        y = x[:, 0] + rng.standard_normal(npix) * 0.1
        x, cov, e, s = kalman_update(x, cov, y, 0.01)
    assert np.all(s > 0)
    for p in range(npix):
        np.testing.assert_allclose(cov[p], cov[p].T, atol=1e-9)  # symmetric
        assert np.all(np.linalg.eigvalsh(cov[p]) >= -1e-9)  # PSD


def test_innovations_whiten_on_synthetic_iwp():
    """Data from the q=1 IWP + Gaussian-noise model -> normalised innovations ~ N(0,1):
    mean ~ 0 and mean NIS (E[z^2]) ~ 1 after a warm-up."""
    rng = np.random.default_rng(42)
    npix, n, dt = 200, 400, 1.0
    sigma2, r_noise = 0.05, 0.2
    a, q = iwp_transition(dt, sigma2)

    # Simulate true IWP states and noisy flux observations per pixel.
    lq = np.linalg.cholesky(q)
    x_true = np.zeros((npix, 2))
    y_obs = np.zeros((npix, n))
    for k in range(n):
        x_true = x_true @ a.T + rng.standard_normal((npix, 2)) @ lq.T
        y_obs[:, k] = x_true[:, 0] + rng.standard_normal(npix) * np.sqrt(r_noise)

    # Run the filter from a diffuse prior; collect normalised innovations after warm-up.
    x_filt = np.zeros((npix, 2))
    p_cov = np.broadcast_to(np.eye(2) * 1e6, (npix, 2, 2)).copy()
    z = []
    for k in range(n):
        x_filt, p_cov = kalman_predict(x_filt, p_cov, a, q)
        x_filt, p_cov, e, s = kalman_update(x_filt, p_cov, y_obs[:, k], r_noise)
        z.append(e / np.sqrt(s))
    z = np.asarray(z[50:])  # drop warm-up samples

    assert abs(float(z.mean())) < 0.05
    assert 0.9 < float((z**2).mean()) < 1.1  # mean NIS ~ chi^2_1 mean = 1
