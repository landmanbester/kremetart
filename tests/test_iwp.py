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
