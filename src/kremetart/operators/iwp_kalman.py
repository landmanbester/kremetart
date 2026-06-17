"""Holoscan operator: per-pixel q=1 IWP-Kalman whitening filter (GPU-resident, xp=cupy).

Holds the Kalman state (means ``X``, covariances ``P``, previous timestamp ``t_prev``) as
attributes; consumes the imager's per-frame dirty map (the observation) and timestamp, runs the
exact IWP predict+update (kremetart.utils.iwp) with the per-frame Delta from the timestamp
stream, and emits the dirty map (passthrough), the filtered flux x_{k|k}[0] and the normalised
innovation z_k. See docs/superpowers/specs/2026-06-17-smoovie-iwp-filter-design.md.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.utils.iwp import iwp_transition, kalman_predict, kalman_update

_DIFFUSE = 1e6  # diffuse-prior variance for the frame-0 warm-up


class IWPKalmanOperator(Operator):
    """Per-pixel q=1 IWP-Kalman filter.

    Args:
        fragment: Holoscan fragment.
        npix: number of HEALPix pixels (independent filters).
        sigma2: IWP driving variance sigma^2.
        noise: scalar measurement-noise variance R.
    """

    def __init__(self, fragment, npix, *args, sigma2, noise, **kwargs):
        self.npix = npix
        self.sigma2 = float(sigma2)
        self.noise = float(noise)
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Diffuse prior: zero mean, large covariance. Frame 0 runs update-only (no Delta yet).
        self.X = cp.zeros((self.npix, 2))
        self.P = cp.broadcast_to(cp.eye(2) * _DIFFUSE, (self.npix, 2, 2)).copy()
        self.t_prev = None

    def setup(self, spec: OperatorSpec):
        spec.input("cube")
        spec.input("time_out")
        spec.output("cube")
        spec.output("filtered")
        spec.output("znorm")
        spec.output("time_out")

    def compute(self, op_input, op_output, context):
        cube = cp.asarray(op_input.receive("cube"))  # (1, npix)
        time_out = cp.asarray(op_input.receive("time_out"))  # (1,)
        y = cube[0]  # (npix,)
        t = float(time_out[0])

        if self.t_prev is not None:
            a, q = iwp_transition(t - self.t_prev, self.sigma2, xp=cp)
            self.X, self.P = kalman_predict(self.X, self.P, a, q, xp=cp)

        self.X, self.P, e, s = kalman_update(self.X, self.P, y, self.noise, xp=cp)
        self.t_prev = t

        filtered = self.X[:, 0]
        znorm = e / cp.sqrt(s)

        op_output.emit(hs.as_tensor(cube), "cube")  # passthrough dirty map
        op_output.emit(hs.as_tensor(filtered[None, :]), "filtered")
        op_output.emit(hs.as_tensor(znorm[None, :]), "znorm")
        op_output.emit(hs.as_tensor(time_out), "time_out")
