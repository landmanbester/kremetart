"""The IWP filter stage: a stream of maps -> smooth map and normalised innovation per snapshot."""

from __future__ import annotations

from kremetart.imaging.snapshot import Array
from kremetart.utils.iwp import frame_has_observation, iwp_filter_step

_DIFFUSE = 1e6  # diffuse-prior variance for the first snapshot's warm-up


class IWPFilter:
    """Per-pixel integrated-Wiener-process (q=1) Kalman filter over a map stream.

    State across snapshots: per-pixel means and covariances, and the previous timestamp. The first map
    is update-only under a diffuse prior. A map that carries no data (all zero or non-finite) is not
    ingested: the filter coasts on its prediction and reports a zero normalised innovation.

    Maps must arrive in time order with distinct timestamps (``dt > 0``); a ``dt <= 0`` would break the
    covariance's positive-definiteness.

    Args:
        npix: number of pixels (independent filters).
        sigma2: IWP driving variance ``σ²``.
        noise: measurement-noise variance ``R``.
        xp: array module holding the state.
    """

    def __init__(self, npix: int, *, sigma2: float, noise: float, xp):
        self.sigma2 = float(sigma2)
        self.noise = float(noise)
        self.xp = xp
        # float64 on purpose: the _DIFFUSE covariance is conditioning-sensitive.
        self._X = xp.zeros((npix, 2))
        self._P = xp.broadcast_to(xp.eye(2) * _DIFFUSE, (npix, 2, 2)).copy()
        self._t_prev: float | None = None

    def update(self, map_: Array, time: float) -> tuple[Array, Array]:
        """Ingest one map at ``time``; return ``(smooth, normalised_innovation)``, both ``(npix,)``."""
        xp = self.xp
        dt = None if self._t_prev is None else time - self._t_prev
        self._X, self._P, smooth, innovation = iwp_filter_step(
            self._X,
            self._P,
            dt=dt,
            y=map_,
            sigma2=self.sigma2,
            R=self.noise,
            has_obs=frame_has_observation(map_, xp=xp),
            xp=xp,
        )
        self._t_prev = float(time)
        return smooth, innovation
