"""Independent geodetic truth + simulation helpers for accuracy verification (test-only).

Truth antenna ECEF comes from pyproj/PROJ (WGS84) -- a code path independent of both the kremetart
reader (hand-rolled transform) and tart2ms (mean-Earth-radius offset_by). See
docs/superpowers/specs/2026-06-15-accuracy-verification-design.md.
"""

from __future__ import annotations

import numpy as np

LIGHTSPEED = 299792458.0


def enu_to_ecef_truth(enu, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Independent ENU->ECEF via PROJ topocentric (WGS84).

    Args:
        enu: ``(n, 3)`` East/North/Up offsets (m) relative to the site.
        lat_deg, lon_deg, alt_m: site geodetic origin.

    Returns:
        ``(n, 3)`` geocentric ECEF positions (m).
    """
    from pyproj import Transformer

    enu = np.asarray(enu, dtype=np.float64)
    # Inverse PROJ topocentric: topocentric ENU -> geocentric ECEF (the forward direction maps
    # geocentric XYZ -> ENU). from_crs(topocentric, geocentric) trips a units mismatch, so build
    # the pipeline explicitly.
    pipe = f"+proj=pipeline +step +inv +proj=topocentric +ellps=WGS84 +lon_0={lon_deg} +lat_0={lat_deg} +h_0={alt_m}"
    tr = Transformer.from_pipeline(pipe)
    x, y, z = tr.transform(enu[:, 0], enu[:, 1], enu[:, 2])
    return np.stack([x, y, z], axis=1)


def baselines_from_positions(positions, ant1_idx, ant2_idx) -> np.ndarray:
    """Baseline vectors ``pos[ant1] - pos[ant2]`` -> ``(nbl, 3)``."""
    positions = np.asarray(positions)
    return positions[np.asarray(ant1_idx)] - positions[np.asarray(ant2_idx)]
