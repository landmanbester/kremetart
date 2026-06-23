"""Tests for the parked Mollweide renderer (kremetart.utils.visualisation).

These helpers were moved out of core/smoovie.py (which now streams to the web viewer) and are
retained for a future Mollweide-rendering sub-command. Gated on the matplotlib/healpy stack.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("healpy")


@pytest.fixture
def viz():
    import kremetart.utils.visualisation as viz

    return viz


def test_overlay_tracks_marker_trail_label(viz):
    class FakeAx:
        def __init__(self):
            self.c = {"projscatter": 0, "projplot": 0}

        def projscatter(self, *a, **k):
            self.c["projscatter"] += 1

        def projplot(self, *a, **k):
            self.c["projplot"] += 1

    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.0), (1, 12.0, -19.0, 1.0)]}

    ax = FakeAx()
    viz._overlay_tracks(ax, tracks, 0)  # frame 0: marker, no trail yet
    assert ax.c == {"projscatter": 1, "projplot": 0}

    ax = FakeAx()
    viz._overlay_tracks(ax, tracks, 1)  # frame 1: marker + trail (>1 past point)
    assert ax.c == {"projscatter": 1, "projplot": 1}

    ax = FakeAx()
    viz._overlay_tracks(ax, tracks, 5)  # satellite absent at frame 5: nothing drawn
    assert ax.c == {"projscatter": 0, "projplot": 0}


def test_render_frames_overlay_uses_axes_not_drawing_wrappers(tmp_path, viz, monkeypatch):
    # Root-cause guard: the overlay must use the projection-axes methods, NOT the module-level
    # hp.proj* wrappers -- each wrapper forces a full pylab.draw(), turning an N-satellite overlay
    # into ~N full-figure re-rasterizations per frame (the cause of ~15 s/frame rendering).
    import healpy as hp

    def boom(*a, **k):
        raise AssertionError("overlay must call ax.proj* (no pylab.draw), not the hp.proj* wrappers")

    monkeypatch.setattr(hp, "projscatter", boom)
    monkeypatch.setattr(hp, "projplot", boom)

    nside = 8
    npix = 12 * nside * nside
    maps = [np.arange(npix, dtype=float), np.arange(npix, dtype=float) + 1.0]
    stamps = ["t0 UTC", "t1 UTC"]
    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.0), (1, 12.0, -19.0, 1.0)]}

    pngs = viz.render_frames(maps, stamps, nside, "inferno", tmp_path, rot=(0.0, -30.0), tracks=tracks)
    assert len(pngs) == 2
    assert all(Path(p).exists() for p in pngs)
