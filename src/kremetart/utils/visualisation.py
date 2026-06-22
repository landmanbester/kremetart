import subprocess
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def _overlay_tracks(ax, tracks, frame_index):
    """Draw each satellite present in ``frame_index``: trailing line, current marker, name label.

    ``tracks`` maps name -> list of ``(frame_index, ra_deg, dec_deg, flux_jy)``. ``ax`` is the active
    healpy Mollweide projection axes (``plt.gca()`` after ``mollview``); its ``projscatter`` /
    ``projplot`` / ``projtext`` methods are called directly rather than the module-level ``hp.proj*``
    wrappers, because each wrapper forces a full ``pylab.draw()`` on every call -- turning an
    N-satellite overlay into ~N full-figure re-rasterizations per frame (the cause of ~15 s/frame
    rendering). The axes methods draw nothing until the single ``savefig`` per frame. Coordinates use
    ``lonlat=True`` (degrees, ``lon == RA``) so the active ``rot`` is applied and the overlay lands in
    the same projected ICRS frame as the imaged pixels.
    """
    for name, points in tracks.items():
        trail = [(ra, dec) for (f, ra, dec, _jy) in points if f <= frame_index]
        current = [(ra, dec) for (f, ra, dec, _jy) in points if f == frame_index]
        if not current:
            continue  # satellite not above the cutoff in this frame
        if len(trail) > 1:
            ax.projplot(
                [ra for ra, _ in trail],
                [dec for _, dec in trail],
                lonlat=True,
                color="cyan",
                linewidth=0.7,
                alpha=0.6,
            )
        ra0, dec0 = current[0]
        ax.projscatter([ra0], [dec0], lonlat=True, color="cyan", marker="x", s=30)


def render_frames(
    maps,
    timestamps,
    nside: int,
    cmap: str,
    outdir,
    *,
    rot: tuple[float, float] | None = None,
    nest: bool = True,
    tracks=None,
    diverging: bool = False,
):
    """Render each map as a Mollweide PNG with a fixed colour scale. Returns ordered PNG paths.

    ``rot=(lon, lat)`` (degrees) re-centers every frame on the common phase direction so the observed
    patch sits stably at the projection center across the movie. ``tracks`` (if given) overlays
    per-satellite ICRS trajectories (trailing line + current marker + name label) on each frame.
    """

    outdir = Path(outdir)
    stacked = np.concatenate([np.asarray(m) for m in maps])
    if diverging:
        # Symmetric scale centred on 0 with a diverging cmap (for the normalised innovation z_k).
        vmax = float(np.percentile(np.abs(stacked), 99.0))
        vmin, cmap = -vmax, "coolwarm"
    else:
        vmin, vmax = (float(v) for v in np.percentile(stacked, [1.0, 99.0]))
    paths = []
    for i, (m, ts) in enumerate(zip(maps, timestamps)):
        hp.mollview(np.asarray(m), nest=nest, title=ts, cmap=cmap, min=vmin, max=vmax, rot=rot)
        hp.graticule()
        if tracks:
            _overlay_tracks(plt.gca(), tracks, i)
        out = outdir / f"frame_{i:04d}.png"
        plt.savefig(out, dpi=100)
        plt.close("all")
        paths.append(out)
    return paths


def _encode_movie(first_png, fps: int, out) -> None:
    """Encode the ``frame_%04d.png`` sequence in ``first_png``'s directory to mp4 ``out``."""
    pattern = str(Path(first_png).parent / "frame_%04d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
