"""Utils for kremetart."""

import datetime

import cupy


def partition_datatree(dt):
    return dt[list(dt.children)[0]]


def unix_to_utc(unix_seconds) -> str:
    dt = datetime.datetime.fromtimestamp(float(unix_seconds), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def gpu_available() -> bool:
    """True if a CUDA device plus the GPU imaging stack (cupy/holoscan/healpy) is importable.

    Drives ``smoovie``'s auto-routing: when true the imaging runs through the Holoscan GPU app
    (:func:`kremetart.core.smoovie_app.image_via_app`); otherwise it falls back to the CPU
    :func:`frame_dirty_maps`. Any import error or absent device -> CPU path, so CPU-only CI and
    machines without a GPU keep working.
    """
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        else:
            return True
    except Exception:
        return False
