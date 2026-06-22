"""Host-side helpers shared by the web sink operator and the frame server.

Pure numpy/healpy — no GPU, no web framework. ``LatestFrameHolder`` is the thread-safe
bridge between the GXF scheduler thread (the sink writes) and the asyncio server thread
(the connection tasks read). One latest-wins slot per named output.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

NAMES: tuple[str, ...] = ("raw", "smooth", "znorm")
SYMMETRIC: frozenset[str] = frozenset({"znorm"})


@dataclass(frozen=True)
class FrameSlot:
    """One named output's latest frame: shared sequence number, timestamp, scale, bytes."""

    seq: int
    t: float
    vmin: float
    vmax: float
    data: bytes


def encode_frame(values: np.ndarray, *, symmetric: bool) -> tuple[float, float, bytes]:
    """Return ``(vmin, vmax, float32-little-endian bytes)`` for a HEALPix map.

    ``symmetric=True`` (the normalised innovation) centres the scale on zero so the ramp
    reads as a signed diagnostic; otherwise the scale spans the frame's own min/max.
    """
    arr = np.ascontiguousarray(values, dtype="<f4")
    if symmetric:
        vmax = float(np.max(np.abs(arr))) if arr.size else 0.0
        vmin = -vmax
    else:
        vmin = float(arr.min()) if arr.size else 0.0
        vmax = float(arr.max()) if arr.size else 0.0
    return vmin, vmax, arr.tobytes()


class LatestFrameHolder:
    """Thread-safe latest-wins frame holder, one slot per name, plus a finish flag.

    ``put`` holds the lock only for the dict swap; stored payloads are immutable, so the
    server reads them out under the lock and sends after releasing. A slow/disconnected
    consumer can never block ``put`` (and therefore never backpressures the scheduler).
    """

    def __init__(self, names: tuple[str, ...]):
        self._lock = threading.Lock()
        self._slots: dict[str, FrameSlot | None] = {name: None for name in names}
        self._current_seq = -1
        self._finished = False

    def put(self, name: str, seq: int, t: float, vmin: float, vmax: float, data: bytes) -> None:
        slot = FrameSlot(seq=seq, t=t, vmin=vmin, vmax=vmax, data=data)
        with self._lock:
            self._slots[name] = slot
            if seq > self._current_seq:
                self._current_seq = seq

    def snapshot(self) -> dict[str, FrameSlot | None]:
        with self._lock:
            return dict(self._slots)

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def current_seq(self) -> int:
        with self._lock:
            return self._current_seq
