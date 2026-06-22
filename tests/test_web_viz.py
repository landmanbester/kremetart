import threading

import numpy as np

from kremetart.utils.healpix_viz import (
    NAMES,
    SYMMETRIC,
    FrameSlot,
    LatestFrameHolder,
    encode_frame,
)


def test_names_and_symmetric():
    assert NAMES == ("raw", "smooth", "znorm")
    assert SYMMETRIC == frozenset({"znorm"})


def test_encode_frame_plain():
    values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    vmin, vmax, data = encode_frame(values, symmetric=False)
    assert (vmin, vmax) == (0.0, 3.0)
    assert np.frombuffer(data, dtype="<f4").tolist() == [0.0, 1.0, 2.0, 3.0]


def test_encode_frame_symmetric_centers_on_zero():
    values = np.array([-2.0, 0.5, 1.0], dtype=np.float32)
    vmin, vmax, _ = encode_frame(values, symmetric=True)
    assert vmax == 2.0 and vmin == -2.0


def test_encode_frame_empty():
    vmin, vmax, data = encode_frame(np.array([], dtype=np.float32), symmetric=False)
    assert (vmin, vmax) == (0.0, 0.0) and data == b""
    vmin, vmax, data = encode_frame(np.array([], dtype=np.float32), symmetric=True)
    assert (vmin, vmax) == (0.0, 0.0) and data == b""


def test_holder_put_and_snapshot_latest_wins():
    h = LatestFrameHolder(NAMES)
    assert h.snapshot() == {"raw": None, "smooth": None, "znorm": None}
    h.put("raw", 0, 1.0, 0.0, 1.0, b"a")
    h.put("raw", 1, 2.0, 0.0, 1.0, b"b")  # latest wins
    snap = h.snapshot()
    assert isinstance(snap["raw"], FrameSlot)
    assert snap["raw"].seq == 1 and snap["raw"].data == b"b"
    assert h.current_seq == 1


def test_holder_finish_flag():
    h = LatestFrameHolder(NAMES)
    assert h.finished is False
    h.finish()
    assert h.finished is True


def test_holder_is_thread_safe_under_concurrent_puts():
    h = LatestFrameHolder(("raw",))

    def worker(start):
        for seq in range(start, start + 500):
            h.put("raw", seq, float(seq), 0.0, 1.0, b"x")

    threads = [threading.Thread(target=worker, args=(i * 500,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No crash, slot holds a valid FrameSlot, current_seq is the max seen.
    assert h.snapshot()["raw"] is not None
    expected_max = max(i * 500 + 499 for i in range(4))  # 1999, derived from the worker ranges
    assert h.current_seq == expected_max
