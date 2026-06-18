import contextlib
import time


@contextlib.contextmanager
def stage_timer(name, timings):
    """Record wall-clock seconds for a named stage into ``timings`` (a list of ``(name, seconds)``)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings.append((name, time.perf_counter() - t0))


def print_profile(timings, nframes):
    """Print a per-stage timing summary table to stdout."""
    total = sum(dt for _, dt in timings) or 1.0
    print("\n=== smoovie profile ===")
    print(f"{'stage':<18}{'seconds':>10}{'%total':>9}{'ms/frame':>11}")
    for name, dt in timings:
        per_frame = f"{1000.0 * dt / nframes:.1f}" if nframes else "-"
        print(f"{name:<18}{dt:>10.3f}{100.0 * dt / total:>8.1f}%{per_frame:>11}")
    print(f"{'TOTAL':<18}{total:>10.3f}{100.0:>8.1f}%")
