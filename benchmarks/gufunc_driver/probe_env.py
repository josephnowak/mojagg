"""Environment stability check: reproduce both contradictory numbagg
measurements in one process + report machine load/affinity."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def bench_us(fn, *args, min_time=0.2, **kw) -> float:
    fn(*args, **kw)
    t0 = time.perf_counter()
    fn(*args, **kw)
    est = max(time.perf_counter() - t0, 1e-8)
    reps = min(1000, max(3, int(min_time / est)))
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args, **kw)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e6


def main() -> int:
    with Path("/proc/loadavg").open() as f:
        print("loadavg:", f.read().strip())
    print("affinity cpus:", len(os.sched_getaffinity(0)))

    import numbagg

    rs = np.random.RandomState(0)
    big = rs.rand(10_000, 1_000)
    big[rs.rand(10_000, 1_000) < 0.15] = np.nan

    # repeat the same measurement 5 times to see drift
    for i in range(5):
        t = bench_us(numbagg.allnan, big, axis=1)
        print(f"  numbagg.allnan (10k,1000) run {i}: {t:9.1f} us")
    for i in range(3):
        t = bench_us(numbagg.nansum, big, axis=1)
        print(f"  numbagg.nansum (10k,1000) run {i}: {t:9.1f} us")

    import threading

    print("numba threading layer:")
    try:
        from numba.np.ufunc import parallel as _par

        print("  layer:", _par.threading_layer())
    except Exception as e:
        print("  (unavailable:", e, ")")
    print("  numba threads env:", os.environ.get("NUMBA_NUM_THREADS", "<unset>"))
    print("  active python threads:", threading.active_count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
