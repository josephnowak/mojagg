"""Thread-pool coexistence probe: does loading mojagg/max's pool change
numbagg's per-call cost?

Phase A: numbagg alone (fresh process, no mojagg import).
Phase B: after importing mojagg AND running one parallel max workload.
Phase C: mojagg timings under each condition for reference.

Run: bash benchmarks/gufunc_driver/run_probe2.sh (inside pixi env / WSL)
"""

from __future__ import annotations

import time

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


def numbagg_suite(tag: str) -> None:
    import numbagg

    rs = np.random.RandomState(0)
    small = np.full((16, 1), 1.0)
    big = rs.rand(10_000, 1_000)
    big[rs.rand(10_000, 1_000) < 0.15] = np.nan
    nan8 = np.full((10_000, 8), np.nan)

    print(f"  [{tag}] allnan (16,1) tiny       : {bench_us(numbagg.allnan, small, axis=1):9.1f} us")
    print(f"  [{tag}] allnan (10k,1000) 15%NaN : {bench_us(numbagg.allnan, big, axis=1):9.1f} us")
    print(f"  [{tag}] allnan (10k,8) all-NaN   : {bench_us(numbagg.allnan, nan8, axis=1):9.1f} us")
    print(f"  [{tag}] nansum  (10k,1000)       : {bench_us(numbagg.nansum, big, axis=1):9.1f} us")


def main() -> int:
    print("== Phase A: numbagg alone (no mojagg import yet) ==")
    numbagg_suite("numbagg-only")

    print()
    print("== Phase B: import mojagg + run one parallel max workload ==")
    import mojagg

    rs = np.random.RandomState(1)
    warm = rs.rand(2000, 3000)
    with mojagg.config(parallel_threshold=1):  # force parallelize once
        mojagg.allnan(warm, axis=1)
    print("  (max pool now active)")
    numbagg_suite("after-max-pool")

    print()
    print("== Phase C: mojagg timings (same process) ==")
    rs = np.random.RandomState(0)
    big = rs.rand(10_000, 1_000)
    big[rs.rand(10_000, 1_000) < 0.15] = np.nan
    nan8 = np.full((10_000, 8), np.nan)
    with mojagg.config(parallel_threshold=10**15):
        print(
            f"  [mojagg serial ] allnan (10k,1000): {bench_us(mojagg.allnan, big, axis=1):9.1f} us"
        )
        print(
            f"  [mojagg serial ] allnan (10k,8)   : {bench_us(mojagg.allnan, nan8, axis=1):9.1f} us"
        )
    with mojagg.config(parallel_threshold=1):
        print(
            f"  [mojagg par    ] allnan (10k,1000): {bench_us(mojagg.allnan, big, axis=1):9.1f} us"
        )
        print(
            f"  [mojagg par    ] allnan (10k,8)   : {bench_us(mojagg.allnan, nan8, axis=1):9.1f} us"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
