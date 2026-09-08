"""Isolate where numbagg beats us on early-exit workloads.

Three suspects, measured independently:
1. parallelize fixed dispatch cost (no-op tasks, fine vs 16-chunk)
2. numba parallel gufunc dispatch floor (numbagg.allnan on tiny arrays)
3. our serial per-slice cost slope (driver loop without the pool)

Run: bash benchmarks/gufunc_driver/run_probe.sh (inside pixi env / WSL)
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sysconfig
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "_build"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def build() -> object:
    mojo = shutil.which("mojo")
    BUILD_DIR.mkdir(exist_ok=True)
    out = BUILD_DIR / f"gufunc_bench{EXT_SUFFIX}"
    subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "shared-lib",
            "-O3",
            "-o",
            str(out),
            str(HERE / "driver_bench.mojo"),
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("gufunc_bench", out)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench_us(fn, *args, min_time=0.3, **kw) -> float:
    """Best mean call time in microseconds."""
    fn(*args, **kw)
    t0 = time.perf_counter()
    fn(*args, **kw)
    est = max(time.perf_counter() - t0, 1e-8)
    reps = min(2000, max(3, int(min_time / est)))
    best = float("inf")
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args, **kw)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e6


def main() -> int:
    mod = build()
    import numbagg

    print("== 1. max.algorithm.parallelize dispatch cost (no-op tasks) ==")
    print(f"  serial loop, 1M iters : {bench_us(mod.noop_serial, 1_000_000):9.1f} us")
    for n in (16, 1_000, 10_000, 100_000, 1_000_000):
        fine = bench_us(mod.noop_par_fine, n)
        print(f"  parallelize {n:>8} no-op tasks: {fine:9.1f} us")
    print(f"  parallelize 16 chunked no-op    : {bench_us(mod.noop_par_chunked, 0):9.1f} us")

    print()
    print("== 2. numba gufunc dispatch floor (numbagg.allnan, tiny arrays) ==")
    for shape, axis in [((1, 1), 1), ((16, 1), 1), ((1000, 1), 1), ((10_000, 1), 1)]:
        a = np.full(shape, 1.0)  # valid data: allnan early-exits instantly
        t = bench_us(numbagg.allnan, a, axis=axis)
        print(f"  numbagg.allnan {str(shape):>12} axis=1: {t:9.1f} us")

    print()
    print("== 3. our serial driver per-slice slope (all-NaN worst case) ==")
    import mojagg

    with mojagg.config(parallel_threshold=10**15):  # force serial
        for rows in (1_000, 10_000, 100_000):
            a = np.full((rows, 8), np.nan)  # worst case: scans all 8 elems
            t = bench_us(mojagg.allnan, a, axis=1)
            print(
                f"  mojagg.allnan ({rows:>7}, 8) serial: {t:9.1f} us"
                f"  ({t * 1000 / rows:6.2f} ns/row)"
            )
            tn = bench_us(numbagg.allnan, a, axis=1)
            print(
                f"  numbagg.allnan same          : {tn:9.1f} us  ({tn * 1000 / rows:6.2f} ns/row)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
