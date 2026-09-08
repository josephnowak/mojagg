"""Benchmark comparison of ffill, bfill, nancovmatrix, nancorrmatrix vs numbagg."""

from __future__ import annotations

import sys
import timeit
from pathlib import Path

# Ensure mojagg from python/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numbagg
import numpy as np

import mojagg


def run_benchmarks():
    print("=" * 60)
    print("BENCHMARK: mojagg vs numbagg (fill & matrix operations)")
    print("=" * 60)

    # 1. 1D ffill (100k, 5% NaNs)
    arr = np.random.default_rng(0).standard_normal(100_000)
    arr[np.random.default_rng(1).random(100_000) < 0.05] = np.nan
    # warmup
    numbagg.ffill(arr)
    mojagg.ffill(arr)

    t_nb = min(timeit.repeat(lambda: numbagg.ffill(arr), number=20, repeat=5)) / 20 * 1e6
    t_mj = min(timeit.repeat(lambda: mojagg.ffill(arr), number=20, repeat=5)) / 20 * 1e6
    print(
        f"ffill 1D (100k, 5% NaNs):   numbagg = {t_nb:7.1f} µs"
        f" | mojagg = {t_mj:7.1f} µs | speedup = {t_nb / t_mj:5.2f}x"
    )

    # 2. 1D bfill (100k, 5% NaNs)
    numbagg.bfill(arr)
    mojagg.bfill(arr)

    t_nb = min(timeit.repeat(lambda: numbagg.bfill(arr), number=20, repeat=5)) / 20 * 1e6
    t_mj = min(timeit.repeat(lambda: mojagg.bfill(arr), number=20, repeat=5)) / 20 * 1e6
    print(
        f"bfill 1D (100k, 5% NaNs):   numbagg = {t_nb:7.1f} µs"
        f" | mojagg = {t_mj:7.1f} µs | speedup = {t_nb / t_mj:5.2f}x"
    )

    # 3. 2D ffill (500 rows x 1000 cols)
    arr2d = np.random.default_rng(2).standard_normal((500, 1000))
    arr2d[np.random.default_rng(3).random((500, 1000)) < 0.05] = np.nan
    numbagg.ffill(arr2d)
    mojagg.ffill(arr2d)

    t_nb = min(timeit.repeat(lambda: numbagg.ffill(arr2d), number=10, repeat=3)) / 10 * 1e3
    t_mj = min(timeit.repeat(lambda: mojagg.ffill(arr2d), number=10, repeat=3)) / 10 * 1e3
    print(
        f"ffill 2D (500x1000):       numbagg = {t_nb:7.2f} ms"
        f" | mojagg = {t_mj:7.2f} ms | speedup = {t_nb / t_mj:5.2f}x"
    )

    # 4. nancovmatrix (50 vars x 2000 obs)
    mat = np.random.default_rng(4).standard_normal((50, 2000))
    mat[np.random.default_rng(5).random((50, 2000)) < 0.05] = np.nan
    numbagg.nancovmatrix(mat)
    mojagg.nancovmatrix(mat)

    t_nb = min(timeit.repeat(lambda: numbagg.nancovmatrix(mat), number=5, repeat=3)) / 5 * 1e3
    t_mj = min(timeit.repeat(lambda: mojagg.nancovmatrix(mat), number=5, repeat=3)) / 5 * 1e3
    print(
        f"nancovmatrix (50x2000):    numbagg = {t_nb:7.2f} ms"
        f" | mojagg = {t_mj:7.2f} ms | speedup = {t_nb / t_mj:5.2f}x"
    )

    # 5. nancorrmatrix (50 vars x 2000 obs)
    numbagg.nancorrmatrix(mat)
    mojagg.nancorrmatrix(mat)

    t_nb = min(timeit.repeat(lambda: numbagg.nancorrmatrix(mat), number=5, repeat=3)) / 5 * 1e3
    t_mj = min(timeit.repeat(lambda: mojagg.nancorrmatrix(mat), number=5, repeat=3)) / 5 * 1e3
    print(
        f"nancorrmatrix (50x2000):   numbagg = {t_nb:7.2f} ms"
        f" | mojagg = {t_mj:7.2f} ms | speedup = {t_nb / t_mj:5.2f}x"
    )
    print("=" * 60)


def run_matrix_benchmarks():
    print("\n" + "=" * 80)
    print("DETAILED MATRIX BENCHMARKS: mojagg vs numbagg")
    print("=" * 80)
    hdr = (
        f"{'Operation':<14} | {'Shape / Dtype':<19} | {'NaN%':<5} | "
        f"{'Numbagg':<10} | {'mojagg':<10} | {'Speedup':<8}"
    )
    print(hdr)
    print("-" * 80)

    configs = [
        ((20, 1000), np.float64, 0.05),
        ((50, 2000), np.float64, 0.05),
        ((100, 2000), np.float64, 0.05),
        ((30, 10000), np.float64, 0.05),
        ((50, 2000), np.float64, 0.0),
        ((50, 2000), np.float64, 0.20),
        ((50, 2000), np.float32, 0.05),
        ((100, 2000), np.float32, 0.05),
        ((4, 25, 1000), np.float64, 0.05),
    ]

    for shape, dtype, nan_frac in configs:
        rng = np.random.default_rng(42)
        arr_data = rng.standard_normal(shape).astype(dtype)
        if nan_frac > 0:
            arr_data[rng.random(shape) < nan_frac] = np.nan

        # Warmup
        numbagg.nancovmatrix(arr_data)
        mojagg.nancovmatrix(arr_data)
        numbagg.nancorrmatrix(arr_data)
        mojagg.nancorrmatrix(arr_data)

        repeats = 3
        number = 3

        def bench_cov_nb(a=arr_data):
            return numbagg.nancovmatrix(a)

        def bench_cov_mj(a=arr_data):
            return mojagg.nancovmatrix(a)

        def bench_corr_nb(a=arr_data):
            return numbagg.nancorrmatrix(a)

        def bench_corr_mj(a=arr_data):
            return mojagg.nancorrmatrix(a)

        t_nb_cov = min(timeit.repeat(bench_cov_nb, number=number, repeat=repeats)) / number * 1e3
        t_mj_cov = min(timeit.repeat(bench_cov_mj, number=number, repeat=repeats)) / number * 1e3
        sp_cov = t_nb_cov / t_mj_cov

        t_nb_corr = min(timeit.repeat(bench_corr_nb, number=number, repeat=repeats)) / number * 1e3
        t_mj_corr = min(timeit.repeat(bench_corr_mj, number=number, repeat=repeats)) / number * 1e3
        sp_corr = t_nb_corr / t_mj_corr

        dtype_str = "f64" if dtype == np.float64 else "f32"
        shape_str = f"{shape} {dtype_str}"
        nan_str = f"{int(nan_frac * 100)}%"

        line_cov = (
            f"{'nancovmatrix':<14} | {shape_str:<19} | {nan_str:<5} | "
            f"{t_nb_cov:7.2f} ms | {t_mj_cov:7.2f} ms | {sp_cov:6.2f}x"
        )
        line_corr = (
            f"{'nancorrmatrix':<14} | {shape_str:<19} | {nan_str:<5} | "
            f"{t_nb_corr:7.2f} ms | {t_mj_corr:7.2f} ms | {sp_corr:6.2f}x"
        )
        print(line_cov)
        print(line_corr)

    print("=" * 80)


def run_quantile_benchmarks():
    print("\n" + "=" * 80)
    print("BENCHMARK: nanquantile & nanmedian vs numbagg")
    print("=" * 80)
    print(
        f"{'Operation':<14} | {'Input Shape':<19} | {'NaNs':<5} | "
        f"{'Numbagg':<10} | {'mojagg':<10} | {'Speedup':<8}"
    )
    print("-" * 80)

    cases = [
        ((50_000,), 0.05, np.float64, [0.25, 0.5, 0.75], None),
        ((50_000,), 0.05, np.float32, [0.25, 0.5, 0.75], None),
        ((100, 1000), 0.10, np.float64, [0.1, 0.5, 0.9], 1),
        ((100, 1000), 0.10, np.float32, [0.1, 0.5, 0.9], 1),
        ((100, 1000), 0.10, np.float64, 0.5, 1),
    ]

    for shape, nan_frac, dtype, q, axis in cases:
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(shape).astype(dtype)
        if nan_frac > 0:
            arr[rng.random(shape) < nan_frac] = np.nan

        is_median = q == 0.5 and not isinstance(q, list)
        op_name = "nanmedian" if is_median else "nanquantile"

        # Warmup
        if is_median:
            numbagg.nanmedian(arr, axis=axis)
            mojagg.nanmedian(arr, axis=axis)
        else:
            numbagg.nanquantile(arr, q, axis=axis)
            mojagg.nanquantile(arr, q, axis=axis)

        repeats = 3
        number = 5

        def bench_nb(a=arr, q_val=q, ax=axis, med=is_median):
            if med:
                return numbagg.nanmedian(a, axis=ax)
            return numbagg.nanquantile(a, q_val, axis=ax)

        def bench_mj(a=arr, q_val=q, ax=axis, med=is_median):
            if med:
                return mojagg.nanmedian(a, axis=ax)
            return mojagg.nanquantile(a, q_val, axis=ax)

        t_nb = min(timeit.repeat(bench_nb, number=number, repeat=repeats)) / number * 1e3
        t_mj = min(timeit.repeat(bench_mj, number=number, repeat=repeats)) / number * 1e3
        sp = t_nb / t_mj

        dtype_str = "f64" if dtype == np.float64 else "f32"
        shape_str = f"{shape} {dtype_str}"
        nan_str = f"{int(nan_frac * 100)}%"

        print(
            f"{op_name:<14} | {shape_str:<19} | {nan_str:<5} | "
            f"{t_nb:7.2f} ms | {t_mj:7.2f} ms | {sp:6.2f}x"
        )
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
    run_matrix_benchmarks()
    run_quantile_benchmarks()
