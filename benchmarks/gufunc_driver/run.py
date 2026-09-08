"""Gufunc-driver design benchmark — compares ways to implement the
numba.guvectorize outer loop in Mojo.

Variants (all wrap the same masked-SIMD sum kernel; f64 only):
  numpy            np.nansum reference
  numbagg          numba gufunc (compiled outer loop, parallel target)
  mojagg-current   current public API (Python per-row loop + ascontiguousarray)
  ffi-per-row*     driver-only cost of the current approach: Python loop over a
                   pre-contiguous (outer, n) array calling rows_f64 per row;
                   *setup/copy excluded*
  mojo-contig      single FFI call, Mojo loops rows, SIMD kernel
  mojo-contig-par  mojo-contig + std.algorithm.parallelize over rows
  mojo-view        NuMojo-style: per-row heap view construction (lower bound)
  mojo-strided     fully general N-D odometer driver, no copies, any axis

Run from the repo root inside the pixi env (WSL):
    bash benchmarks/gufunc_driver/run.sh
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sysconfig
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "_build"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

REPS_MIN_TIME = 0.12  # seconds per timing batch
BATCHES = 5


def build() -> object:
    mojo = shutil.which("mojo")
    if not mojo:
        raise SystemExit("mojo not on PATH; run via run.sh (pixi env)")
    BUILD_DIR.mkdir(exist_ok=True)
    out = BUILD_DIR / f"gufunc_bench{EXT_SUFFIX}"
    cmd = [
        mojo,
        "build",
        "--emit",
        "shared-lib",
        "-O3",
        "-o",
        str(out),
        str(HERE / "driver_bench.mojo"),
    ]
    print("build:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    spec = importlib.util.spec_from_file_location("gufunc_bench", out)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench(fn, *, min_time=REPS_MIN_TIME, batches=BATCHES) -> float:
    """Best-of-batches mean runtime in milliseconds."""
    fn()  # warmup
    t0 = time.perf_counter()
    fn()
    est = max(time.perf_counter() - t0, 1e-7)
    reps = min(200, max(3, int(min_time / est)))
    best = float("inf")
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = (time.perf_counter() - t0) / reps
        best = min(best, dt)
    return best * 1e3


def make_arr(shape, seed=1):
    rs = np.random.RandomState(seed)
    a = rs.rand(*shape).astype(np.float64)
    a[rs.rand(*shape) < 0.15] = np.nan
    return np.ascontiguousarray(a)


def main() -> int:
    bench_mod = build()
    import numbagg

    import mojagg

    # (label, shape, axis)
    cases = [
        ("big-rows (10000x1000) axis=1", (10_000, 1_000), 1),
        ("tiny-rows (1000000x8) axis=1", (1_000_000, 8), 1),
        ("strided (1000x10000) axis=0", (1_000, 10_000), 0),
        ("full (2000x5000) axis=None", (2_000, 5_000), None),
    ]

    results = {}  # (case, variant) -> ms
    for label, shape, axis in cases:
        arr = make_arr(shape)
        if axis is None:
            flat = arr.ravel()
            rows2d = flat.reshape(1, flat.size)  # view, free
            contig2d = rows2d
            n = flat.size
            outer = 1
            strided_axis = 0
            strided_arr = flat
            ref = np.nansum(arr)
        else:
            moved = np.moveaxis(arr, axis, -1)
            rows2d = np.ascontiguousarray(moved).reshape(-1, arr.shape[axis])
            outer, n = rows2d.shape
            strided_axis = axis
            strided_arr = arr
            contig2d = arr if (axis == arr.ndim - 1) else None
            ref = np.nansum(arr, axis=axis)

        out = np.empty(outer, dtype=np.float64)

        def ffi_per_row(rows2d=rows2d, out=out):
            for i in range(rows2d.shape[0]):
                out[i] = bench_mod.rows_f64(rows2d[i])
            return out

        variants = {
            "numpy": lambda arr=arr, axis=axis: np.nansum(arr, axis=axis),
            "numbagg": lambda arr=arr, axis=axis: numbagg.nansum(arr, axis=axis),
            "mojagg-current": lambda arr=arr, axis=axis: mojagg.nansum(arr, axis=axis),
            "ffi-per-row*": ffi_per_row,
            "mojo-strided": lambda s=strided_arr, o=out, a=strided_axis: bench_mod.strided_f64(
                s, o, a
            ),
            "mojo-strided-par": lambda s=strided_arr, o=out, a=strided_axis: (
                bench_mod.strided_par_f64(s, o, a)
            ),
        }
        if contig2d is not None:
            variants["mojo-contig"] = lambda c=contig2d, o=out: bench_mod.contig_f64(c, o)
            variants["mojo-contig-par"] = lambda c=contig2d, o=out: bench_mod.contig_par_f64(c, o)
            variants["mojo-view"] = lambda c=contig2d, o=out: bench_mod.view_f64(c, o)

        # correctness first (cheap insurance before trusting timings)
        for name, fn in variants.items():
            got = fn()
            np.testing.assert_allclose(
                np.asarray(got),
                ref,
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"{label} / {name} produced wrong results",
            )

        for name, fn in variants.items():
            results[(label, name)] = bench(fn)

    variant_order = [
        "numpy",
        "numbagg",
        "mojagg-current",
        "ffi-per-row*",
        "mojo-contig",
        "mojo-contig-par",
        "mojo-view",
        "mojo-strided",
        "mojo-strided-par",
    ]
    case_labels = [c[0] for c in cases]

    print()
    print("# Gufunc driver benchmark — nansum, float64, ~15% NaN")
    print()
    print(f"- cpu: {platform.processor() or platform.machine()}")
    print(f"- cores: {os.cpu_count()}")
    print(
        f"- python {platform.python_version()}, numpy {np.__version__}, "
        f"numbagg {numbagg.__version__}"
    )
    v = subprocess.run([shutil.which("mojo"), "--version"], capture_output=True, text=True)
    print(f"- {v.stdout.strip()}")
    print()
    print("## Absolute (ms, lower is better)")
    print()
    header = "| case | " + " | ".join(variant_order) + " |"
    print(header)
    print("|" + "---|" * (len(variant_order) + 1))
    for label in case_labels:
        row = f"| {label} |"
        for name in variant_order:
            ms = results.get((label, name))
            row += f" {ms:.3f} |" if ms is not None else " — |"
        print(row)
    print()
    print("## Relative to numbagg (>1 = slower than numbagg)")
    print()
    print(header)
    print("|" + "---|" * (len(variant_order) + 1))
    for label in case_labels:
        base = results[(label, "numbagg")]
        row = f"| {label} |"
        for name in variant_order:
            ms = results.get((label, name))
            row += f" {ms / base:.2f}x |" if ms is not None else " — |"
        print(row)
    print()
    print("*ffi-per-row: driver cost only; moveaxis/ascontiguousarray copy")
    print("excluded (mojagg-current includes it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
