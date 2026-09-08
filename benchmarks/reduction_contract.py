"""Native reduction timings, including early-exit and multi-axis workloads.

Run after build-ext with PYTHONPATH=python:
    python benchmarks/reduction_contract.py --save baseline.json
    python benchmarks/reduction_contract.py --compare baseline.json
    python benchmarks/reduction_contract.py --baseline-library before.so

Reports median microseconds per call. Input/output allocation, axis
normalization and reference calculations are outside the timed region.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import timeit
from pathlib import Path

import numpy as np

from mojagg import _native
from mojagg._reduce import _resolve_axes


def cases():
    for dtype, suffix in (
        (np.float32, "f32"),
        (np.float64, "f64"),
        (np.int32, "i32"),
        (np.int64, "i64"),
    ):
        floating = np.issubdtype(dtype, np.floating)
        for n in (31, 100_003, 1_000_003):
            a = (np.arange(n) % 13).astype(dtype)
            if floating:
                a[::7] = np.nan
            for layout, data in (("contig", a), ("strided", a[::2])):
                yield "nansum", suffix, f"{layout}/{n}", data, None

        for layout, shape, axis in (
            ("contig", (1_000_003,), None),
            ("strided", (100_003, 2), 0),
            ("multi", (1024, 8, 17), (0, 2)),
            ("multi-strided", (1024, 8, 17, 2), (0, 2)),
        ):
            a = np.full(shape, np.nan if floating else 1, dtype=dtype)
            positions = (
                ("none", "first", "last", "each-first", "each-last") if floating else ("integer",)
            )
            for position in positions:
                if position in ("first", "last"):
                    a.fill(np.nan)
                    a.flat[0 if position == "first" else -1] = np.inf
                elif position.startswith("each-"):
                    a.fill(np.nan)
                    axes = _resolve_axes(axis, a)
                    index = 0 if position == "each-first" else -1
                    a[tuple(index if d in axes else slice(None) for d in range(a.ndim))] = np.inf
                yield "allnan", suffix, f"{layout}/{position}", a, axis

        a = (np.arange(1024 * 8 * 17) % 13).astype(dtype).reshape(1024, 8, 17)
        if floating:
            a[::7] = np.nan
        yield "nansum", suffix, "multi", a, (0, 2)


def measure(op, suffix, a, axis, baseline=None, min_time=0.02):
    axes = _resolve_axes(axis, a)
    shape = tuple(s for d, s in enumerate(a.shape) if d not in axes)
    result = np.empty(shape, dtype=np.bool_ if op == "allnan" else a.dtype)
    entry = getattr(_native, f"{op}_{suffix}")

    def call():
        entry(a, axes, result, 2**60)

    call()
    expected = np.isnan(a).all(axis=axis) if op == "allnan" else np.nansum(a, axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)
    timer = timeit.Timer(call)
    number = 1
    while timer.timeit(number) < min_time:
        number *= 2
    if baseline is None:
        samples = [elapsed / number * 1e6 for elapsed in timer.repeat(7, number)]
        return {"median_us": statistics.median(samples), "samples_us": samples}

    before = getattr(baseline, f"{op}_{suffix}")
    old_result = np.empty_like(result)
    before(a, axes, old_result, 2**60)
    np.testing.assert_array_equal(result, old_result)
    old_timer = timeit.Timer(lambda: before(a, axes, old_result, 2**60))
    samples, old_samples, ratios = [], [], []
    for repeat in range(9):
        if repeat % 2:
            old = old_timer.timeit(number)
            new = timer.timeit(number)
        else:
            new = timer.timeit(number)
            old = old_timer.timeit(number)
        samples.append(new / number * 1e6)
        old_samples.append(old / number * 1e6)
        ratios.append(new / old)
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "baseline_us": statistics.median(old_samples),
        "baseline_samples_us": old_samples,
        "paired_ratio": statistics.median(ratios),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--filter", default="", help="case-name substring")
    parser.add_argument("--min-time", type=float, default=0.02)
    args = parser.parse_args()
    if not np.isfinite(args.min_time) or args.min_time <= 0:
        parser.error("--min-time must be finite and positive")
    previous = json.loads(args.compare.read_text())["cases"] if args.compare else {}
    baseline = None
    if args.baseline_library:
        spec = importlib.util.spec_from_file_location(
            "baseline.nanfuncs_native", args.baseline_library
        )
        if spec is None or spec.loader is None:
            parser.error("cannot load baseline extension")
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    results = {}
    for op, suffix, label, a, axis in cases():
        key = f"{op}/{suffix}/{label}"
        if args.filter not in key:
            continue
        result = measure(op, suffix, a, axis, baseline, args.min_time)
        results[key] = result
        ratio = ""
        if baseline is not None:
            ratio = f"  {result['paired_ratio']:.3f}x paired baseline"
        elif key in previous:
            ratio = f"  {result['median_us'] / previous[key]['median_us']:.3f}x baseline"
        print(f"{key:46} {result['median_us']:10.3f} us{ratio}", flush=True)
    if not results:
        parser.error("no cases match --filter")
    if args.save:
        args.save.write_text(
            json.dumps({"platform": platform.platform(), "cases": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
