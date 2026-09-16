"""Compare every implemented reduction on the pinned upstream test corpus.

Run inside the pixi environment with PYTHONPATH=python:
    python -m benchmarks.compare_numbagg --save comparison.json
    python -m benchmarks.compare_numbagg --full --ops nanmean nansum

Default matrices contain 10K elements; --full uses the upstream 1M-element
cases. Allocation, correctness checks and JIT warmup are outside timings.
nanprod is explicitly NumPy-only: numbagg has no standalone nanprod.
count is an alias of nancount and uses the same implementation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import timeit
import warnings
from dataclasses import asdict
from pathlib import Path

import numba
import numbagg
import numpy as np

import mojagg

OPS = (
    "nansum",
    "nanmean",
    "nanmin",
    "nanmax",
    "nanstd",
    "nanvar",
    "nanargmin",
    "nanargmax",
    "nancount",
    "allnan",
    "anynan",
    "nanprod",
    "move_corr",
    "move_cov",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
)
MOVING_OPS = {
    "move_corr",
    "move_cov",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
}
MOVING_BINARY_OPS = {"move_corr", "move_cov"}
DTYPES = ("float32", "float64", "int32", "int64")
PROVENANCE = (
    Path(__file__).resolve().parents[1] / "tests" / "vendor" / "numbagg" / "provenance.json"
)


def corpus_cases(op, dtype, full):
    from tests.vendor.numbagg.util import array_generator, array_iter

    shape = (1000, 1000) if full else (100, 100)
    for a in array_iter(array_generator, op, (np.dtype(dtype).type,)):
        if a.shape == shape and a.dtype == dtype and a.flags.c_contiguous:
            break
    else:
        raise RuntimeError(f"upstream corpus has no {dtype} C-order {shape} case")

    yield "flat", a.reshape(-1), None
    yield "rows", a, -1
    yield "columns", a, 0
    yield "negative-stride", a.reshape(-1)[::-2], None
    multi_shape = (100, 100, 100) if full else (20, 25, 20)
    yield "multi", a.reshape(multi_shape), (0, 2)
    if np.issubdtype(a.dtype, np.floating):
        if op == "allnan":
            yield "all-nan", np.full_like(a, np.nan), -1
        elif op == "anynan":
            yield "no-nan", np.nan_to_num(a), -1


def numpy_product(a, axis=None):
    return np.nanprod(a, axis=axis, dtype=a.dtype)


def reference(op):
    if op == "nanprod":
        return numpy_product, "numpy.nanprod(dtype=input.dtype)"
    return getattr(numbagg, op), f"numbagg.{op}"


def moving_cases(dtype, full):
    specs = (
        ((1_000_000, 8), (1_000_000, 127), (1_000_000, 1024))
        if full
        else (
            (16, 8),
            (64, 8),
            (256, 31),
            (4_096, 127),
            (100_000, 8),
            (100_000, 127),
            (100_000, 1024),
        )
    )
    for length, window in specs:
        values = np.linspace(-1.0, 1.0, length, dtype=np.dtype(dtype))
        values[::19] = np.nan
        yield f"flat-n{length}-window-{window}", values, -1, window


def repetitions(timer, min_time):
    number = 1
    while timer.timeit(number) < min_time:
        number *= 2
    return number


def measure(op, a, axis, rounds, min_time):
    fn = getattr(mojagg, op)
    ref, label = reference(op)
    actual = np.asarray(fn(a, axis=axis))
    expected = np.asarray(ref(a, axis=axis))
    assert actual.shape == expected.shape, (op, actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (op, actual.dtype, expected.dtype)
    if np.issubdtype(actual.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True)
    else:
        np.testing.assert_array_equal(actual, expected)

    timers = (
        timeit.Timer(lambda: fn(a, axis=axis)),
        timeit.Timer(lambda: ref(a, axis=axis)),
    )
    numbers = [repetitions(timer, min_time) for timer in timers]
    samples = ([], [])
    for trial in range(rounds):
        for index in (0, 1) if trial % 2 == 0 else (1, 0):
            samples[index].append(timers[index].timeit(numbers[index]) / numbers[index])
    medians = [statistics.median(sample) * 1e6 for sample in samples]
    return {
        "reference": label,
        "input_shape": a.shape,
        "axis": axis,
        "result_shape": actual.shape,
        "result_dtype": str(actual.dtype),
        "mojagg_us": medians[0],
        "reference_us": medians[1],
        "new_over_reference": statistics.median(
            new / old for new, old in zip(samples[0], samples[1], strict=True)
        ),
        "samples_seconds": {"mojagg": samples[0], "reference": samples[1]},
        "repetitions": numbers,
    }


def measure_moving(op, a, axis, window, rounds, min_time):
    mojagg_fn = getattr(mojagg, op)
    numbagg_fn = getattr(numbagg, op)
    if op in MOVING_BINARY_OPS:
        b = a * 0.5 + 1.0
        actual = np.asarray(mojagg_fn(a, b, axis=axis, window=window, min_count=window))
        expected = np.asarray(numbagg_fn(a, b, axis=axis, window=window, min_count=window))
        timers = (
            timeit.Timer(
                lambda: mojagg_fn(
                    a,
                    b,
                    axis=axis,
                    window=window,
                    min_count=window,
                )
            ),
            timeit.Timer(
                lambda: numbagg_fn(
                    a,
                    b,
                    axis=axis,
                    window=window,
                    min_count=window,
                )
            ),
        )
    else:
        actual = np.asarray(mojagg_fn(a, axis=axis, window=window, min_count=window))
        expected = np.asarray(numbagg_fn(a, axis=axis, window=window, min_count=window))
        timers = (
            timeit.Timer(lambda: mojagg_fn(a, axis=axis, window=window, min_count=window)),
            timeit.Timer(lambda: numbagg_fn(a, axis=axis, window=window, min_count=window)),
        )
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    nan_mismatch = np.flatnonzero(np.isnan(actual) != np.isnan(expected))
    if nan_mismatch.size:
        first = nan_mismatch[:8]
        raise AssertionError(
            f"{op} NaN mask mismatch at {first.tolist()}: "
            f"actual={actual[first].tolist()} expected={expected[first].tolist()}"
        )
    if actual.dtype == np.dtype(np.float32) or op in {"move_std", "move_var"}:
        tolerance = (2e-5, 2e-6)
    else:
        tolerance = (1e-6, 1e-8)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance[0],
        atol=tolerance[1],
        equal_nan=True,
    )

    numbers = [repetitions(timer, min_time) for timer in timers]
    samples = ([], [])
    for trial in range(rounds):
        for index in (0, 1) if trial % 2 == 0 else (1, 0):
            samples[index].append(timers[index].timeit(numbers[index]) / numbers[index])
    medians = [statistics.median(sample) * 1e6 for sample in samples]
    return {
        "reference": f"numbagg.{op}",
        "input_shape": a.shape,
        "axis": axis,
        "window": window,
        "result_shape": actual.shape,
        "result_dtype": str(actual.dtype),
        "mojagg_us": medians[0],
        "reference_us": medians[1],
        "new_over_reference": statistics.median(
            new / old for new, old in zip(samples[0], samples[1], strict=True)
        ),
        "samples_seconds": {"mojagg": samples[0], "reference": samples[1]},
        "repetitions": numbers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--ops", nargs="+", choices=OPS, default=OPS)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DTYPES,
        default=DTYPES,
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-time", type=float, default=0.01)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or not np.isfinite(args.min_time) or args.min_time <= 0:
        parser.error("--rounds and --min-time must be positive and finite")
    report = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numbagg": importlib.metadata.version("numbagg"),
        "numba_threads": numba.get_num_threads(),
        "mojagg_config": asdict(mojagg.get_config()),
        "upstream_tests": (
            json.loads(PROVENANCE.read_text(encoding="utf-8")) if PROVENANCE.exists() else None
        ),
        "full": args.full,
        "cases": {},
    }
    with warnings.catch_warnings(), np.errstate(invalid="ignore", over="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for op in args.ops:
            if op in MOVING_OPS:
                for dtype in (dtype for dtype in args.dtypes if dtype in ("float32", "float64")):
                    for layout, a, axis, window in moving_cases(dtype, args.full):
                        key = f"{op}/{dtype}/{layout}"
                        result = measure_moving(
                            op,
                            a,
                            axis,
                            window,
                            args.rounds,
                            args.min_time,
                        )
                        report["cases"][key] = result
                        print(
                            f"{key:39} {result['mojagg_us']:10.2f} us / "
                            f"{result['reference_us']:10.2f} us "
                            f"({result['new_over_reference']:.3f}x {result['reference']})",
                            flush=True,
                        )
                continue
            for dtype in args.dtypes:
                for layout, a, axis in corpus_cases(op, dtype, args.full):
                    key = f"{op}/{dtype}/{layout}"
                    result = measure(op, a, axis, args.rounds, args.min_time)
                    report["cases"][key] = result
                    print(
                        f"{key:39} {result['mojagg_us']:10.2f} us / "
                        f"{result['reference_us']:10.2f} us "
                        f"({result['new_over_reference']:.3f}x {result['reference']})",
                        flush=True,
                    )
    if args.save:
        args.save.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
