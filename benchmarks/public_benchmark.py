"""Run the manual mojagg performance comparison and build a standalone report.

The public benchmark is deliberately separate from the CodSpeed suite. It is
intended for an occasional, pinned-hardware run (for example on AWS), while
``benchmarks/codspeed`` stays small enough to run on every change.

The configuration is object based so each workload owns all of its data
choices. A typical custom run looks like this::

    from benchmarks.public_benchmark import Public, run_benchmark

    suite = Public()
    suite.reduction_functions = ["nansum", "nanmean"]
    suite.reduction_tests[0].nan_fraction = 0.20
    suite.reduction_tests[0].implementations = ("mojagg", "numbagg")
    suite.groupby_tests[0].num_groups = 4096
    run_benchmark(suite, "docs/benchmarks/latest/index.html")

The generated HTML has no external assets or CDN dependencies. It can be
copied into a static documentation site and opened directly from disk.
"""

from __future__ import annotations

import argparse
import gc
import html
import importlib
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import mojagg

Axis = int | tuple[int, ...] | None
IMPLEMENTATION_NAMES = ("mojagg", "numbagg")
_ONE_GIB_F64_ELEMENTS = (1 << 30) // np.dtype(np.float64).itemsize
_THREE_GIB_F64_ELEMENTS = (3 << 30) // np.dtype(np.float64).itemsize
# Matrix functions allocate a (..., vars, vars) output, which can dwarf the
# input for skewed shapes (e.g. many vars, few obs). Cap the *output* alone
# to this budget so a case can never silently try to allocate tens of GiB.
_MATRIX_OUTPUT_BUDGET_BYTES = 3 << 30

REDUCTION_FUNCTIONS = [
    "allnan",
    "anynan",
    "nanargmax",
    "nanargmin",
    "nancount",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
]

GROUPBY_FUNCTIONS = [
    "group_nanall",
    "group_nanany",
    "group_nanargmax",
    "group_nanargmin",
    "group_nancount",
    "group_nanfirst",
    "group_nanlast",
    "group_nanmax",
    "group_nanmean",
    "group_nanmin",
    "group_nanprod",
    "group_nanstd",
    "group_nansum",
    "group_nansum_of_squares",
    "group_nanvar",
]

MATRIX_FUNCTIONS = ["nancorrmatrix", "nancovmatrix"]
ROLLING_FUNCTIONS = [
    "move_corr",
    "move_cov",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
]
EXPONENTIAL_FUNCTIONS = [
    "move_exp_nancorr",
    "move_exp_nancount",
    "move_exp_nancov",
    "move_exp_nanmean",
    "move_exp_nanstd",
    "move_exp_nansum",
    "move_exp_nanvar",
]
FILL_FUNCTIONS = ["bfill", "ffill"]


@dataclass
class ReductionTest:
    """One reduction workload, including its complete input configuration."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    axis: Axis = 0
    nan_fraction: float = 0.10
    nan_pattern: str = "random"
    seed: int = 0
    ddof: int = 1
    quantiles: tuple[float, ...] = (0.5,)
    implementations: tuple[str, ...] = IMPLEMENTATION_NAMES


@dataclass
class GroupByTest:
    """One grouped reduction workload and its label cardinality controls."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    axis: Axis = 0
    label_shape: tuple[int, ...] | None = None
    label_dtype: str = "int32"
    num_groups: int = 256
    nan_fraction: float = 0.10
    nan_pattern: str = "random"
    seed: int = 0
    ddof: int = 1
    implementations: tuple[str, ...] = IMPLEMENTATION_NAMES


@dataclass
class MatrixTest:
    """One static covariance/correlation matrix workload."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    axis: tuple[int, int] = (0, 1)
    nan_fraction: float = 0.10
    nan_pattern: str = "random"
    seed: int = 0
    implementations: tuple[str, ...] = IMPLEMENTATION_NAMES


def _matrix_output_bytes(test: MatrixTest) -> int:
    """Bytes needed for a matrix test's ``(..., vars, vars)`` output.

    The output is quadratic in the number of variables, so it can dwarf the
    input for skewed shapes (many vars, few obs) even when the input itself
    is small.
    """

    ndim = len(test.shape)
    vars_axis, obs_axis = (a % ndim for a in test.axis)
    vars_count = test.shape[vars_axis]
    batch = 1
    for dim_index, dim_size in enumerate(test.shape):
        if dim_index not in (vars_axis, obs_axis):
            batch *= dim_size
    return batch * vars_count * vars_count * np.dtype(test.dtype).itemsize


def _validate_matrix_tests(tests: list[MatrixTest]) -> None:
    """Guard against matrix cases whose output would blow past the budget.

    ``nancorrmatrix``/``nancovmatrix`` allocate a square ``vars x vars``
    output, so a shape with many "vars" and few "obs" (e.g. a tall matrix)
    can silently require tens of GiB even though the raw input is tiny.
    """

    for test in tests:
        output_bytes = _matrix_output_bytes(test)
        if output_bytes > _MATRIX_OUTPUT_BUDGET_BYTES:
            raise ValueError(
                f"matrix test {test.name!r} would allocate a "
                f"{_format_bytes(output_bytes)} output for shape {test.shape} "
                f"axis={test.axis}; exceeds the "
                f"{_format_bytes(_MATRIX_OUTPUT_BUDGET_BYTES)} output budget. "
                "Reduce the number of vars (the axis[0]/axis[1] dimensions) "
                "for this case."
            )


@dataclass
class RollingTest:
    """One trailing moving-window workload."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    axis: int = 0
    window: int = 32
    min_count: int | None = None
    nan_fraction: float = 0.10
    second_nan_fraction: float | None = None
    nan_pattern: str = "random"
    seed: int = 0
    implementations: tuple[str, ...] = IMPLEMENTATION_NAMES


@dataclass
class ExponentialTest:
    """One exponentially weighted moving workload."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    axis: int = 0
    alpha: float = 0.15
    min_weight: float = 0.0
    nan_fraction: float = 0.10
    second_nan_fraction: float | None = None
    nan_pattern: str = "random"
    seed: int = 0
    implementations: tuple[str, ...] = IMPLEMENTATION_NAMES


@dataclass
class FillTest:
    """One forward/backward fill workload."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float64"
    axis: Axis = 0
    limit: int | None = None
    nan_fraction: float = 0.10
    nan_pattern: str = "blocks"
    seed: int = 0
    implementations: tuple[str, ...] = IMPLEMENTATION_NAMES


@dataclass
class BenchmarkSuite:
    """Composable public benchmark configuration.

    The lists and function selections are intentionally public mutable
    attributes. Construct ``Public`` or ``Quick``, then edit only the cases
    and functions relevant to a run.
    """

    name: str = "custom"
    device_name: str | None = None
    reduction_tests: list[ReductionTest] = field(default_factory=list)
    groupby_tests: list[GroupByTest] = field(default_factory=list)
    matrix_tests: list[MatrixTest] = field(default_factory=list)
    rolling_tests: list[RollingTest] = field(default_factory=list)
    exponential_tests: list[ExponentialTest] = field(default_factory=list)
    fill_tests: list[FillTest] = field(default_factory=list)
    reduction_functions: list[str] = field(default_factory=lambda: list(REDUCTION_FUNCTIONS))
    groupby_functions: list[str] = field(default_factory=lambda: list(GROUPBY_FUNCTIONS))
    matrix_functions: list[str] = field(default_factory=lambda: list(MATRIX_FUNCTIONS))
    rolling_functions: list[str] = field(default_factory=lambda: list(ROLLING_FUNCTIONS))
    exponential_functions: list[str] = field(default_factory=lambda: list(EXPONENTIAL_FUNCTIONS))
    fill_functions: list[str] = field(default_factory=lambda: list(FILL_FUNCTIONS))
    warmups: int = 1
    repeats: int = 5
    verify_results: bool = True
    include_numbagg: bool = True

    def __post_init__(self) -> None:
        _validate_matrix_tests(self.matrix_tests)


class Quick(BenchmarkSuite):
    """A small complete suite suitable for local smoke checks."""

    def __init__(self, **overrides: Any):
        defaults: dict[str, Any] = {
            "name": "quick",
            "reduction_tests": [
                ReductionTest("2d_axis1", (8, 4096), axis=1, nan_fraction=0.10, seed=11),
                ReductionTest(
                    "3d_multi_axis",
                    (4, 16, 128),
                    dtype="float32",
                    axis=(1, 2),
                    nan_fraction=0.20,
                    seed=12,
                ),
            ],
            "groupby_tests": [
                GroupByTest(
                    "1d_128_groups",
                    (16_384,),
                    axis=0,
                    num_groups=128,
                    label_dtype="int32",
                    nan_fraction=0.10,
                    seed=21,
                ),
                GroupByTest(
                    "2d_axis1_64_groups",
                    (8, 2048),
                    axis=1,
                    num_groups=64,
                    label_dtype="int64",
                    nan_fraction=0.20,
                    seed=22,
                ),
            ],
            "matrix_tests": [
                MatrixTest("32_vars_512_obs", (32, 512), axis=(0, 1), seed=31),
                MatrixTest("batched", (4, 16, 256), axis=(1, 2), seed=32),
            ],
            "rolling_tests": [
                RollingTest(
                    "2d_axis1_window32",
                    (8, 4096),
                    axis=1,
                    window=32,
                    min_count=16,
                    nan_fraction=0.10,
                    seed=41,
                )
            ],
            "exponential_tests": [
                ExponentialTest(
                    "2d_axis1_alpha015",
                    (8, 4096),
                    axis=1,
                    alpha=0.15,
                    nan_fraction=0.10,
                    seed=51,
                )
            ],
            "fill_tests": [FillTest("2d_axis1_blocks", (8, 4096), axis=1, limit=32, seed=61)],
            "warmups": 1,
            "repeats": 3,
        }
        defaults.update(overrides)
        super().__init__(**defaults)


class Public(BenchmarkSuite):
    """The default AWS/publication suite.

    It covers contiguous and batched inputs, multiple positive axis layouts,
    several NaN densities, and group cardinality. Cases focus on float64;
    float32 is exercised only on a contiguous case per family so dtype
    coverage does not multiply the case count. It includes 3 GiB float64
    cases for each non-matrix operation family (matrix cases stay at 1 GiB
    because they scale as vars x obs), with matching axis=0 and axis=1
    variants so the non-contiguous axis=0 path is measured at the same scale
    as the contiguous axis=1 path. Cases are prepared serially to keep the
    peak working set suitable for a 10 GiB host. Those large cases compare
    mojagg with numbagg.
    """

    def __init__(self, **overrides: Any):
        defaults: dict[str, Any] = {
            "name": "public",
            "reduction_tests": [
                ReductionTest("1d_clean_f64", (3_000_000,), axis=0, nan_fraction=0.0, seed=101),
                ReductionTest(
                    "1d_contig_clean_f32",
                    (3_000_000,),
                    dtype="float32",
                    axis=0,
                    nan_fraction=0.0,
                    seed=105,
                ),
                ReductionTest(
                    "32x_3gib_clean_f64_axis1",
                    (32, _THREE_GIB_F64_ELEMENTS // 32),
                    axis=1,
                    nan_fraction=0.0,
                    seed=104,
                    implementations=("mojagg", "numbagg"),
                ),
                ReductionTest(
                    "3gib_clean_f64_axis0",
                    (_THREE_GIB_F64_ELEMENTS // 32, 32),
                    axis=0,
                    nan_fraction=0.0,
                    seed=106,
                    implementations=("mojagg", "numbagg"),
                ),
                ReductionTest("2d_nan_f64", (32, 16_384), axis=1, nan_fraction=0.10, seed=102),
                ReductionTest(
                    "3d_multi_axis_f64",
                    (8, 125, 3_000),
                    axis=(1, 2),
                    nan_fraction=0.25,
                    nan_pattern="blocks",
                    seed=103,
                ),
            ],
            "groupby_tests": [
                GroupByTest(
                    "1d_256_groups",
                    (3_000_000,),
                    axis=0,
                    num_groups=256,
                    label_dtype="int32",
                    nan_fraction=0.05,
                    seed=201,
                ),
                GroupByTest(
                    "16x_3gib_16k_groups_axis1",
                    (16, _THREE_GIB_F64_ELEMENTS // 16),
                    axis=1,
                    num_groups=16_384,
                    label_dtype="int32",
                    nan_fraction=0.10,
                    seed=204,
                    implementations=("mojagg", "numbagg"),
                ),
                GroupByTest(
                    "3gib_16k_groups_axis0",
                    (_THREE_GIB_F64_ELEMENTS // 16, 16),
                    axis=0,
                    num_groups=16_384,
                    label_dtype="int32",
                    nan_fraction=0.10,
                    seed=205,
                    implementations=("mojagg", "numbagg"),
                ),
                GroupByTest(
                    "1d_16k_groups",
                    (3_000_000,),
                    axis=0,
                    num_groups=16_384,
                    label_dtype="int64",
                    nan_fraction=0.10,
                    seed=202,
                ),
                GroupByTest(
                    "2d_axis1_1k_groups",
                    (16, 16_384),
                    axis=1,
                    num_groups=1_024,
                    label_dtype="int32",
                    nan_fraction=0.20,
                    seed=203,
                ),
            ],
            "matrix_tests": [
                MatrixTest(
                    "32_vars_32k_obs", (32, 32_768), axis=(0, 1), nan_fraction=0.05, seed=301
                ),
                MatrixTest(
                    "square_1k_vars_1k_obs",
                    (1_024, 1_024),
                    axis=(0, 1),
                    nan_fraction=0.05,
                    seed=304,
                ),
                MatrixTest(
                    "wide_16_vars_64k_obs",
                    (16, 65_536),
                    axis=(0, 1),
                    nan_fraction=0.05,
                    seed=305,
                ),
                MatrixTest(
                    # Keep vars small even though rows >> columns: the
                    # nancorrmatrix/nancovmatrix output is vars x vars, so a
                    # 64k-vars case would try to allocate a 32 GiB array.
                    "tall_8192_vars_128_obs",
                    (8_192, 128),
                    axis=(0, 1),
                    nan_fraction=0.05,
                    seed=306,
                ),
                MatrixTest(
                    "32_vars_1gib_obs",
                    (32, _ONE_GIB_F64_ELEMENTS // 32),
                    axis=(0, 1),
                    nan_fraction=0.05,
                    seed=303,
                    implementations=("mojagg", "numbagg"),
                ),
                MatrixTest(
                    "batched_16_vars_8k_obs",
                    (8, 16, 8_192),
                    axis=(1, 2),
                    nan_fraction=0.15,
                    seed=302,
                ),
            ],
            "rolling_tests": [
                RollingTest(
                    "8x400k_window128",
                    (8, 400_000),
                    axis=1,
                    window=128,
                    min_count=64,
                    nan_fraction=0.05,
                    second_nan_fraction=0.08,
                    seed=401,
                ),
                RollingTest(
                    "1d_contig_window32_f32",
                    (3_000_000,),
                    dtype="float32",
                    axis=0,
                    window=32,
                    min_count=16,
                    nan_fraction=0.05,
                    second_nan_fraction=0.08,
                    seed=402,
                ),
                RollingTest(
                    "16x_3gib_window128_axis1",
                    (16, _THREE_GIB_F64_ELEMENTS // 16),
                    axis=1,
                    window=128,
                    min_count=64,
                    nan_fraction=0.05,
                    second_nan_fraction=0.08,
                    seed=403,
                    implementations=("mojagg", "numbagg"),
                ),
                RollingTest(
                    "3gib_window128_axis0",
                    (_THREE_GIB_F64_ELEMENTS // 16, 16),
                    axis=0,
                    window=128,
                    min_count=64,
                    nan_fraction=0.05,
                    second_nan_fraction=0.08,
                    seed=404,
                    implementations=("mojagg", "numbagg"),
                ),
            ],
            "exponential_tests": [
                ExponentialTest(
                    "8x400k_alpha015",
                    (8, 400_000),
                    axis=1,
                    alpha=0.15,
                    nan_fraction=0.05,
                    second_nan_fraction=0.08,
                    seed=501,
                ),
                ExponentialTest(
                    "16x_3gib_alpha015_axis1",
                    (16, _THREE_GIB_F64_ELEMENTS // 16),
                    axis=1,
                    alpha=0.15,
                    nan_fraction=0.05,
                    second_nan_fraction=0.08,
                    seed=503,
                    implementations=("mojagg", "numbagg"),
                ),
                ExponentialTest(
                    "3gib_alpha015_axis0",
                    (_THREE_GIB_F64_ELEMENTS // 16, 16),
                    axis=0,
                    alpha=0.15,
                    nan_fraction=0.05,
                    second_nan_fraction=0.08,
                    seed=504,
                    implementations=("mojagg", "numbagg"),
                ),
            ],
            "fill_tests": [
                FillTest(
                    "8x625k_axis1_blocks",
                    (8, 625_000),
                    axis=1,
                    limit=128,
                    nan_fraction=0.20,
                    seed=601,
                ),
                FillTest(
                    "16x_3gib_axis1_blocks",
                    (16, _THREE_GIB_F64_ELEMENTS // 16),
                    axis=1,
                    limit=128,
                    nan_fraction=0.20,
                    seed=603,
                    implementations=("mojagg", "numbagg"),
                ),
                FillTest(
                    "3gib_axis0_blocks",
                    (_THREE_GIB_F64_ELEMENTS // 16, 16),
                    axis=0,
                    limit=128,
                    nan_fraction=0.20,
                    seed=604,
                    implementations=("mojagg", "numbagg"),
                ),
                FillTest(
                    "3d_multi_axis_blocks",
                    (8, 125, 3_000),
                    axis=(1, 2),
                    limit=64,
                    nan_fraction=0.25,
                    seed=602,
                ),
            ],
            "warmups": 2,
            "repeats": 5,
        }
        defaults.update(overrides)
        super().__init__(**defaults)


class Stress(Public):
    """A higher-repeat starting point for a dedicated machine."""

    def __init__(self, **overrides: Any):
        defaults: dict[str, Any] = {"name": "stress", "warmups": 2, "repeats": 7}
        defaults.update(overrides)
        super().__init__(**defaults)


@dataclass
class _PreparedCase:
    values: np.ndarray
    second: np.ndarray | None = None
    labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _case_implementations(case: Any) -> tuple[str, ...]:
    configured = getattr(case, "implementations", IMPLEMENTATION_NAMES)
    if isinstance(configured, str):
        raise ValueError(f"{case.name}: implementations must be a sequence of package names")
    try:
        selected = tuple(configured)
    except TypeError as exc:
        raise ValueError(
            f"{case.name}: implementations must be a sequence of package names"
        ) from exc
    if not selected:
        raise ValueError(f"{case.name}: implementations must include mojagg")
    if any(not isinstance(name, str) for name in selected):
        raise ValueError(f"{case.name}: implementations must contain only package names")
    unknown = tuple(name for name in selected if name not in IMPLEMENTATION_NAMES)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"{case.name}: unknown implementation(s): {names}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{case.name}: implementations must not contain duplicates")
    if "mojagg" not in selected:
        raise ValueError(f"{case.name}: implementations must include mojagg")
    return tuple(name for name in IMPLEMENTATION_NAMES if name in selected)


@dataclass
class _Timing:
    median_ns: float
    minimum_ns: float
    p95_ns: float
    median_memory_bytes: float
    median_cpu_wall_ratio: float


def _log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[benchmark {timestamp}] {message}", flush=True)


def _format_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f}{unit}"
        amount /= 1024
    return f"{amount:.1f}TiB"


def _format_duration(ns: float) -> str:
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}ms"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f}s"
    return f"{ns / 1_000_000_000:.2f}s"


def _prepared_bytes(prepared: _PreparedCase) -> int:
    return sum(
        array.nbytes
        for array in (prepared.values, prepared.second, prepared.labels)
        if array is not None
    )


def _process_thread_count() -> int:
    status = _read_text_file("/proc/self/status")
    if status:
        for line in status.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "Threads":
                try:
                    return int(value.strip())
                except ValueError:
                    break
    return threading.active_count()


def _dispatch_dimensions(case: Any) -> tuple[int, int] | None:
    shape = getattr(case, "shape", None)
    axis = getattr(case, "axis", None)
    if shape is None or axis is None and not hasattr(case, "axis"):
        return None
    axes = _axis_tuple(axis, len(shape))
    selected = set(axes)
    inner = math.prod(shape[index] for index in axes)
    outer = math.prod(shape[index] for index in range(len(shape)) if index not in selected)
    return int(outer), int(inner)


def _dispatch_hint(case: Any, mojagg_config: Mapping[str, Any] | None) -> str:
    dimensions = _dispatch_dimensions(case)
    if dimensions is None or mojagg_config is None:
        return ""
    outer, inner = dimensions
    minimum_groups = int(mojagg_config["parallel_min_groups"])
    threshold = int(mojagg_config["parallel_threshold"])
    eligible = outer >= minimum_groups and inner >= threshold
    return (
        f" dispatch_outer={outer:,} dispatch_inner={inner:,}"
        f" parallel_candidate={'yes' if eligible else 'no'}"
        f" (min_outer={minimum_groups:,} min_inner={threshold:,})"
    )


def _run_logged_call(label: str, call: Callable[[], Any]) -> Any:
    _log(f"START {label}")
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    try:
        result = call()
    except Exception as exc:
        elapsed = time.perf_counter_ns() - wall_started
        _log(f"ERROR {label}: {type(exc).__name__}: {exc} after {_format_duration(elapsed)}")
        raise
    elapsed = time.perf_counter_ns() - wall_started
    cpu_elapsed = time.process_time_ns() - cpu_started
    ratio = cpu_elapsed / elapsed if elapsed else 0.0
    _log(
        f"DONE {label} wall={_format_duration(elapsed)}"
        f" cpu/wall={ratio:.2f}x threads_after={_process_thread_count()}"
    )
    return result


def _axis_tuple(axis: Axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    if not axes:
        raise ValueError("benchmark axes cannot be empty")
    if any(value < 0 for value in axes):
        raise ValueError("benchmark workloads use non-negative axes")
    if len(set(axes)) != len(axes) or any(value >= ndim for value in axes):
        raise ValueError(f"invalid benchmark axes {axes} for ndim={ndim}")
    return axes


def _axis_text(axis: Axis) -> str:
    if axis is None:
        return "all"
    if isinstance(axis, tuple):
        return "(" + ", ".join(str(value) for value in axis) + ")"
    return str(axis)


def _validate_fraction(value: float) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"nan_fraction must be between 0 and 1; got {value}")
    return value


def _nan_mask(size: int, fraction: float, pattern: str, rng: np.random.Generator) -> np.ndarray:
    fraction = _validate_fraction(fraction)
    if fraction == 0:
        return np.zeros(size, dtype=bool)
    if fraction == 1:
        return np.ones(size, dtype=bool)
    if pattern == "random":
        return rng.random(size) < fraction
    if pattern == "periodic":
        period = max(2, round(1 / fraction))
        return np.arange(size) % period == 0
    if pattern == "blocks":
        block = max(1, round(1 / fraction))
        period = block * 4
        return np.arange(size) % period < block
    raise ValueError(f"unknown NaN pattern {pattern!r}")


def _make_values(
    shape: tuple[int, ...],
    dtype: str,
    nan_fraction: float,
    nan_pattern: str,
    seed: int,
) -> np.ndarray:
    dtype_obj = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype_obj, np.floating):
        values = rng.normal(loc=1.0, scale=0.25, size=shape).astype(dtype_obj)
        mask = _nan_mask(values.size, nan_fraction, nan_pattern, rng).reshape(shape)
        values[mask] = np.nan
        return values
    if _validate_fraction(nan_fraction):
        raise ValueError(f"NaN fractions require a floating dtype; got {dtype_obj}")
    if np.issubdtype(dtype_obj, np.integer):
        return rng.integers(-100, 100, size=shape, dtype=dtype_obj)
    if dtype_obj == np.dtype(bool):
        return rng.integers(0, 2, size=shape, dtype=np.int8).astype(bool)
    raise TypeError(f"unsupported benchmark dtype {dtype_obj}")


def _ensure_reduction_values(values: np.ndarray, axis: Axis) -> None:
    """Leave one finite value in every reduction slice for arg reductions."""

    if not np.issubdtype(values.dtype, np.floating) or not np.isnan(values).any():
        return
    axes = _axis_tuple(axis, values.ndim)
    moved = np.moveaxis(values, axes, tuple(range(values.ndim - len(axes), values.ndim)))
    flattened = moved.reshape(moved.shape[: -len(axes)] + (-1,))
    flattened[..., 0] = 1.0


def _prepare_reduction(test: ReductionTest) -> _PreparedCase:
    values = _make_values(
        test.shape,
        test.dtype,
        test.nan_fraction,
        test.nan_pattern,
        test.seed,
    )
    _ensure_reduction_values(values, test.axis)
    return _PreparedCase(
        values,
        metadata={
            "shape": list(test.shape),
            "dtype": str(np.dtype(test.dtype)),
            "axis": _axis_text(test.axis),
            "nan_fraction": test.nan_fraction,
            "nan_pattern": test.nan_pattern,
            "ddof": test.ddof,
        },
    )


def _prepare_groupby(test: GroupByTest) -> _PreparedCase:
    values = _make_values(
        test.shape,
        test.dtype,
        test.nan_fraction,
        test.nan_pattern,
        test.seed,
    )
    axes = _axis_tuple(test.axis, values.ndim)
    expected_label_shape = tuple(values.shape[axis] for axis in axes)
    label_shape = expected_label_shape if test.label_shape is None else tuple(test.label_shape)
    if label_shape != expected_label_shape:
        raise ValueError(
            f"{test.name}: label_shape {label_shape} must match reduced shape "
            f"{expected_label_shape}"
        )
    if test.num_groups <= 0:
        raise ValueError(f"{test.name}: num_groups must be positive")
    label_dtype = np.dtype(test.label_dtype)
    if label_dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise ValueError(f"{test.name}: label_dtype must be int32 or int64")
    if test.num_groups > int(np.prod(label_shape, dtype=np.int64)):
        raise ValueError(f"{test.name}: num_groups exceeds the number of label positions")
    rng = np.random.default_rng(test.seed + 10_000)
    labels = rng.integers(0, test.num_groups, size=label_shape, dtype=label_dtype)
    # Populate every group at least once so arg/min/max workloads have useful
    # dense outputs without introducing negative-label compatibility cases.
    labels.reshape(-1)[: test.num_groups] = np.arange(test.num_groups, dtype=label_dtype)
    _ensure_reduction_values(values, test.axis)
    return _PreparedCase(
        values,
        labels=labels,
        metadata={
            "shape": list(test.shape),
            "dtype": str(np.dtype(test.dtype)),
            "axis": _axis_text(test.axis),
            "label_shape": list(label_shape),
            "label_dtype": str(label_dtype),
            "num_groups": test.num_groups,
            "nan_fraction": test.nan_fraction,
            "nan_pattern": test.nan_pattern,
            "ddof": test.ddof,
        },
    )


def _prepare_matrix(test: MatrixTest) -> _PreparedCase:
    values = _make_values(
        test.shape,
        test.dtype,
        test.nan_fraction,
        test.nan_pattern,
        test.seed,
    )
    _ensure_reduction_values(values, test.axis)
    return _PreparedCase(
        values,
        metadata={
            "shape": list(test.shape),
            "dtype": str(np.dtype(test.dtype)),
            "axis": _axis_text(test.axis),
            "nan_fraction": test.nan_fraction,
            "nan_pattern": test.nan_pattern,
        },
    )


def _prepare_pair(
    shape: tuple[int, ...],
    dtype: str,
    nan_fraction: float,
    second_nan_fraction: float | None,
    nan_pattern: str,
    seed: int,
) -> _PreparedCase:
    first = _make_values(shape, dtype, nan_fraction, nan_pattern, seed)
    second = _make_values(
        shape,
        dtype,
        nan_fraction if second_nan_fraction is None else second_nan_fraction,
        nan_pattern,
        seed + 1,
    )
    return _PreparedCase(first, second=second)


def _prepare_rolling(test: RollingTest) -> _PreparedCase:
    prepared = _prepare_pair(
        test.shape,
        test.dtype,
        test.nan_fraction,
        test.second_nan_fraction,
        test.nan_pattern,
        test.seed,
    )
    prepared.metadata = {
        "shape": list(test.shape),
        "dtype": str(np.dtype(test.dtype)),
        "axis": _axis_text(test.axis),
        "window": test.window,
        "min_count": test.min_count,
        "nan_fraction": test.nan_fraction,
        "second_nan_fraction": test.second_nan_fraction,
        "nan_pattern": test.nan_pattern,
    }
    return prepared


def _prepare_exponential(test: ExponentialTest) -> _PreparedCase:
    prepared = _prepare_pair(
        test.shape,
        test.dtype,
        test.nan_fraction,
        test.second_nan_fraction,
        test.nan_pattern,
        test.seed,
    )
    prepared.metadata = {
        "shape": list(test.shape),
        "dtype": str(np.dtype(test.dtype)),
        "axis": _axis_text(test.axis),
        "alpha": test.alpha,
        "min_weight": test.min_weight,
        "nan_fraction": test.nan_fraction,
        "second_nan_fraction": test.second_nan_fraction,
        "nan_pattern": test.nan_pattern,
    }
    return prepared


def _prepare_fill(test: FillTest) -> _PreparedCase:
    values = _make_values(
        test.shape,
        test.dtype,
        test.nan_fraction,
        test.nan_pattern,
        test.seed,
    )
    return _PreparedCase(
        values,
        metadata={
            "shape": list(test.shape),
            "dtype": str(np.dtype(test.dtype)),
            "axis": _axis_text(test.axis),
            "limit": test.limit,
            "nan_fraction": test.nan_fraction,
            "nan_pattern": test.nan_pattern,
        },
    )


def _optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _reduction_call(module: Any, function: str, case: ReductionTest, prepared: _PreparedCase):
    function_obj = getattr(module, function)
    if function == "nanquantile":
        quantiles: float | np.ndarray
        quantiles = case.quantiles[0] if len(case.quantiles) == 1 else np.asarray(case.quantiles)
        return function_obj(prepared.values, quantiles, axis=case.axis)
    kwargs: dict[str, Any] = {"axis": case.axis}
    if function in {"nanvar", "nanstd"}:
        kwargs["ddof"] = case.ddof
    return function_obj(prepared.values, **kwargs)


def _groupby_call(module: Any, function: str, case: GroupByTest, prepared: _PreparedCase):
    function_obj = getattr(module, function)
    kwargs: dict[str, Any] = {"axis": case.axis, "num_labels": case.num_groups}
    if function in {"group_nanvar", "group_nanstd"}:
        kwargs["ddof"] = case.ddof
    return function_obj(prepared.values, prepared.labels, **kwargs)


def _matrix_call(module: Any, function: str, case: MatrixTest, prepared: _PreparedCase):
    return getattr(module, function)(prepared.values, axis=case.axis)


def _numbagg_matrix_call(
    module: Any,
    function: str,
    case: MatrixTest,
    prepared: _PreparedCase,
):
    """Call Numbagg's fixed-layout matrix API for the same logical axes."""

    values = np.moveaxis(prepared.values, case.axis, (-2, -1))
    return getattr(module, function)(values)


def _rolling_call(module: Any, function: str, case: RollingTest, prepared: _PreparedCase):
    kwargs = {"window": case.window, "min_count": case.min_count, "axis": case.axis}
    function_obj = getattr(module, function)
    if function in {"move_corr", "move_cov"}:
        return function_obj(prepared.values, prepared.second, **kwargs)
    return function_obj(prepared.values, **kwargs)


def _exponential_call(
    module: Any,
    function: str,
    case: ExponentialTest,
    prepared: _PreparedCase,
):
    kwargs = {"alpha": case.alpha, "min_weight": case.min_weight, "axis": case.axis}
    function_obj = getattr(module, function)
    if function in {"move_exp_nancorr", "move_exp_nancov"}:
        return function_obj(prepared.values, prepared.second, **kwargs)
    return function_obj(prepared.values, **kwargs)


def _fill_call(module: Any, function: str, case: FillTest, prepared: _PreparedCase):
    return getattr(module, function)(prepared.values, limit=case.limit, axis=case.axis)


def _compare_outputs(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    try:
        return bool(np.allclose(left_array, right_array, equal_nan=True, rtol=2e-5, atol=2e-6))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def _measure(call: Callable[[], Any], warmups: int, repeats: int, label: str) -> _Timing:
    for warmup_index in range(warmups):
        _run_logged_call(f"{label} warmup {warmup_index + 1}/{warmups}", call)
    gc.collect()
    times: list[float] = []
    cpu_wall_ratios: list[float] = []
    memories: list[float] = []
    tracemalloc.start()
    try:
        for repeat_index in range(repeats):
            gc.collect()
            current_before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            started = time.perf_counter_ns()
            cpu_started = time.process_time_ns()
            result = call()
            elapsed = float(time.perf_counter_ns() - started)
            cpu_elapsed = float(time.process_time_ns() - cpu_started)
            _, peak = tracemalloc.get_traced_memory()
            times.append(elapsed)
            cpu_wall_ratios.append(cpu_elapsed / elapsed if elapsed else 0.0)
            memories.append(float(max(0, peak - current_before)))
            del result
            _log(
                f"DONE {label} repeat {repeat_index + 1}/{repeats}"
                f" wall={_format_duration(elapsed)}"
                f" cpu/wall={cpu_wall_ratios[-1]:.2f}x"
                f" python_peak={_format_bytes(memories[-1])}"
                f" threads_after={_process_thread_count()}"
            )
    finally:
        tracemalloc.stop()
    ordered = sorted(times)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return _Timing(
        median_ns=float(statistics.median(times)),
        minimum_ns=float(min(times)),
        p95_ns=float(ordered[p95_index]),
        median_memory_bytes=float(statistics.median(memories)),
        median_cpu_wall_ratio=float(statistics.median(cpu_wall_ratios)),
    )


def _record(
    section: str,
    function: str,
    case_name: str,
    case_metadata: Mapping[str, Any],
    implementation: str,
    status: str,
    timing: _Timing | None = None,
    error: str | None = None,
    verification: str | None = None,
) -> dict[str, Any]:
    return {
        "section": section,
        "function": function,
        "case": case_name,
        "case_metadata": dict(case_metadata),
        "implementation": implementation,
        "status": status,
        "error": error,
        "verification": verification,
        "time_ns": None if timing is None else timing.median_ns,
        "minimum_ns": None if timing is None else timing.minimum_ns,
        "p95_ns": None if timing is None else timing.p95_ns,
        "memory_bytes": None if timing is None else timing.median_memory_bytes,
        "cpu_wall_ratio": None if timing is None else timing.median_cpu_wall_ratio,
        "time_ratio": None,
        "memory_ratio": None,
    }


def _run_adapter(
    section: str,
    function: str,
    case_name: str,
    case_metadata: Mapping[str, Any],
    implementation: str,
    call: Callable[[], Any] | None,
    baseline_output: Any | None,
    suite: BenchmarkSuite,
) -> dict[str, Any]:
    label = f"{section}/{case_name}/{function}/{implementation}"
    if call is None:
        _log(f"SKIP {label}: adapter unavailable")
        return _record(
            section,
            function,
            case_name,
            case_metadata,
            implementation,
            "na",
            error="adapter unavailable",
        )
    try:
        _log(
            f"BEGIN {label} warmups={suite.warmups} repeats={suite.repeats}"
            f" verify={suite.verify_results and baseline_output is not None and implementation != 'mojagg'}"
        )
        verification = None
        if suite.verify_results and baseline_output is not None and implementation != "mojagg":
            candidate = _run_logged_call(f"{label} verification", call)
            verification = "pass" if _compare_outputs(baseline_output, candidate) else "mismatch"
            if verification == "mismatch":
                _log(f"DIVERGE {label}: results did not match mojagg, timing still measured")
        timing = _measure(call, suite.warmups, suite.repeats, label)
        status = "diverge" if verification == "mismatch" else "ok"
        record = _record(
            section,
            function,
            case_name,
            case_metadata,
            implementation,
            status,
            timing=timing,
            error=None if status == "ok" else "result mismatch against mojagg",
            verification=verification,
        )
        _log(
            f"END {label} status={status} median={_format_duration(timing.median_ns)}"
            f" p95={_format_duration(timing.p95_ns)}"
            f" cpu/wall={timing.median_cpu_wall_ratio:.2f}x"
        )
        return record
    except Exception as exc:  # keep one unsupported adapter from hiding the rest
        _log(f"ERROR {label}: {type(exc).__name__}: {exc}")
        return _record(
            section,
            function,
            case_name,
            case_metadata,
            implementation,
            "error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _add_ratios(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["section"], record["function"], record["case"])
        grouped.setdefault(key, []).append(record)
    comparable_statuses = ("ok", "diverge")
    for rows in grouped.values():
        baseline = next((row for row in rows if row["implementation"] == "mojagg"), None)
        if baseline is None or baseline["status"] not in comparable_statuses:
            continue
        baseline_time = baseline["time_ns"]
        baseline_memory = baseline["memory_bytes"]
        for row in rows:
            if row["status"] not in comparable_statuses:
                continue
            if baseline_time and row["time_ns"] is not None:
                row["time_ratio"] = row["time_ns"] / baseline_time
            if baseline_memory and row["memory_bytes"] is not None:
                row["memory_ratio"] = row["memory_bytes"] / baseline_memory


def _run_family(
    section: str,
    functions: Sequence[str],
    cases: Iterable[Any],
    prepare: Callable[[Any], _PreparedCase],
    call_for: Callable[[Any, str, Any, _PreparedCase], Any],
    mojagg: Any,
    numbagg: Any | None,
    suite: BenchmarkSuite,
    mojagg_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    case_items = list(cases)
    family_started = time.perf_counter_ns()
    _log(
        f"FAMILY {section} start cases={len(case_items)} functions={len(functions)}"
        " implementation_selection=per-case"
    )
    for case_index, case in enumerate(case_items, start=1):
        selected_implementations = _case_implementations(case)
        _log(
            f"CASE {section} {case_index}/{len(case_items)} {case.name}"
            f" prepare shape={getattr(case, 'shape', '?')}"
        )
        prepared = prepare(case)
        _log(
            f"READY {section} {case_index}/{len(case_items)} {case.name}"
            f" data={_format_bytes(_prepared_bytes(prepared))}"
            f" dtype={prepared.metadata.get('dtype', '?')}"
            f" implementations={','.join(selected_implementations)}"
            f"{_dispatch_hint(case, mojagg_config)}"
        )
        case_metadata = dict(prepared.metadata)
        case_metadata["implementations"] = list(selected_implementations)
        for function in functions:
            baseline_output: Any | None = None
            try:
                baseline_output = _run_logged_call(
                    f"{section}/{case.name}/{function}/mojagg baseline",
                    lambda function=function, case=case, prepared=prepared: call_for(
                        mojagg, function, case, prepared
                    ),
                )
                records.append(
                    _run_adapter(
                        section,
                        function,
                        case.name,
                        case_metadata,
                        "mojagg",
                        lambda function=function, case=case, prepared=prepared: call_for(
                            mojagg, function, case, prepared
                        ),
                        None,
                        suite,
                    )
                )
            except AttributeError:
                _log(f"SKIP {section}/{case.name}/{function}/mojagg: function unavailable")
                records.append(
                    _record(
                        section,
                        function,
                        case.name,
                        case_metadata,
                        "mojagg",
                        "na",
                        error="function unavailable",
                    )
                )
            except Exception as exc:
                _log(f"ERROR {section}/{case.name}/{function}/mojagg: {type(exc).__name__}: {exc}")
                records.append(
                    _record(
                        section,
                        function,
                        case.name,
                        case_metadata,
                        "mojagg",
                        "error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                baseline_output = None

            for implementation, module in (("numbagg", numbagg),):
                if implementation not in selected_implementations:
                    _log(
                        f"SKIP {section}/{case.name}/{function}/{implementation}: disabled for case"
                    )
                    records.append(
                        _record(
                            section,
                            function,
                            case.name,
                            case_metadata,
                            implementation,
                            "na",
                            error="disabled for case",
                        )
                    )
                    continue
                if module is None:
                    reason = "package unavailable" if suite.include_numbagg else "disabled by suite"
                    _log(f"SKIP {section}/{case.name}/{function}/{implementation}: {reason}")
                    records.append(
                        _record(
                            section,
                            function,
                            case.name,
                            case_metadata,
                            implementation,
                            "na",
                            error=reason,
                        )
                    )
                    continue
                try:
                    numbagg_call = _numbagg_matrix_call if section == "matrix" else call_for

                    def call(
                        function=function,
                        case=case,
                        prepared=prepared,
                        numbagg_call=numbagg_call,
                    ):
                        return numbagg_call(numbagg, function, case, prepared)

                    _run_logged_call(
                        f"{section}/{case.name}/{function}/{implementation} probe", call
                    )
                except (AttributeError, NotImplementedError):
                    _log(
                        f"SKIP {section}/{case.name}/{function}/{implementation}"
                        ": adapter unavailable"
                    )
                    call = None
                except TypeError as exc:
                    if "cannot specify both 'axis' and 'axes'" in str(exc):
                        _log(
                            f"SKIP {section}/{case.name}/{function}/{implementation}"
                            ": incompatible axis arguments"
                        )
                        call = None
                    else:
                        _log(
                            f"ERROR {section}/{case.name}/{function}/{implementation}:"
                            f" {type(exc).__name__}: {exc}"
                        )
                        records.append(
                            _record(
                                section,
                                function,
                                case.name,
                                case_metadata,
                                implementation,
                                "error",
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue
                except Exception as exc:
                    _log(
                        f"ERROR {section}/{case.name}/{function}/{implementation}:"
                        f" {type(exc).__name__}: {exc}"
                    )
                    records.append(
                        _record(
                            section,
                            function,
                            case.name,
                            case_metadata,
                            implementation,
                            "error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                records.append(
                    _run_adapter(
                        section,
                        function,
                        case.name,
                        case_metadata,
                        implementation,
                        call,
                        baseline_output,
                        suite,
                    )
                )
        _log(f"CASE {section} {case_index}/{len(case_items)} {case.name} complete")
    _log(
        f"FAMILY {section} complete records={len(records)}"
        f" elapsed={_format_duration(time.perf_counter_ns() - family_started)}"
    )
    return records


def _suite_records(
    suite: BenchmarkSuite,
    mojagg_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    numbagg = _optional_import("numbagg") if suite.include_numbagg else None
    records: list[dict[str, Any]] = []
    records.extend(
        _run_family(
            "reduction",
            suite.reduction_functions,
            suite.reduction_tests,
            _prepare_reduction,
            _reduction_call,
            mojagg,
            numbagg,
            suite,
            mojagg_config,
        )
    )
    records.extend(
        _run_family(
            "groupby",
            suite.groupby_functions,
            suite.groupby_tests,
            _prepare_groupby,
            _groupby_call,
            mojagg,
            numbagg,
            suite,
            mojagg_config,
        )
    )
    records.extend(
        _run_family(
            "matrix",
            suite.matrix_functions,
            suite.matrix_tests,
            _prepare_matrix,
            _matrix_call,
            mojagg,
            numbagg,
            suite,
            mojagg_config,
        )
    )
    records.extend(
        _run_family(
            "rolling",
            suite.rolling_functions,
            suite.rolling_tests,
            _prepare_rolling,
            _rolling_call,
            mojagg,
            numbagg,
            suite,
            mojagg_config,
        )
    )
    records.extend(
        _run_family(
            "exponential",
            suite.exponential_functions,
            suite.exponential_tests,
            _prepare_exponential,
            _exponential_call,
            mojagg,
            numbagg,
            suite,
            mojagg_config,
        )
    )
    records.extend(
        _run_family(
            "non-reduction",
            suite.fill_functions,
            suite.fill_tests,
            _prepare_fill,
            _fill_call,
            mojagg,
            numbagg,
            suite,
            mojagg_config,
        )
    )
    _add_ratios(records)
    return records


def _read_text_file(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return None


def _cpu_model() -> str:
    cpuinfo = _read_text_file("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware"} and value.strip():
                return value.strip()
    return platform.processor() or platform.machine() or "unknown"


def _physical_cpu_count() -> int | None:
    cpuinfo = _read_text_file("/proc/cpuinfo")
    if not cpuinfo:
        return None
    pairs: set[tuple[str, str]] = set()
    for block in cpuinfo.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip().lower()] = value.strip()
        physical_id = fields.get("physical id")
        core_id = fields.get("core id")
        if physical_id is not None and core_id is not None:
            pairs.add((physical_id, core_id))
    return len(pairs) or None


def _memory_total_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and pages > 0:
            return page_size * pages
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    meminfo = _read_text_file("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(float(parts[1]) * 1024)
                    except ValueError:
                        pass
    return None


def _process_affinity() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _device_profile(device_name: str | None = None) -> dict[str, Any]:
    logical_cores = os.cpu_count()
    affinity_cores = _process_affinity()
    physical_cores = _physical_cpu_count()
    memory_total = _memory_total_bytes()
    override = (device_name or os.environ.get("MOJAGG_BENCHMARK_DEVICE_NAME", "")).strip()
    identity = override or "benchmark host"
    thread_environment = {
        name: os.environ[name]
        for name in (
            "MOJAGG_THREADS",
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if name in os.environ
    }
    return {
        "identity": identity,
        "provider": "unspecified",
        "cpu": {
            "model": _cpu_model(),
            "architecture": platform.machine() or "unknown",
            "logical_cores": logical_cores,
            "physical_cores": physical_cores,
            "affinity_cores": affinity_cores,
        },
        "memory": {"total_bytes": memory_total},
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pid": os.getpid(),
            "thread_environment": thread_environment,
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _suite_metadata(suite: BenchmarkSuite) -> dict[str, Any]:
    return _json_safe(
        {
            "name": suite.name,
            "device_name": suite.device_name
            or os.environ.get("MOJAGG_BENCHMARK_DEVICE_NAME")
            or None,
            "warmups": suite.warmups,
            "repeats": suite.repeats,
            "verify_results": suite.verify_results,
            "include_numbagg": suite.include_numbagg,
            "functions": {
                "reduction": suite.reduction_functions,
                "groupby": suite.groupby_functions,
                "matrix": suite.matrix_functions,
                "rolling": suite.rolling_functions,
                "exponential": suite.exponential_functions,
                "non-reduction": suite.fill_functions,
            },
            "case_counts": {
                "reduction": len(suite.reduction_tests),
                "groupby": len(suite.groupby_tests),
                "matrix": len(suite.matrix_tests),
                "rolling": len(suite.rolling_tests),
                "exponential": len(suite.exponential_tests),
                "non-reduction": len(suite.fill_tests),
            },
        }
    )


def _resolve_mojagg_config() -> dict[str, Any]:
    try:
        from mojagg.config import get_config

        cfg = get_config()
        mojagg_config = {
            "threads": cfg.threads,
            "parallel_threshold": cfg.parallel_threshold,
            "parallel_min_groups": cfg.parallel_min_groups,
            "backend": cfg.backend,
            "gpu_min_bytes": cfg.gpu_min_bytes,
            "simd_width": cfg.simd_width,
        }
    except Exception:
        mojagg_config = {
            "threads": 0,
            "parallel_threshold": 200_000,
            "parallel_min_groups": 16,
            "backend": "auto",
            "gpu_min_bytes": 1 << 26,
            "simd_width": 0,
        }
    return mojagg_config


def _log_run_start(suite: BenchmarkSuite, mojagg_config: Mapping[str, Any]) -> None:
    case_count = sum(
        len(cases)
        for cases in (
            suite.reduction_tests,
            suite.groupby_tests,
            suite.matrix_tests,
            suite.rolling_tests,
            suite.exponential_tests,
            suite.fill_tests,
        )
    )
    function_count = sum(
        len(functions)
        for functions in (
            suite.reduction_functions,
            suite.groupby_functions,
            suite.matrix_functions,
            suite.rolling_functions,
            suite.exponential_functions,
            suite.fill_functions,
        )
    )
    thread_environment = {
        name: os.environ[name]
        for name in (
            "MOJAGG_THREADS",
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if name in os.environ
    }
    configured_threads = int(mojagg_config["threads"])
    effective_threads = (
        f"auto physical={_physical_cpu_count() or '?'}"
        if configured_threads == 0
        else str(configured_threads)
    )
    _log(
        f"RUN start suite={suite.name} cases={case_count} functions={function_count}"
        f" warmups={suite.warmups} repeats={suite.repeats}"
        f" verify={suite.verify_results}"
    )
    _log(
        f"HOST pid={os.getpid()} logical={os.cpu_count() or '?'}"
        f" physical={_physical_cpu_count() or '?'} affinity={_process_affinity() or '?'}"
        f" process_threads={_process_thread_count()}"
    )
    _log(
        f"MOJAGG backend={mojagg_config['backend']} configured_threads={configured_threads}"
        f" effective_threads={effective_threads}"
        f" parallel_threshold={mojagg_config['parallel_threshold']:,}"
        f" parallel_min_groups={mojagg_config['parallel_min_groups']}"
        f" simd_width={mojagg_config['simd_width']}"
    )
    _log(
        "CPU/wall is process CPU time divided by elapsed time; values near 1x"
        " indicate serial execution, while higher values indicate CPU parallelism."
    )
    if thread_environment:
        _log(f"THREAD_ENV {thread_environment}")


def build_report(suite: BenchmarkSuite) -> dict[str, Any]:
    """Run ``suite`` and return the JSON-serializable report model."""

    if suite.warmups < 0 or suite.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    mojagg_config = _resolve_mojagg_config()
    _log_run_start(suite, mojagg_config)
    records = _suite_records(suite, mojagg_config)
    _log(f"RUN records complete count={len(records)}")
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": "-".join(
                value
                for value in (platform.system(), platform.release(), platform.machine())
                if value
            ),
            "processor": platform.processor() or "unknown",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
        "device": _device_profile(suite.device_name),
        "suite": _suite_metadata(suite),
        "mojagg_config": mojagg_config,
        "measurement": {
            "time": "median wall-clock time across measured calls",
            "memory": "median peak Python-tracked allocation during a call (tracemalloc)",
            "cpu_wall_ratio": "median process CPU time divided by wall-clock time",
            "ratio": "implementation divided by mojagg; values below 1 are better for mojagg",
        },
        "records": records,
    }


def _render_html(report: Mapping[str, Any]) -> str:
    embedded = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("</", "<\\/")
    title = html.escape(f"Mojagg benchmark · {report['suite']['name']}")
    stylesheet = r"""
:root {
  --bg: #07111f;
  --panel: #0e1d31;
  --panel2: #12263d;
  --panel3: #162f4c;
  --line: #26445f;
  --text: #e9f2ff;
  --muted: #90a9c2;
  --mint: #55e6bd;
  --blue: #6da8ff;
  --amber: #ffc857;
  --rose: #ff7f9f;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--text);
  background: radial-gradient(circle at 10% -10%, #183c5a 0, #07111f 42%), var(--bg);
  line-height: 1.45;
}
a { color: var(--mint); text-decoration: none; }
.hero {
  padding: 36px max(24px, calc((100vw - 1560px) / 2)) 28px;
  background: linear-gradient(135deg, rgba(22, 64, 93, 0.8), rgba(7, 17, 31, 0.2) 60%);
  border-bottom: 1px solid rgba(109, 168, 255, 0.18);
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--mint);
  font-size: 0.72rem;
  font-weight: 800;
}
.hero h1 {
  font-size: clamp(2rem, 4.2vw, 3.8rem);
  line-height: 1.02;
  max-width: 820px;
  margin: 10px 0 12px;
  letter-spacing: -0.05em;
}
.hero p {
  max-width: 850px;
  color: var(--muted);
  font-size: 0.98rem;
  margin: 0;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}
.pill {
  border: 1px solid var(--line);
  background: rgba(14, 29, 49, 0.72);
  border-radius: 999px;
  padding: 6px 11px;
  color: #cfe2f7;
  font-size: 0.75rem;
}
.pill-badge {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 6px;
  background: rgba(38, 68, 95, 0.55);
  border: 1px solid rgba(109, 168, 255, 0.2);
  font-size: 0.7rem;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  color: #cfe2f7;
}

/* App Layout with Sticky Left Index */
.app-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  max-width: 1560px;
  margin: 0 auto;
  padding: 22px 24px 72px;
}
.sidebar {
  width: 245px;
  flex-shrink: 0;
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 32px);
  overflow-y: auto;
  background: rgba(14, 29, 49, 0.88);
  backdrop-filter: blur(16px);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
  scrollbar-width: thin;
  z-index: 5;
}
.sidebar-title {
  font-size: 0.68rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--mint);
  margin: 12px 0 6px 6px;
}
.sidebar-title:first-child {
  margin-top: 0;
}
.sidebar-nav {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.sidebar-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  border-radius: 9px;
  color: var(--muted);
  font-size: 0.77rem;
  font-weight: 550;
  transition: all 0.15s ease;
}
.sidebar-link:hover {
  color: var(--text);
  background: rgba(109, 168, 255, 0.08);
}
.sidebar-link.active {
  color: var(--mint);
  background: rgba(85, 230, 189, 0.12);
  font-weight: 700;
  border-left: 3px solid var(--mint);
}
.sidebar-badge {
  font-size: 0.67rem;
  padding: 1px 6px;
  border-radius: 6px;
  background: rgba(38, 68, 95, 0.6);
  color: #cfe2f7;
  font-weight: 600;
}
.sidebar-divider {
  height: 1px;
  background: rgba(38, 68, 95, 0.5);
  margin: 10px 4px;
}
.app-main {
  flex: 1 1 0%;
  min-width: 0;
}

/* Toolbar */
.toolbar {
  position: sticky;
  top: 12px;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: 12px 16px;
  background: rgba(14, 29, 49, 0.92);
  backdrop-filter: blur(18px);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
  margin-bottom: 22px;
}
.search {
  flex: 1 1 240px;
  min-width: 180px;
  background: #081625;
  border: 1px solid var(--line);
  color: var(--text);
  padding: 9px 13px;
  border-radius: 10px;
  outline: none;
  font-size: 0.82rem;
}
.search:focus {
  border-color: var(--mint);
}
.check {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  font-size: 0.82rem;
  color: var(--text);
  cursor: pointer;
}
.check input {
  accent-color: var(--mint);
}

/* Summary stats */
.summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin: 0 0 24px;
}
.stat {
  background: linear-gradient(145deg, rgba(18, 38, 61, 0.96), rgba(10, 25, 42, 0.96));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 13px 35px rgba(0, 0, 0, 0.18);
}
.stat .label {
  font-size: 0.74rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.stat .value {
  font-size: 1.9rem;
  font-weight: 850;
  letter-spacing: -0.05em;
  margin-top: 4px;
}

/* Top 3 Champions Showcase */
.showcase-panel {
  background: linear-gradient(145deg, rgba(18, 38, 61, 0.96), rgba(10, 25, 42, 0.96));
  border: 1px solid var(--line);
  border-radius: 20px;
  margin: 0 0 26px;
  padding: 22px;
  box-shadow: 0 15px 48px rgba(0, 0, 0, 0.18);
}
.showcase-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 18px;
}
.showcase-head h2 {
  margin: 4px 0 2px;
  font-size: clamp(1.25rem, 2.5vw, 1.8rem);
  letter-spacing: -0.04em;
}
.showcase-head p {
  color: var(--muted);
  margin: 0;
  font-size: 0.82rem;
  max-width: 800px;
}
.top3-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.top3-card {
  background: rgba(7, 17, 31, 0.45);
  border: 1px solid rgba(38, 68, 95, 0.8);
  border-radius: 15px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.top3-card-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(38, 68, 95, 0.5);
  padding-bottom: 8px;
}
.top3-card-title h3 {
  font-size: 0.92rem;
  margin: 0;
  color: #cfe2f7;
  text-transform: capitalize;
}
.top3-card-title span {
  font-size: 0.72rem;
  color: var(--muted);
}
.top3-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid rgba(38, 68, 95, 0.3);
}
.top3-item:last-child {
  border-bottom: none;
  padding-bottom: 0;
}
.top3-rank {
  font-size: 1.1rem;
  line-height: 1;
  flex-shrink: 0;
}
.top3-details {
  flex: 1 1 0%;
  min-width: 0;
}
.top3-func {
  font-weight: 700;
  font-size: 0.82rem;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.top3-meta {
  font-size: 0.68rem;
  color: var(--muted);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.top3-stat {
  text-align: right;
  flex-shrink: 0;
}
.top3-speedup {
  font-size: 0.88rem;
  font-weight: 800;
  color: var(--mint);
  white-space: nowrap;
}
.top3-times {
  font-size: 0.64rem;
  color: var(--muted);
  margin-top: 2px;
  white-space: nowrap;
}

/* Scaling Trends & General Graphics */
.scaling-panel {
  background: linear-gradient(145deg, rgba(18, 38, 61, 0.96), rgba(10, 25, 42, 0.96));
  border: 1px solid var(--line);
  border-radius: 20px;
  margin: 0 0 26px;
  padding: 22px;
  box-shadow: 0 15px 48px rgba(0, 0, 0, 0.18);
}
.scaling-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
  gap: 16px;
  margin-top: 16px;
}
.scaling-chart-card {
  background: rgba(7, 17, 31, 0.45);
  border: 1px solid rgba(38, 68, 95, 0.8);
  border-radius: 15px;
  padding: 16px;
}
.scaling-chart-card h3 {
  font-size: 0.9rem;
  margin: 0 0 4px;
  color: #cfe2f7;
}
.scaling-chart-card p {
  font-size: 0.76rem;
  color: var(--muted);
  margin: 0 0 12px;
}

/* Device profile */
.device-panel {
  background: linear-gradient(145deg, rgba(18, 38, 61, 0.96), rgba(10, 25, 42, 0.96));
  border: 1px solid var(--line);
  border-radius: 22px;
  margin: 0 0 26px;
  padding: 20px;
  box-shadow: 0 15px 48px rgba(0, 0, 0, 0.18);
}
.device-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  flex-wrap: wrap;
}
.device-head h2 {
  margin: 4px 0 3px;
  font-size: clamp(1.35rem, 3vw, 2.1rem);
  letter-spacing: -0.04em;
}
.device-head p {
  color: var(--muted);
  margin: 0;
  max-width: 800px;
}
.device-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}
.device-card {
  background: rgba(7, 17, 31, 0.38);
  border: 1px solid rgba(38, 68, 95, 0.8);
  border-radius: 15px;
  padding: 14px;
  min-width: 0;
}
.device-card h3 {
  font-size: 0.9rem;
  margin: 0 0 10px;
  color: #d9eaff;
}
.device-field {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  border-top: 1px solid rgba(38, 68, 95, 0.48);
  padding: 7px 0;
  font-size: 0.76rem;
}
.device-field:first-of-type {
  border-top: 0;
  padding-top: 0;
}
.device-field span:first-child {
  color: var(--muted);
}
.device-field strong {
  font-weight: 650;
  text-align: right;
  overflow-wrap: anywhere;
}
.device-env {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.env-pill {
  border: 1px solid var(--line);
  background: #0a192b;
  border-radius: 8px;
  padding: 5px 7px;
  font: 600 0.7rem ui-monospace, SFMono-Regular, Consolas, monospace;
  color: #cfe2f7;
}
.device-raw {
  margin-top: 14px;
  border-top: 1px solid rgba(38, 68, 95, 0.7);
  padding-top: 13px;
}
.device-raw summary {
  cursor: pointer;
  color: var(--mint);
  font-size: 0.8rem;
  font-weight: 700;
}
.raw-json {
  overflow: auto;
  max-height: 320px;
  margin: 10px 0 0;
  padding: 12px;
  border-radius: 10px;
  background: #081625;
  color: #cfe2f7;
  font: 0.72rem/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
  white-space: pre;
}
.device-note {
  font-size: 0.73rem;
  color: var(--muted);
  margin-top: 12px;
}

/* Benchmark Sections */
.section-panel {
  background: linear-gradient(145deg, rgba(18, 38, 61, 0.96), rgba(10, 25, 42, 0.96));
  border: 1px solid var(--line);
  border-radius: 22px;
  margin: 22px 0 28px;
  padding: 22px;
  box-shadow: 0 15px 48px rgba(0, 0, 0, 0.18);
}
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  flex-wrap: wrap;
}
.section-head h2 {
  margin: 4px 0 3px;
  font-size: clamp(1.35rem, 3vw, 2.1rem);
  letter-spacing: -0.04em;
}
.section-head p {
  color: var(--muted);
  margin: 4px 0 0;
  max-width: 750px;
  font-size: 0.85rem;
}
.section-filter-box {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 650px;
  width: 100%;
}
.section-filter-top {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.func-search {
  background: #081625;
  border: 1px solid var(--line);
  color: var(--text);
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 0.74rem;
  outline: none;
  min-width: 170px;
}
.func-search:focus {
  border-color: var(--mint);
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: flex-end;
  max-height: 180px;
  overflow-y: auto;
  padding: 2px 0;
  scrollbar-width: thin;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 8px;
  color: #cfe2f7;
  font-size: 0.72rem;
  cursor: pointer;
  background: #0a192b;
  user-select: none;
  transition: all 0.12s ease;
}
.chip input {
  accent-color: var(--mint);
  margin: 0;
}
.chip:hover {
  border-color: rgba(109, 168, 255, 0.4);
}
.chip:has(input:checked) {
  border-color: var(--mint);
  background: rgba(85, 230, 189, 0.12);
  color: var(--mint);
}
.chip-only {
  background: transparent;
  border: 1px solid rgba(85, 230, 189, 0.25);
  color: var(--mint);
  font-size: 0.6rem;
  border-radius: 4px;
  padding: 0 4px;
  cursor: pointer;
  margin-left: 2px;
  opacity: 0.7;
  line-height: 1.2;
}
.chip-only:hover {
  opacity: 1;
  background: rgba(85, 230, 189, 0.2);
}
.section-actions {
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 14px 0 12px;
}
.section-actions button {
  border: 1px solid var(--line);
  background: #142c45;
  color: var(--text);
  padding: 5px 10px;
  border-radius: 7px;
  cursor: pointer;
  font-size: 0.72rem;
  font-weight: 600;
}
.section-actions button:hover {
  border-color: var(--mint);
  color: var(--mint);
}
.func-count-badge {
  font-size: 0.72rem;
  color: var(--muted);
  margin-left: auto;
}

/* Visual Grid & Chart Cards */
.visual-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
  gap: 14px;
}
.card {
  background: rgba(18, 38, 61, 0.62);
  border: 1px solid rgba(38, 68, 95, 0.8);
  border-radius: 16px;
  padding: 14px;
  min-width: 0;
}
.card h3 {
  font-size: 0.9rem;
  margin: 0 0 2px;
}
.card p {
  font-size: 0.76rem;
  color: var(--muted);
  margin: 0 0 12px;
}
.chart {
  width: 100%;
  overflow: auto;
}
.legend {
  display: flex;
  gap: 12px;
  margin-top: 12px;
  font-size: 0.74rem;
  color: var(--muted);
}
.legend span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

/* FIXED BAR ROW - Flex layout guarantees number stays strictly to the right of the bar */
.bar-row {
  display: grid;
  grid-template-columns: minmax(180px, 32%) 1fr;
  gap: 12px;
  align-items: center;
  margin: 8px 0;
  padding: 4px 0;
  border-bottom: 1px solid rgba(38, 68, 95, 0.25);
}
.bar-row:last-child {
  border-bottom: none;
}
.bar-label-block {
  min-width: 0;
}
.bar-label {
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #cfe2f7;
}
.bar-meta-sub {
  font-size: 0.67rem;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-top: 1px;
}
.bar-track {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.bar-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}
.bar-fill-track {
  flex: 1 1 0%;
  height: 9px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 5px;
  overflow: hidden;
  min-width: 40px;
}
.bar-line {
  height: 100%;
  border-radius: 5px;
  display: block;
}
.bar-name {
  font-size: 0.7rem;
  color: var(--muted);
  white-space: nowrap;
  flex-shrink: 0;
  min-width: 150px;
  font-variant-numeric: tabular-nums;
}

/* Memory Explanation Callout */
.memory-callout {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(7, 17, 31, 0.5);
  border: 1px solid rgba(38, 68, 95, 0.6);
  font-size: 0.72rem;
  color: var(--muted);
  line-height: 1.45;
}
.memory-callout strong {
  color: var(--mint);
}

/* Data Table with Case Metadata Columns */
.table-wrap {
  margin-top: 14px;
  overflow: auto;
  border: 1px solid rgba(38, 68, 95, 0.8);
  border-radius: 14px;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  min-width: 880px;
  table-layout: fixed;
  font-size: 0.71rem;
}
.data-table th {
  position: sticky;
  top: 0;
  background: #12263d;
  color: #cfe2f7;
  text-align: left;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.6rem;
  padding: 8px 7px;
  z-index: 2;
}
.data-table td {
  border-top: 1px solid rgba(38, 68, 95, 0.55);
  padding: 6px 7px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 150px;
}
.data-table tr:hover {
  background: rgba(109, 168, 255, 0.06);
}

/* Table controls: multi-criteria sort + pagination */
.table-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.sort-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}
.sort-field {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.72rem;
  color: var(--muted);
}
.sort-field select {
  border: 1px solid var(--line);
  background: #142c45;
  color: var(--text);
  padding: 4px 7px;
  border-radius: 7px;
  font-size: 0.72rem;
}
.sort-dir-btn {
  border: 1px solid var(--line);
  background: #142c45;
  color: var(--text);
  padding: 4px 9px;
  border-radius: 7px;
  cursor: pointer;
  font-size: 0.72rem;
  font-weight: 600;
}
.sort-dir-btn:hover {
  border-color: var(--mint);
  color: var(--mint);
}
.pager {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.72rem;
  color: var(--muted);
}
.page-btn {
  border: 1px solid var(--line);
  background: #142c45;
  color: var(--text);
  padding: 5px 10px;
  border-radius: 7px;
  cursor: pointer;
  font-size: 0.72rem;
  font-weight: 600;
}
.page-btn:hover:not(:disabled) {
  border-color: var(--mint);
  color: var(--mint);
}
.page-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.good { color: var(--mint); font-weight: 750; }
.warn { color: var(--amber); font-weight: 750; }
.bad { color: var(--rose); font-weight: 750; }
.muted { color: var(--muted); }
.empty { padding: 30px; color: var(--muted); text-align: center; font-size: 0.85rem; }

.footer {
  margin-top: 36px;
  color: var(--muted);
  font-size: 0.76rem;
  border-top: 1px solid rgba(38, 68, 95, 0.6);
  padding-top: 16px;
}

/* Responsive */
@media (max-width: 1200px) {
  .top3-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .scaling-grid { grid-template-columns: 1fr; }
}
@media (max-width: 1024px) {
  .app-layout { flex-direction: column; padding: 14px 14px 60px; }
  .sidebar {
    width: 100%;
    position: static;
    max-height: none;
    margin-bottom: 16px;
  }
  .sidebar-nav {
    flex-direction: row;
    flex-wrap: wrap;
    gap: 6px;
  }
  .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .visual-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .summary { grid-template-columns: 1fr; }
  .top3-grid { grid-template-columns: 1fr; }
  .bar-row { grid-template-columns: 1fr; gap: 4px; }
}
"""
    script = r"""
const REPORT=__REPORT__;
const COLORS={mojagg:'#55e6bd',numbagg:'#6da8ff'};
const $=(selector,root=document)=>root.querySelector(selector);
const $$=(selector,root=document)=>Array.from(root.querySelectorAll(selector));
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtNs=value=>{if(value==null)return'N/A';if(value<1000)return`${value.toFixed(0)} ns`;if(value<1e6)return`${(value/1e3).toFixed(1)} µs`;if(value<1e9)return`${(value/1e6).toFixed(2)} ms`;return`${(value/1e9).toFixed(2)} s`};
const fmtBytes=value=>{if(value==null)return'N/A';if(value<1024)return`${value.toFixed(0)} B`;if(value<1048576)return`${(value/1024).toFixed(1)} KiB`;if(value<1073741824)return`${(value/1048576).toFixed(2)} MiB`;return`${(value/1073741824).toFixed(2)} GiB`};
const fmtRatio=value=>value==null?'N/A':`${value.toFixed(2)}×`;
const ratioClass=value=>value==null?'muted':value<.98?'good':value>1.02?'warn':'muted';
const fmtCount=n=>{if(n==null)return'';if(n<1000)return String(n);if(n<1e6)return(n/1e3).toFixed(1)+'K';return(n/1e6).toFixed(1)+'M'};
const COMPARABLE_STATUSES=['ok','diverge'];
const CHART_CASE_LIMIT=15;
const PAGE_SIZE=25;
const SORT_FIELDS=[{key:'function',label:'Function'},{key:'case',label:'Case'},{key:'dtype',label:'Dtype'},{key:'elements',label:'Elements'},{key:'implementation',label:'Implementation'},{key:'status',label:'Status'},{key:'time_ns',label:'Median time'},{key:'memory_bytes',label:'Memory'},{key:'time_ratio',label:'Time ratio'},{key:'memory_ratio',label:'Memory ratio'}];
const tableState={};
function getTableState(section){
  if(!tableState[section])tableState[section]={page:1,sort:[{key:'function',dir:'asc'},{key:'case',dir:'asc'}]};
  return tableState[section];
}

const sectionNames=['reduction','groupby','matrix','rolling','exponential','non-reduction'];
const sectionTitles={reduction:'Reductions',groupby:'Group by',matrix:'Matrix functions',rolling:'Rolling windows',exponential:'Exponential moving','non-reduction':'Non-reduction'};
const sectionDescriptions={reduction:'NaN-aware scalar reductions across contiguous, batched, and multi-axis inputs.',groupby:'Dense grouped reductions across label dtypes and cardinalities.',matrix:'Static covariance and correlation matrices with batched cores.',rolling:'Trailing window operations with controlled NaN density and window size.',exponential:'Exponentially weighted moving operations and alpha sensitivity.','non-reduction':'Forward and backward fill workloads, including multi-axis cores.'};
const records=REPORT.records||[];
const implementations=['mojagg','numbagg'];

function formatCaseMeta(m){
  if(!m)return'';
  const parts=[];
  if(m.dtype)parts.push(m.dtype);
  if(m.shape){
    const s=Array.isArray(m.shape)?m.shape.join('×'):m.shape;
    const c=Array.isArray(m.shape)?m.shape.reduce((a,b)=>a*b,1):null;
    parts.push(c?`${s} (${fmtCount(c)} el)`:s);
  }
  if(m.nan_fraction!=null){
    const pct=Math.round(m.nan_fraction*100)+'% NaN';
    const pat=m.nan_pattern?` (${m.nan_pattern})`:'';
    parts.push(pct+pat);
  }
  if(m.axis!=null)parts.push('axis '+m.axis);
  if(m.num_groups!=null)parts.push(m.num_groups+' groups');
  if(m.window!=null)parts.push('win '+m.window);
  if(m.alpha!=null)parts.push('α='+m.alpha);
  if(m.ddof!=null&&m.ddof!==0)parts.push('ddof='+m.ddof);
  return parts.join(' · ');
}

function fmtMemCell(bytes){
  if(bytes==null)return'<span class="muted">N/A</span>';
  if(bytes===0||bytes<4096){
    return`${fmtBytes(bytes)} <span class="muted" style="font-size:0.67rem" title="Zero-copy native kernel execution">(zero-copy)</span>`;
  }
  return fmtBytes(bytes);
}

function selectedImplementations(){return $$('.impl-check').filter(x=>x.checked).map(x=>x.value)}
function selectedFunctions(section){return $$('.function-check[data-section="'+section+'"]').filter(x=>x.checked).map(x=>x.value)}

function filtered(section){
  const query=($('#search').value||'').toLowerCase();
  const impls=selectedImplementations();
  const funcs=selectedFunctions(section);
  return records.filter(r=>r.section===section&&impls.includes(r.implementation)&&funcs.includes(r.function)&&(!query||JSON.stringify(r).toLowerCase().includes(query)));
}

function groups(rows){
  const map=new Map();
  for(const row of rows){
    const key=row.function+'|||'+row.case;
    if(!map.has(key))map.set(key,[]);
    map.get(key).push(row);
  }
  return[...map.values()];
}

function caseElements(meta){
  if(!meta||!meta.shape)return 0;
  const shape=Array.isArray(meta.shape)?meta.shape:[meta.shape];
  return shape.reduce((a,b)=>a*b,1);
}

function topGroups(rows,limit){
  const all=groups(rows);
  all.sort((a,b)=>caseElements(b[0].case_metadata)-caseElements(a[0].case_metadata));
  return limit?all.slice(0,limit):all;
}

function sortValue(r,key){
  const m=r.case_metadata||{};
  switch(key){
    case'function':return r.function||'';
    case'case':return r.case||'';
    case'dtype':return m.dtype||'';
    case'elements':return caseElements(m);
    case'implementation':return r.implementation||'';
    case'status':return r.status||'';
    case'time_ns':return r.time_ns;
    case'memory_bytes':return r.memory_bytes;
    case'time_ratio':return r.time_ratio;
    case'memory_ratio':return r.memory_ratio;
    default:return null;
  }
}

function compareRows(a,b,sortSpec){
  for(const{key,dir}of sortSpec){
    const av=sortValue(a,key),bv=sortValue(b,key);
    if(av==null&&bv==null)continue;
    if(av==null)return 1;
    if(bv==null)return-1;
    const cmp=typeof av==='string'?av.localeCompare(bv):av-bv;
    if(cmp!==0)return dir==='asc'?cmp:-cmp;
  }
  return 0;
}

function sortControlsMarkup(section){
  const state=getTableState(section);
  let out='<div class="sort-controls">';
  state.sort.forEach((s,idx)=>{
    const options=SORT_FIELDS.map(f=>'<option value="'+f.key+'"'+(f.key===s.key?' selected':'')+'>'+f.label+'</option>').join('');
    out+='<label class="sort-field">'+(idx===0?'Sort by':'then by')+' <select class="sort-key" data-section="'+esc(section)+'" data-idx="'+idx+'">'+options+'</select><button type="button" class="sort-dir-btn" data-section="'+esc(section)+'" data-idx="'+idx+'" title="Toggle direction">'+(s.dir==='asc'?'\u2191 asc':'\u2193 desc')+'</button></label>';
  });
  out+='</div>';
  return out;
}

function pagerMarkup(section,page,totalPages,total){
  return'<div class="pager"><button type="button" class="page-btn" data-section="'+esc(section)+'" data-dir="prev"'+(page<=1?' disabled':'')+'>\u2039 Prev</button><span class="page-info">Page '+page+' / '+totalPages+' ('+total+' rows)</span><button type="button" class="page-btn" data-section="'+esc(section)+'" data-dir="next"'+(page>=totalPages?' disabled':'')+'>Next \u203a</button></div>';
}

function ratioColor(value){
  if(value==null)return'rgba(144,169,194,.08)';
  const strength=Math.min(1,Math.abs(Math.log(value))/1.4);
  return value<1?'rgba(85,230,189,'+(.12+.42*strength)+')':'rgba(255,127,159,'+(.10+.34*strength)+')';
}

function statMarkup(){
  const comparable=records.filter(r=>r.implementation==='numbagg'&&COMPARABLE_STATUSES.includes(r.status)&&r.time_ratio!=null);
  const wins=comparable.filter(r=>r.time_ratio>1).length;
  const median=comparable.length?comparable.map(r=>r.time_ratio).sort((a,b)=>a-b)[Math.floor(comparable.length/2)]:null;
  const ok=records.filter(r=>r.implementation==='mojagg'&&r.status==='ok').length;
  const cases=new Set(records.map(r=>r.section+'|'+r.function+'|'+r.case)).size;
  return '<div class="stat"><div class="label">Mojagg wins vs numbagg</div><div class="value">'+wins+'<span class="muted" style="font-size:1rem"> / '+comparable.length+'</span></div></div><div class="stat"><div class="label">Median mojagg speedup</div><div class="value">'+(median==null?'N/A':median.toFixed(2)+'×')+'</div></div><div class="stat"><div class="label">Measured scenarios</div><div class="value">'+cases+'</div></div><div class="stat"><div class="label">Mojagg measurements</div><div class="value">'+ok+'</div></div>';
}

function top3Markup(){
  let cards='';
  for(const sec of sectionNames){
    const secRecords=records.filter(r=>r.section===sec);
    if(!secRecords.length)continue;
    const caseMap=new Map();
    for(const r of secRecords){
      const key=r.function+'|||'+r.case;
      if(!caseMap.has(key))caseMap.set(key,{});
      caseMap.get(key)[r.implementation]=r;
    }
    const candidates=[];
    for(const [key,impls] of caseMap.entries()){
      const moj=impls.mojagg;
      if(!moj||moj.status!=='ok'||!moj.time_ns)continue;
      const num=impls.numbagg;
      if(num&&num.status==='ok'&&num.time_ns){
        const speedup=num.time_ns/moj.time_ns;
        candidates.push({speedup,func:moj.function,case:moj.case,meta:moj.case_metadata,mojTime:moj.time_ns,refTime:num.time_ns,refName:'numbagg'});
      }
    }
    candidates.sort((a,b)=>b.speedup-a.speedup);
    const top3=candidates.slice(0,3);
    if(!top3.length)continue;
    const medals=['🥇','🥈','🥉'];
    let items='';
    top3.forEach((item,idx)=>{
      const metaStr=formatCaseMeta(item.meta);
      items+='<div class="top3-item"><span class="top3-rank">'+medals[idx]+'</span><div class="top3-details"><div class="top3-func">'+esc(item.func)+' <span class="muted" style="font-weight:400;font-size:0.7rem">('+esc(item.case)+')</span></div><div class="top3-meta" title="'+esc(metaStr)+'">'+esc(metaStr)+'</div></div><div class="top3-stat"><div class="top3-speedup">'+item.speedup.toFixed(2)+'× faster</div><div class="top3-times">'+fmtNs(item.mojTime)+' vs '+fmtNs(item.refTime)+'</div></div></div>';
    });
    cards+='<div class="top3-card"><div class="top3-card-title"><h3>'+esc(sectionTitles[sec]||sec)+'</h3><span>Top 3 Speedups</span></div><div>'+items+'</div></div>';
  }
  return '<section class="showcase-panel" id="top3-showcase"><div class="showcase-head"><div><div class="eyebrow">Performance Highlights</div><h2>Top 3 Champions by Category</h2><p>The highest measured speedups of mojagg against reference implementations across each functional domain.</p></div></div><div class="top3-grid">'+cards+'</div></section>';
}

function worst3Markup(){
  let cards='';
  for(const sec of sectionNames){
    const secRecords=records.filter(r=>r.section===sec);
    if(!secRecords.length)continue;
    const caseMap=new Map();
    for(const r of secRecords){
      const key=r.function+'|||'+r.case;
      if(!caseMap.has(key))caseMap.set(key,{});
      caseMap.get(key)[r.implementation]=r;
    }
    const candidates=[];
    for(const [key,impls] of caseMap.entries()){
      const moj=impls.mojagg;
      if(!moj||moj.status!=='ok'||!moj.time_ns)continue;
      const num=impls.numbagg;
      if(num&&num.status==='ok'&&num.time_ns){
        const speedup=num.time_ns/moj.time_ns;
        candidates.push({speedup,func:moj.function,case:moj.case,meta:moj.case_metadata,mojTime:moj.time_ns,refTime:num.time_ns,refName:'numbagg'});
      }
    }
    candidates.sort((a,b)=>a.speedup-b.speedup);
    const worst3=candidates.slice(0,3);
    if(!worst3.length)continue;
    const ranks=['#1','#2','#3'];
    let items='';
    worst3.forEach((item,idx)=>{
      const metaStr=formatCaseMeta(item.meta);
      const isSlow=item.speedup<1;
      const speedupStr=isSlow?(1/item.speedup).toFixed(2)+'× slower':item.speedup.toFixed(2)+'× vs '+item.refName;
      const speedupColor=isSlow?'var(--rose)':'var(--amber)';
      items+='<div class="top3-item"><span class="top3-rank" style="font-size:0.85rem;font-weight:800;color:var(--rose);padding-top:2px">'+ranks[idx]+'</span><div class="top3-details"><div class="top3-func">'+esc(item.func)+' <span class="muted" style="font-weight:400;font-size:0.7rem">('+esc(item.case)+')</span></div><div class="top3-meta" title="'+esc(metaStr)+'">'+esc(metaStr)+'</div></div><div class="top3-stat"><div class="top3-speedup" style="color:'+speedupColor+'">'+speedupStr+'</div><div class="top3-times">'+fmtNs(item.mojTime)+' vs '+fmtNs(item.refTime)+'</div></div></div>';
    });
    cards+='<div class="top3-card"><div class="top3-card-title"><h3>'+esc(sectionTitles[sec]||sec)+'</h3><span style="color:var(--rose)">Bottom 3 Ratios</span></div><div>'+items+'</div></div>';
  }
  return '<section class="showcase-panel" id="worst3-showcase"><div class="showcase-head"><div><div class="eyebrow" style="color:var(--rose)">Performance Opportunities</div><h2>Worst 3 by Category</h2><p>The lowest relative speedups (or regressions) of mojagg against reference implementations across each functional domain.</p></div></div><div class="top3-grid">'+cards+'</div></section>';
}

function scalingMarkup(){
  const sizeMap=new Map();
  for(const r of records){
    if(!COMPARABLE_STATUSES.includes(r.status)||r.time_ns==null)continue;
    const shape=r.case_metadata?.shape;
    if(!shape||!Array.isArray(shape))continue;
    const sz=shape.reduce((a,b)=>a*b,1);
    if(!sizeMap.has(sz))sizeMap.set(sz,{mojagg:[],numbagg:[]});
    if(sizeMap.get(sz)[r.implementation]){
      sizeMap.get(sz)[r.implementation].push(r.time_ns);
    }
  }
  const sizes=[...sizeMap.keys()].sort((a,b)=>a-b);
  let sizeChartSvg='';
  if(sizes.length>0){
    const W=580,H=250,L=65,R=25,T=20,B=45;
    const dataPoints=sizes.map(sz=>{
      const b=sizeMap.get(sz);
      const mMed=b.mojagg.length?b.mojagg.sort((x,y)=>x-y)[Math.floor(b.mojagg.length/2)]:null;
      const nMed=b.numbagg.length?b.numbagg.sort((x,y)=>x-y)[Math.floor(b.numbagg.length/2)]:null;
      return{sz,mMed,nMed};
    });
    const allTimes=dataPoints.flatMap(d=>[d.mMed,d.nMed].filter(x=>x!=null));
    const maxTime=Math.max(1,...allTimes);
    const xPos=i=>L+(i/Math.max(1,sizes.length-1))*(W-L-R);
    const yPos=t=>(H-B)-(t/maxTime)*(H-T-B);

    let gridLines='';
    for(let s=0;s<=4;s++){
      const val=(maxTime*s)/4;
      const y=(H-B)-(s/4)*(H-T-B);
      gridLines+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="rgba(38,68,95,0.4)" stroke-dasharray="2 3"/><text x="'+(L-8)+'" y="'+(y+4)+'" fill="#90a9c2" font-size="9" text-anchor="end">'+fmtNs(val)+'</text>';
    }
    const drawLine=(impl,color)=>{
      const pts=[];
      dataPoints.forEach((d,i)=>{
        const t=impl==='mojagg'?d.mMed:d.nMed;
        if(t!=null)pts.push({x:xPos(i),y:yPos(t),t,sz:d.sz});
      });
      if(!pts.length)return'';
      let path='M '+pts[0].x+' '+pts[0].y;
      for(let i=1;i<pts.length;i++)path+=' L '+pts[i].x+' '+pts[i].y;
      let dots='';
      for(const p of pts){
        dots+='<circle cx="'+p.x+'" cy="'+p.y+'" r="4" fill="'+color+'"><title>'+esc(impl)+': '+fmtNs(p.t)+' at '+fmtCount(p.sz)+' elements</title></circle>';
      }
      return'<path d="'+path+'" fill="none" stroke="'+color+'" stroke-width="2.5"/>'+dots;
    };
    let xLabels='';
    sizes.forEach((sz,i)=>{
      xLabels+='<text x="'+xPos(i)+'" y="'+(H-15)+'" fill="#90a9c2" font-size="10" text-anchor="middle">'+fmtCount(sz)+'</text>';
    });
    sizeChartSvg='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto" role="img" aria-label="Scaling by data size">'+gridLines+drawLine('numbagg',COLORS.numbagg)+drawLine('mojagg',COLORS.mojagg)+xLabels+'<text x="'+((W+L)/2)+'" y="'+(H-2)+'" fill="#90a9c2" font-size="10" text-anchor="middle">Input elements (array size) →</text></svg>';
  }else{
    sizeChartSvg='<div class="empty">No size data available.</div>';
  }

  const catStats=[];
  for(const sec of sectionNames){
    const secRecs=records.filter(r=>r.section===sec&&r.implementation==='numbagg'&&COMPARABLE_STATUSES.includes(r.status)&&r.time_ratio!=null);
    if(!secRecs.length)continue;
    const ratios=secRecs.map(r=>r.time_ratio).sort((a,b)=>a-b);
    const medRatio=ratios[Math.floor(ratios.length/2)];
    const wins=secRecs.filter(r=>r.time_ratio>1).length;
    catStats.push({sec,title:sectionTitles[sec]||sec,medRatio,wins,total:secRecs.length});
  }
  let catChartSvg='';
  if(catStats.length>0){
    const W=520,rowH=34,topPad=15,leftW=140,barW=250;
    const H=topPad+catStats.length*rowH+30;
    const maxRatio=Math.max(2.5,...catStats.map(c=>c.medRatio));
    let rowsOut='';
    catStats.forEach((c,i)=>{
      const y=topPad+i*rowH;
      const w=Math.max(4,Math.min(barW,(c.medRatio/maxRatio)*barW));
      const winPct=Math.round((c.wins/c.total)*100);
      rowsOut+='<text x="'+(leftW-10)+'" y="'+(y+16)+'" fill="#cfe2f7" font-size="11" text-anchor="end" font-weight="600">'+esc(c.title)+'</text><rect x="'+leftW+'" y="'+(y+4)+'" width="'+barW+'" height="18" rx="4" fill="rgba(255,255,255,0.04)"/><rect x="'+leftW+'" y="'+(y+4)+'" width="'+w+'" height="18" rx="4" fill="url(#mintGrad)"/><text x="'+(leftW+w+8)+'" y="'+(y+17)+'" fill="'+(c.medRatio>=1?'#55e6bd':'#ff7f9f')+'" font-size="11" font-weight="700">'+c.medRatio.toFixed(2)+'× speedup</text><text x="'+(leftW+barW+115)+'" y="'+(y+16)+'" fill="#90a9c2" font-size="9" text-anchor="end">'+c.wins+'/'+c.total+' wins ('+winPct+'%)</text>';
    });
    const refX=leftW+(1.0/maxRatio)*barW;
    catChartSvg='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto" role="img" aria-label="Median speedup across operations"><defs><linearGradient id="mintGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#3bba98"/><stop offset="100%" stop-color="#55e6bd"/></linearGradient></defs><line x1="'+refX+'" x2="'+refX+'" y1="'+topPad+'" y2="'+(H-25)+'" stroke="#ffc857" stroke-dasharray="3 3"/><text x="'+refX+'" y="'+(H-10)+'" fill="#ffc857" font-size="9" text-anchor="middle">1.0× baseline</text>'+rowsOut+'</svg>';
  }else{
    catChartSvg='<div class="empty">No comparable operations data available.</div>';
  }

  return '<section class="scaling-panel" id="scaling-trends"><div class="showcase-head"><div><div class="eyebrow">Scaling & Distribution</div><h2>Scaling Profile & Operational Advantage</h2><p>Empirical evidence of how Mojagg scales with array size and maintains consistent speedups across functional categories.</p></div></div><div class="scaling-grid"><div class="scaling-chart-card"><h3>Latency vs Dataset Size</h3><p>Median execution time across element counts. Mojagg scales flatter and widens its advantage as datasets grow.</p><div>'+sizeChartSvg+'</div><div class="legend" style="justify-content:center;margin-top:8px"><span><i class="dot" style="background:'+COLORS.mojagg+'"></i>mojagg</span><span><i class="dot" style="background:'+COLORS.numbagg+'"></i>numbagg</span></div></div><div class="scaling-chart-card"><h3>Speedup Across All Operation Categories</h3><p>Median speedup factor of mojagg vs numbagg (values &gt; 1.0× indicate mojagg is faster).</p><div>'+catChartSvg+'</div><div style="font-size:0.7rem;color:var(--muted);text-align:center;margin-top:8px">Dashed line = 1.0× parity with numbagg. Longer mint bars indicate higher mojagg speedup.</div></div></div></section>';
}

function renderSidebar(){
  const sidebar=$('#sidebar');
  if(!sidebar)return;
  let catLinks='';
  for(const sec of sectionNames){
    const secRecs=records.filter(r=>r.section===sec);
    if(!secRecs.length)continue;
    const funcs=new Set(secRecs.map(r=>r.function)).size;
    catLinks+='<a href="#section-'+esc(sec)+'" class="sidebar-link" data-nav="'+esc(sec)+'"><span>'+esc(sectionTitles[sec]||sec)+'</span><span class="sidebar-badge">'+funcs+'</span></a>';
  }
  sidebar.innerHTML='<div class="sidebar-title">Navigation</div><nav class="sidebar-nav"><a href="#summary" class="sidebar-link active" data-nav="summary">📊 Executive Stats</a><a href="#top3-showcase" class="sidebar-link" data-nav="top3-showcase">🏆 Top 3 Champions</a><a href="#worst3-showcase" class="sidebar-link" data-nav="worst3-showcase">⚠️ Worst 3 by Category</a><a href="#scaling-trends" class="sidebar-link" data-nav="scaling-trends">📈 Scaling & Trends</a><a href="#device-profile" class="sidebar-link" data-nav="device-profile">💻 Device Profile</a><a href="#threading-dispatch" class="sidebar-link" data-nav="threading-dispatch">⚙️ Threading & Dispatch</a></nav><div class="sidebar-divider"></div><div class="sidebar-title">Benchmarks</div><nav class="sidebar-nav">'+catLinks+'</nav>';

  const links=$$('.sidebar-link');
  const targets=links.map(l=>{
    const id=l.getAttribute('href')?.replace('#','');
    return{link:l,el:id?document.getElementById(id):null};
  }).filter(t=>t.el);

  window.addEventListener('scroll',()=>{
    const scrollPos=window.scrollY+140;
    let current=null;
    for(const t of targets){
      if(t.el.offsetTop<=scrollPos)current=t.link;
    }
    if(current){
      links.forEach(l=>l.classList.remove('active'));
      current.classList.add('active');
    }
  },{passive:true});
}

function barChart(rows){
  const scenarios=topGroups(rows,CHART_CASE_LIMIT);
  if(!scenarios.length)return'<div class="empty">No rows match the current filters.</div>';
  const valid=scenarios.flatMap(g=>g.filter(r=>r.time_ratio!=null).map(r=>r.time_ratio));
  const max=Math.max(1.25,...valid,1);
  let out='<div class="axis-line"></div>';
  for(const group of scenarios){
    const first=group[0];
    const label=first.function+' · '+first.case;
    const metaStr=formatCaseMeta(first.case_metadata);
    out+='<div class="bar-row"><div class="bar-label-block"><div class="bar-label" title="'+esc(label)+'">'+esc(label)+'</div><div class="bar-meta-sub" title="'+esc(metaStr)+'">'+esc(metaStr)+'</div></div><div class="bar-track">';
    for(const row of group){
      if(row.time_ratio==null)continue;
      const width=Math.max(1,Math.min(100,(row.time_ratio/max)*100));
      out+='<div class="bar-item"><div class="bar-fill-track"><span class="bar-line" title="'+esc(row.implementation)+' · '+fmtRatio(row.time_ratio)+' ('+fmtNs(row.time_ns)+')" style="width:'+width+'%;background:'+COLORS[row.implementation]+'"></span></div><span class="bar-name"><strong style="color:'+COLORS[row.implementation]+'">'+esc(row.implementation)+'</strong> '+fmtRatio(row.time_ratio)+' <span class="muted">('+fmtNs(row.time_ns)+')</span></span></div>';
    }
    out+='</div></div>';
  }
  return out;
}

function scatterChart(rows){
  const topKeys=new Set(topGroups(rows,CHART_CASE_LIMIT).map(g=>g[0].function+'|||'+g[0].case));
  const points=rows.filter(r=>r.implementation==='numbagg'&&COMPARABLE_STATUSES.includes(r.status)&&r.time_ratio>0&&r.memory_ratio>0&&topKeys.has(r.function+'|||'+r.case));
  if(!points.length)return'<div class="empty">No comparable time/memory points for this filter.</div>';
  const width=700,height=390,left=66,right=22,top=30,bottom=58;
  const values=points.flatMap(r=>[r.time_ratio,r.memory_ratio]);
  const minValue=Math.max(.1,Math.min(.5,...values)*.8);
  const maxValue=Math.max(2,Math.max(...values)*1.2);
  const logMin=Math.log10(minValue),logMax=Math.log10(maxValue),logSpan=logMax-logMin;
  const x=v=>left+((Math.log10(v)-logMin)/logSpan)*(width-left-right);
  const y=v=>(height-bottom)-((Math.log10(v)-logMin)/logSpan)*(height-top-bottom);
  const ticks=[];
  for(let exponent=Math.floor(logMin);exponent<=Math.ceil(logMax);exponent++){
    for(const multiplier of [1,2,5]){
      const value=multiplier*Math.pow(10,exponent);
      if(value>=minValue*.999&&value<=maxValue*1.001)ticks.push(value);
    }
  }
  const tickLabel=value=>value>=10?value.toFixed(0)+'×':value>=1?value.toFixed(1)+'×':value.toFixed(2)+'×';
  let grid='';
  for(const tick of ticks){
    const tickX=x(tick),tickY=y(tick),major=Math.abs(Math.log10(tick)-Math.round(Math.log10(tick)))<.001;
    grid+='<line x1="'+tickX+'" x2="'+tickX+'" y1="'+top+'" y2="'+(height-bottom)+'" stroke="rgba(38,68,95,'+(major?.55:.32)+')" stroke-dasharray="'+(major?'4 4':'2 5')+'"/><line x1="'+left+'" x2="'+(width-right)+'" y1="'+tickY+'" y2="'+tickY+'" stroke="rgba(38,68,95,'+(major?.55:.32)+')" stroke-dasharray="'+(major?'4 4':'2 5')+'"/><text x="'+tickX+'" y="'+(height-bottom+18)+'" fill="#90a9c2" font-size="9" text-anchor="middle">'+tickLabel(tick)+'</text><text x="'+(left-8)+'" y="'+(tickY+3)+'" fill="#90a9c2" font-size="9" text-anchor="end">'+tickLabel(tick)+'</text>';
  }
  const quadrantColor=(time,memory)=>time<1&&memory<1?'#55e6bd':time<1?'#6da8ff':memory<1?'#ffc857':'#ef8b87';
  let out='<svg viewBox="0 0 '+width+' '+height+'" style="width:100%;height:auto;min-width:520px" role="img" aria-label="Logarithmic time and memory ratio quadrant chart">'+grid+'<rect x="'+left+'" y="'+top+'" width="'+(x(1)-left)+'" height="'+(y(1)-top)+'" fill="rgba(109,168,255,.045)"/><rect x="'+x(1)+'" y="'+top+'" width="'+(width-right-x(1))+'" height="'+(y(1)-top)+'" fill="rgba(239,139,135,.045)"/><rect x="'+left+'" y="'+y(1)+'" width="'+(x(1)-left)+'" height="'+(height-bottom-y(1))+'" fill="rgba(85,230,189,.045)"/><rect x="'+x(1)+'" y="'+y(1)+'" width="'+(width-right-x(1))+'" height="'+(height-bottom-y(1))+'" fill="rgba(255,200,87,.045)"/><line x1="'+x(1)+'" x2="'+x(1)+'" y1="'+top+'" y2="'+(height-bottom)+'" stroke="#d5e6df" stroke-width="1.5" stroke-dasharray="6 5"/><line x1="'+left+'" x2="'+(width-right)+'" y1="'+y(1)+'" y2="'+y(1)+'" stroke="#d5e6df" stroke-width="1.5" stroke-dasharray="6 5"/><text x="'+(left+8)+'" y="'+(top+14)+'" fill="#6da8ff" font-size="10" font-weight="700">LOWER TIME</text><text x="'+(width-right-8)+'" y="'+(top+14)+'" fill="#ef8b87" font-size="10" text-anchor="end" font-weight="700">HIGHER BOTH</text><text x="'+(left+8)+'" y="'+(height-bottom-8)+'" fill="#55e6bd" font-size="10" font-weight="700">LOWER BOTH</text><text x="'+(width-right-8)+'" y="'+(height-bottom-8)+'" fill="#ffc857" font-size="10" text-anchor="end" font-weight="700">LOWER MEMORY</text><text x="'+(width/2)+'" y="'+(height-10)+'" fill="#90a9c2" text-anchor="middle" font-size="11">numbagg / mojagg time ratio · lower is better</text><text x="14" y="'+(height/2)+'" fill="#90a9c2" text-anchor="middle" font-size="11" transform="rotate(-90 14 '+(height/2)+')">numbagg / mojagg memory ratio · lower is better</text>';
  for(const point of points){
    const color=quadrantColor(point.time_ratio,point.memory_ratio);
    out+='<circle cx="'+x(point.time_ratio)+'" cy="'+y(point.memory_ratio)+'" r="5.5" fill="'+color+'" fill-opacity=".82" stroke="#d8fff2" stroke-opacity=".7"><title>'+esc(point.function)+' · '+esc(point.case)+' · time '+fmtRatio(point.time_ratio)+' · memory '+fmtRatio(point.memory_ratio)+'</title></circle>';
  }
  out+='</svg><div class="legend" style="justify-content:center;margin-top:4px"><span><i class="dot" style="background:#55e6bd"></i>lower time &amp; memory</span><span><i class="dot" style="background:#6da8ff"></i>lower time, higher memory</span><span><i class="dot" style="background:#ffc857"></i>higher time, lower memory</span><span><i class="dot" style="background:#ef8b87"></i>higher both</span></div><div class="memory-callout"><strong>How to read it:</strong> Each point is numbagg relative to mojagg. Both axes use logarithmic scales, the dashed crosshair is 1.00× parity, and hover a point for its operation and exact ratios. Lower memory is toward the bottom.</div>';
  return out;
}

function heatmap(rows){
  const scenarios=topGroups(rows,CHART_CASE_LIMIT);
  if(!scenarios.length)return'';
  const impls=selectedImplementations();
  let head='<tr><th>Function · case</th><th>Metadata</th>';
  for(const impl of impls)head+='<th>'+esc(impl)+' time</th><th>'+esc(impl)+' memory</th>';
  head+='</tr>';
  let body='';
  for(const group of scenarios){
    const first=group[0];
    const metaStr=formatCaseMeta(first.case_metadata);
    body+='<tr><td><strong>'+esc(first.function)+'</strong><br><span class="muted">'+esc(first.case)+'</span></td><td><span class="muted" style="font-size:0.7rem">'+esc(metaStr)+'</span></td>';
    for(const impl of impls){
      const row=group.find(r=>r.implementation===impl);
      body+=row?'<td style="background:'+ratioColor(row.time_ratio)+'" class="'+ratioClass(row.time_ratio)+'">'+fmtRatio(row.time_ratio)+'</td><td style="background:'+ratioColor(row.memory_ratio)+'" class="'+ratioClass(row.memory_ratio)+'">'+fmtRatio(row.memory_ratio)+'</td>':'<td class="muted">N/A</td><td class="muted">N/A</td>';
    }
    body+='</tr>';
  }
  return'<table class="data-table"><thead>'+head+'</thead><tbody>'+body+'</tbody></table>';
}

function statusBadge(status){
  if(status==='ok')return'<span class="good">OK</span>';
  if(status==='diverge')return'<span class="warn">DIVERGE</span>';
  if(status==='na')return'<span class="muted">N/A</span>';
  return'<span class="bad">ERROR</span>';
}

function detailTable(rows,section){
  if(!rows.length)return'<div class="empty">No rows match the current filters.</div>';
  const state=getTableState(section);
  const sorted=rows.slice().sort((a,b)=>compareRows(a,b,state.sort));
  const totalPages=Math.max(1,Math.ceil(sorted.length/PAGE_SIZE));
  if(state.page>totalPages)state.page=totalPages;
  if(state.page<1)state.page=1;
  const startIdx=(state.page-1)*PAGE_SIZE;
  const pageRows=sorted.slice(startIdx,startIdx+PAGE_SIZE);
  let body='';
  for(const r of pageRows){
    const status=statusBadge(r.status);
    const m=r.case_metadata||{};
    const shapeStr=m.shape?m.shape.join('×')+(m.shape.length?' <span class="muted">('+fmtCount(m.shape.reduce((a,b)=>a*b,1))+')</span>':''):'—';
    const nanStr=m.nan_fraction!=null?Math.round(m.nan_fraction*100)+'% <span class="muted">('+esc(m.nan_pattern||'random')+')</span>':'0%';
    const params=[];
    if(m.axis!=null)params.push('axis='+m.axis);
    if(m.num_groups!=null)params.push('groups='+m.num_groups);
    if(m.window!=null)params.push('win='+m.window);
    if(m.alpha!=null)params.push('α='+m.alpha);
    if(m.ddof!=null&&m.ddof!==0)params.push('ddof='+m.ddof);
    const paramStr=params.join(', ')||'—';

    body+='<tr><td title="'+esc(r.function)+'"><strong>'+esc(r.function)+'</strong></td><td title="'+esc(r.case)+'">'+esc(r.case)+'</td><td><span class="pill-badge">'+esc(m.dtype||'—')+'</span></td><td title="'+esc(shapeStr.replace(/<[^>]+>/g,''))+'">'+shapeStr+'</td><td>'+nanStr+'</td><td class="muted" title="'+esc(paramStr)+'">'+esc(paramStr)+'</td><td><strong style="color:'+COLORS[r.implementation]+'">'+esc(r.implementation)+'</strong></td><td>'+status+'</td><td>'+fmtNs(r.time_ns)+'</td><td>'+fmtMemCell(r.memory_bytes)+'</td><td class="'+ratioClass(r.time_ratio)+'">'+fmtRatio(r.time_ratio)+'</td><td class="'+ratioClass(r.memory_ratio)+'">'+fmtRatio(r.memory_ratio)+'</td><td title="'+esc(r.error||'')+'">'+(r.verification?esc(r.verification):'<span class="muted">—</span>')+'</td></tr>';
  }
  const controls='<div class="table-controls">'+sortControlsMarkup(section)+pagerMarkup(section,state.page,totalPages,sorted.length)+'</div>';
  return controls+'<table class="data-table"><thead><tr><th>Function</th><th>Case</th><th>Dtype</th><th>Shape / Elements</th><th>NaNs</th><th>Parameters</th><th>Implementation</th><th>Status</th><th>Median time</th><th>Peak memory</th><th>Time ratio</th><th>Memory ratio</th><th>Verification</th></tr></thead><tbody>'+body+'</tbody></table>'+controls;
}

function renderSections(){
  const root=$('#sections');
  root.innerHTML='';
  for(const section of sectionNames){
    const all=records.filter(r=>r.section===section);
    if(!all.length)continue;
    const functions=[...new Set(all.map(r=>r.function))];
    const chips=functions.map(fn=>'<label class="chip" data-fn="'+esc(fn)+'"><input class="function-check" data-section="'+esc(section)+'" value="'+esc(fn)+'" type="checkbox" checked> <span>'+esc(fn)+'</span><button type="button" class="chip-only" title="Show only '+esc(fn)+'" data-only-sec="'+esc(section)+'" data-only-fn="'+esc(fn)+'">only</button></label>').join('');

    root.insertAdjacentHTML('beforeend','<section class="section-panel" id="section-'+esc(section)+'" data-panel="'+esc(section)+'"><div class="section-head"><div><div class="eyebrow">'+esc(section)+'</div><h2>'+esc(sectionTitles[section])+'</h2><p>'+esc(sectionDescriptions[section])+'</p></div><div class="section-filter-box"><div class="section-filter-top"><input class="func-search" data-search-sec="'+esc(section)+'" type="search" placeholder="Filter algorithms (e.g. sum, min)..."><div class="func-count-badge" data-count="'+esc(section)+'">'+functions.length+' algorithms</div></div><div class="chips" data-chips-sec="'+esc(section)+'">'+chips+'</div></div></div><div class="section-actions"><button data-all="'+esc(section)+'">Select all</button><button data-none="'+esc(section)+'">Clear</button><button data-invert="'+esc(section)+'">Invert</button></div><div class="visual-grid"><div class="card"><h3>Time ratio</h3><p>Bars are implementation time divided by mojagg. Shorter is faster.</p><div class="chart" data-chart="'+esc(section)+'"></div><div class="legend">'+implementations.map(i=>'<span><i class="dot" style="background:'+COLORS[i]+'"></i>'+i+'</span>').join('')+'</div></div><div class="card"><h3>Time / memory quadrants</h3><p>Logarithmic ratios keep small memory differences readable; each point compares numbagg with mojagg.</p><div class="chart scatter" data-scatter="'+esc(section)+'"></div></div></div><div class="card" style="margin-top:14px"><h3>Ratio heatmap</h3><p>Green is below mojagg; pink is above it. N/A means that adapter is unavailable.</p><div class="heatmap" data-heatmap="'+esc(section)+'"></div></div><div class="table-wrap"><div data-table="'+esc(section)+'"></div></div></section>');
  }

  for(const button of $$('[data-all]')){
    button.onclick=()=>{
      $$('.function-check[data-section="'+button.dataset.all+'"]').forEach(x=>x.checked=true);
      updateFuncCount(button.dataset.all);
      renderAll();
    };
  }
  for(const button of $$('[data-none]')){
    button.onclick=()=>{
      $$('.function-check[data-section="'+button.dataset.none+'"]').forEach(x=>x.checked=false);
      updateFuncCount(button.dataset.none);
      renderAll();
    };
  }
  for(const button of $$('[data-invert]')){
    button.onclick=()=>{
      $$('.function-check[data-section="'+button.dataset.invert+'"]').forEach(x=>x.checked=!x.checked);
      updateFuncCount(button.dataset.invert);
      renderAll();
    };
  }
  for(const button of $$('[data-only-fn]')){
    button.onclick=(e)=>{
      e.preventDefault();
      e.stopPropagation();
      const sec=button.dataset.onlySec;
      const fn=button.dataset.onlyFn;
      $$('.function-check[data-section="'+sec+'"]').forEach(x=>x.checked=(x.value===fn));
      updateFuncCount(sec);
      renderAll();
    };
  }
  for(const input of $$('[data-search-sec]')){
    input.oninput=()=>{
      const sec=input.dataset.searchSec;
      const q=input.value.toLowerCase();
      $$('.chip[data-fn]', $('[data-chips-sec="'+sec+'"]')).forEach(chip=>{
        const fn=chip.dataset.fn.toLowerCase();
        chip.style.display=(!q||fn.includes(q))?'inline-flex':'none';
      });
    };
  }
  for(const check of $$('.function-check')){
    check.onchange=()=>{
      updateFuncCount(check.dataset.section);
      renderAll();
    };
  }
}

function updateFuncCount(sec){
  const total=$$('.function-check[data-section="'+sec+'"]').length;
  const sel=$$('.function-check[data-section="'+sec+'"]').filter(x=>x.checked).length;
  const el=$('[data-count="'+sec+'"]');
  if(el)el.textContent='Showing '+sel+' of '+total;
}

function renderAll(){
  for(const section of sectionNames){
    const rows=filtered(section);
    const chart=$('[data-chart="'+section+'"]');
    const scatter=$('[data-scatter="'+section+'"]');
    const heat=$('[data-heatmap="'+section+'"]');
    const table=$('[data-table="'+section+'"]');
    if(chart)chart.innerHTML=barChart(rows);
    if(scatter)scatter.innerHTML=scatterChart(rows);
    if(heat)heat.innerHTML=heatmap(rows);
    if(table)table.innerHTML=detailTable(rows,section);
  }
}

function wireTableControls(){
  document.addEventListener('change',e=>{
    if(!e.target.classList.contains('sort-key'))return;
    const section=e.target.dataset.section,idx=+e.target.dataset.idx;
    const state=getTableState(section);
    state.sort[idx].key=e.target.value;
    state.page=1;
    renderAll();
  });
  document.addEventListener('click',e=>{
    const dirBtn=e.target.closest('.sort-dir-btn');
    if(dirBtn){
      const section=dirBtn.dataset.section,idx=+dirBtn.dataset.idx;
      const state=getTableState(section);
      state.sort[idx].dir=state.sort[idx].dir==='asc'?'desc':'asc';
      renderAll();
      return;
    }
    const pageBtn=e.target.closest('.page-btn');
    if(pageBtn&&!pageBtn.disabled){
      const section=pageBtn.dataset.section;
      const state=getTableState(section);
      state.page+=pageBtn.dataset.dir==='next'?1:-1;
      renderAll();
    }
  });
}

function init(){
  const host=REPORT.host||{},device=REPORT.device||{};
  $('#hero-meta').innerHTML=['suite: '+esc(REPORT.suite.name),'device: '+esc(device.identity||'unknown'),'generated: '+esc(REPORT.generated_at),'platform: '+esc(host.platform||'unknown'),'python '+esc(host.python||''),'NumPy '+esc(host.numpy||'')].map(x=>'<span class="pill">'+x+'</span>').join('');
  $('#summary').innerHTML=statMarkup();
  $('#top3-container').innerHTML=top3Markup();
  $('#worst3-container').innerHTML=worst3Markup();
  $('#scaling-container').innerHTML=scalingMarkup();
  $('#device-profile').innerHTML=deviceMarkup();
  $('#threading-dispatch').innerHTML=threadingMarkup();
  $('#impls').innerHTML=implementations.map(i=>'<label class="check"><input class="impl-check" value="'+i+'" type="checkbox" checked> '+i+'</label>').join('');
  renderSections();
  renderSidebar();
  wireTableControls();
  for(const check of $$('.impl-check'))check.onchange=renderAll;
  $('#search').oninput=renderAll;
  renderAll();
}

function display(value){return value==null||value===''?'<span class="muted">N/A</span>':esc(value)}
function deviceField(label,value){return'<div class="device-field"><span>'+esc(label)+'</span><strong>'+display(value)+'</strong></div>'}
function deviceMarkup(){
  const device=REPORT.device||{},cpu=device.cpu||{},memory=device.memory||{},os=device.os||{},runtime=device.runtime||{},threads=runtime.thread_environment||{};
  const env=Object.entries(threads).map(([name,value])=>'<span class="env-pill">'+esc(name)+'='+esc(value)+'</span>').join('')||'<span class="muted">No thread variables exported</span>';
  const operatingSystem=[os.system,os.release].filter(Boolean).join(' ')||null;
  return'<section class="device-panel" id="device-profile"><div class="device-head"><div><div class="eyebrow">device profile</div><h2>'+display(device.identity||'Unknown device')+'</h2><p>Captured automatically at benchmark start using non-identifying runtime characteristics.</p></div></div><div class="device-grid"><div class="device-card"><h3>Compute and memory</h3>'+deviceField('CPU model',cpu.model)+deviceField('Architecture',cpu.architecture)+deviceField('Logical CPUs',cpu.logical_cores)+deviceField('Physical CPUs',cpu.physical_cores)+deviceField('Affinity CPUs',cpu.affinity_cores)+deviceField('Total memory',fmtBytes(memory.total_bytes))+'</div><div class="device-card"><h3>Runtime</h3>'+deviceField('Operating system',operatingSystem)+deviceField('OS / kernel build',os.version)+deviceField('Python',runtime.python)+deviceField('NumPy',runtime.numpy)+deviceField('Process ID',runtime.pid)+'</div></div><div class="device-card" style="margin-top:12px"><h3>Thread environment</h3><div class="device-env">'+env+'</div><div class="device-note">Thread variables are shown when exported by the runner or numerical libraries; CPU affinity reports the cores available to this process.</div></div></section>';
}

function threadingMarkup(){
  const cfg=REPORT.mojagg_config||{};
  const device=REPORT.device||{},cpu=device.cpu||{};
  const physicalCores=cpu.physical_cores||cpu.logical_cores||0;
  const threadsConfig=cfg.threads===0?'Auto ('+(physicalCores?physicalCores+' physical cores':'all cores')+')':(cfg.threads!=null?cfg.threads+' threads':'Auto');
  const thresholdVal=cfg.parallel_threshold!=null?cfg.parallel_threshold:200000;
  const thresholdStr=fmtCount(thresholdVal)+' elements ('+(thresholdVal>=1000?Math.round(thresholdVal/1000)+'K':thresholdVal)+')';
  const minGroupsVal=cfg.parallel_min_groups!=null?cfg.parallel_min_groups:16;
  const minGroupsStr=minGroupsVal+' outer groups';
  const backendStr=esc(cfg.backend||'auto')+' (CPU SIMD + multi-worker)';
  const gpuMinStr=cfg.gpu_min_bytes?fmtBytes(cfg.gpu_min_bytes):'64 MiB';
  const simdStr=cfg.simd_width===0?'Native CPU vector width':'Explicit ('+cfg.simd_width+' lanes)';

  return '<section class="device-panel" id="threading-dispatch"><div class="device-head"><div><div class="eyebrow">dispatch &amp; parallelism</div><h2>Threading &amp; Dispatch Configuration</h2><p>Explicit execution thresholds governing Mojagg multi-threading dispatch and SIMD kernel distribution.</p></div><span class="pill">MojaggConfig</span></div><div class="device-grid"><div class="device-card"><h3>Concurrency &amp; Workers</h3>'+deviceField('Configured threads',threadsConfig)+deviceField('Physical CPU cores',cpu.physical_cores||'N/A')+deviceField('Logical CPU cores',cpu.logical_cores||'N/A')+deviceField('Execution backend',backendStr)+'</div><div class="device-card"><h3>Parallel Thresholds</h3>'+deviceField('Parallel threshold',thresholdStr)+deviceField('Parallel min groups',minGroupsStr)+deviceField('GPU threshold',gpuMinStr)+'</div><div class="device-card"><h3>Dispatch Policy</h3>'+deviceField('Dispatch rule','Parallel when outer ≥ '+minGroupsVal+' &amp; inner ≥ '+fmtCount(thresholdVal))+deviceField('SIMD register mode',simdStr)+deviceField('Worker allocation','Zero-copy padded accumulators')+'</div></div><div class="device-card" style="margin-top:12px"><h3>Multi-threading Rationale</h3><div class="device-note">To avoid thread-spawn overhead on small workloads, Mojagg executes purely serial SIMD kernels unless the outer iteration count reaches at least <strong>'+minGroupsVal+' groups</strong> AND each inner reduction/operation slice has at least <strong>'+thresholdStr+'</strong>. Above these boundaries, iterations are distributed across workers with zero per-invocation heap allocations.</div></div></section>';
}

init();
""".replace("__REPORT__", embedded)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n<style>{stylesheet}</style>\n</head>\n<body>\n"
        '<header class="hero"><div class="eyebrow">mojagg performance lab</div>'
        "<h1>Fast paths, visible.</h1>"
        "<p>One reproducible snapshot across mojagg and numbagg. Ratios are normalized to mojagg, with lower values representing less time or memory.</p>"
        '<div class="meta" id="hero-meta"></div></header>\n'
        '<div class="app-layout">\n'
        '<aside class="sidebar" id="sidebar"></aside>\n'
        '<main class="app-main">\n'
        '<div class="toolbar"><input id="search" class="search" type="search" placeholder="Filter functions, cases, dtypes, axes…"><span class="muted" style="font-size:.8rem">Show:</span><span id="impls" style="display:flex;gap:12px;flex-wrap:wrap"></span></div>\n'
        '<div class="summary" id="summary"></div>\n'
        '<div id="top3-container"></div>\n'
        '<div id="worst3-container"></div>\n'
        '<div id="scaling-container"></div>\n'
        '<div id="device-profile"></div>\n'
        '<div id="threading-dispatch"></div>\n'
        '<div id="sections"></div>\n'
        '<div class="footer">Memory uses the peak allocation tracked by Python\'s <code>tracemalloc</code> during each call. It is a comparable signal for Python and NumPy allocations, not a complete process RSS measurement. Generated by <code>benchmarks/public_benchmark.py</code>.</div>\n'
        "</main>\n"
        "</div>\n"
        f"<script>{script}</script>\n</body>\n</html>\n"
    )


def run_benchmark(
    suite: BenchmarkSuite | None = None,
    output: str | Path = "docs/benchmarks/latest/index.html",
    *,
    json_output: str | Path | None = None,
) -> dict[str, Any]:
    """Run a suite, write HTML plus JSON, and return the report."""

    suite = Public() if suite is None else suite
    report = build_report(suite)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(report), encoding="utf-8")
    json_path = output_path.with_name("results.json") if json_output is None else Path(json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _parse_only(values: Sequence[str], suite: BenchmarkSuite) -> None:
    for value in values:
        if ":" not in value:
            raise ValueError("--only values must use SECTION:FUNCTION[,FUNCTION] syntax")
        section, names = value.split(":", 1)
        names_list = [name.strip() for name in names.split(",") if name.strip()]
        if section == "reduction":
            suite.reduction_functions = names_list
        elif section == "groupby":
            suite.groupby_functions = names_list
        elif section == "matrix":
            suite.matrix_functions = names_list
        elif section == "rolling":
            suite.rolling_functions = names_list
        elif section == "exponential":
            suite.exponential_functions = names_list
        elif section in {"fill", "non-reduction"}:
            suite.fill_functions = names_list
        else:
            raise ValueError(f"unknown section {section!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "public", "stress"), default="public")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/latest/index.html"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--device-name",
        help="human-readable device label; defaults to a generic benchmark host label",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SECTION:FUNCTIONS",
        help="select functions, e.g. --only reduction:nansum,nanmean",
    )
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--no-verify", action="store_true", help="skip result comparison before timing"
    )
    parser.add_argument("--no-numbagg", action="store_true")
    args = parser.parse_args(argv)
    suite_cls = {"quick": Quick, "public": Public, "stress": Stress}[args.profile]
    suite = suite_cls()
    suite.device_name = args.device_name
    _parse_only(args.only, suite)
    if args.warmups is not None:
        suite.warmups = args.warmups
    if args.repeats is not None:
        suite.repeats = args.repeats
    suite.verify_results = not args.no_verify
    suite.include_numbagg = not args.no_numbagg
    report = run_benchmark(suite, args.output, json_output=args.json_output)
    successful = sum(row["status"] == "ok" for row in report["records"])
    print(f"Wrote {args.output} ({successful} successful measurements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
