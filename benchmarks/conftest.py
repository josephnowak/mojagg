"""Benchmark fixtures shared across families.

Deterministic NaN patterns (no randomness — CodSpeed needs stable workloads).
Cases deliberately include adversarial shapes from the driver investigation:
row-heavy, column-heavy, strided, and full reductions.

`--benchmark-quick` (pixi run bench-quick) shrinks sizes for local smoke runs.
"""

from __future__ import annotations

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-quick",
        action="store_true",
        default=False,
        help="Run a reduced benchmark matrix (local smoke).",
    )


@pytest.fixture(scope="session")
def quick(request) -> bool:
    return request.config.getoption("--benchmark-quick")


def _f64(shape, nan_frac=0.15, seed=1) -> np.ndarray:
    """Deterministic float64 array with a fixed NaN fraction."""
    rs = np.random.RandomState(seed)
    a = rs.rand(*shape)
    a[rs.rand(*shape) < nan_frac] = np.nan
    return np.ascontiguousarray(a)


def _f32(shape, nan_frac=0.15, seed=1) -> np.ndarray:
    return _f64(shape, nan_frac, seed).astype(np.float32)


# (name, shape, axis) — sizes halved under --benchmark-quick.
_CASES = [
    ("row-major-10kx1k", (10_000, 1_000), 1),
    ("tiny-rows-500kx8", (500_000, 8), 1),
    ("col-major-1kx10k", (1_000, 10_000), 0),
    ("full-2kx5k", (2_000, 5_000), None),
]

_QUICK_CASES = [
    ("row-major-2kx256", (2_000, 256), 1),
    ("tiny-rows-50kx8", (50_000, 8), 1),
    ("col-major-256x2k", (256, 2_000), 0),
    ("full-1kx1k", (1_000, 1_000), None),
]


@pytest.fixture(scope="session")
def cases(quick):
    """List of (name, f64 array, axis)."""
    src = _QUICK_CASES if quick else _CASES
    return [(name, _f64(shape), axis) for name, shape, axis in src]


@pytest.fixture(scope="session", params=[np.float32, np.float64], ids=["f32", "f64"])
def mean_cases(request, cases):
    """Prepare dtype variants outside the benchmarked functions."""
    return [(name, a.astype(request.param, copy=False), axis) for name, a, axis in cases]
