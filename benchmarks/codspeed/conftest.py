"""Shared deterministic inputs for the small per-change CodSpeed suite.

Every input is fixed and moderate: the suite has to cover each public family on
every change without turning into the publication matrix owned by
``benchmarks/public_benchmark.py``. Case names are part of the CodSpeed
benchmark identity, so they are kept stable and descriptive of the layout being
measured rather than of the exact size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

_SEED = 7


def _nan_values(shape: tuple[int, ...], *, stride: int = 97, dtype: Any = np.float64):
    """Deterministic array with NaNs injected on a fixed flat stride."""
    rng = np.random.default_rng(_SEED)
    values = rng.normal(1.0, 0.25, size=shape).astype(dtype)
    values.reshape(-1)[::stride] = np.nan
    return values


@pytest.fixture(scope="module")
def hot_path_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    values = rng.normal(1.0, 0.25, size=(16, 8_192)).astype(np.float64)
    values[:, ::97] = np.nan

    labels = np.arange(values.shape[1], dtype=np.int32) % 256

    matrix = rng.normal(1.0, 0.25, size=(16, 1_024)).astype(np.float64)
    matrix[:, ::113] = np.nan

    moving = rng.normal(1.0, 0.25, size=(8, 8_192)).astype(np.float64)
    moving[:, ::89] = np.nan

    return {
        "values": values,
        "labels": labels,
        "matrix": matrix,
        "moving": moving,
    }


# --------------------------------------------------------------------------
# Reductions
#
# The three layouts are the ones the driver treats differently: reducing the
# contiguous axis, reducing the strided axis, and collapsing every axis.
# --------------------------------------------------------------------------

REDUCTION_CASES = ("rows", "cols", "full")


@pytest.fixture(scope="session")
def reduction_cases() -> dict[str, tuple[np.ndarray, Any]]:
    """name -> (values, axis) for the float64 reduction layouts."""
    return {
        "rows": (_nan_values((64, 4_096)), 1),
        "cols": (_nan_values((4_096, 64)), 0),
        "full": (_nan_values((256, 1_024)), None),
    }


@pytest.fixture(scope="session")
def float32_case(reduction_cases) -> tuple[np.ndarray, Any]:
    """The contiguous layout in float32, which selects a different kernel."""
    values, axis = reduction_cases["rows"]
    return values.astype(np.float32), axis


@pytest.fixture(scope="session")
def int64_case() -> tuple[np.ndarray, Any]:
    """Integer input: no NaNs to mask, and a distinct kernel instantiation."""
    rng = np.random.default_rng(_SEED)
    return rng.integers(-1_000, 1_000, size=(64, 4_096), dtype=np.int64), 1


# --------------------------------------------------------------------------
# Grouped reductions
#
# Cardinality is the interesting dimension: 256 groups keep the accumulators
# hot in cache, 4096 groups push the scatter path out of it.
# --------------------------------------------------------------------------

GROUP_CASES = ("low-cardinality", "high-cardinality")


@pytest.fixture(scope="session")
def group_cases() -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """name -> (values, labels, num_labels); also covers both label dtypes."""
    values = _nan_values((16, 8_192))
    return {
        "low-cardinality": (
            values,
            np.arange(values.shape[1], dtype=np.int32) % 256,
            256,
        ),
        "high-cardinality": (
            values,
            (np.arange(values.shape[1], dtype=np.int64) * 7) % 4_096,
            4_096,
        ),
    }


# --------------------------------------------------------------------------
# Moving windows
# --------------------------------------------------------------------------

MOVING_CASES = ("short-window", "long-window")

# name -> (window, min_count)
MOVING_WINDOWS = {
    "short-window": (16, 8),
    "long-window": (256, 128),
}


@pytest.fixture(scope="session")
def moving_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two independent series, for the unary and the pairwise window kernels."""
    first = _nan_values((8, 8_192), stride=89)
    second = _nan_values((8, 8_192), stride=83) * -0.5
    return first, second


# --------------------------------------------------------------------------
# Matrix reductions — static kernels take (..., vars, obs), the moving
# kernels take the transposed (..., obs, vars) core.
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def matrix_values() -> np.ndarray:
    return _nan_values((16, 1_024), stride=113)


@pytest.fixture(scope="session")
def moving_matrix_values() -> np.ndarray:
    return _nan_values((1_024, 16), stride=113)


# --------------------------------------------------------------------------
# Fill
# --------------------------------------------------------------------------

FILL_CASES = ("rows", "cols")


@pytest.fixture(scope="session")
def fill_cases() -> dict[str, tuple[np.ndarray, int]]:
    """name -> (values, axis); sparse NaNs, filled along each layout."""
    return {
        "rows": (_nan_values((8, 8_192), stride=89), 1),
        "cols": (_nan_values((8_192, 8), stride=89), 0),
    }
