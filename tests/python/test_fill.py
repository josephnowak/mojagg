from __future__ import annotations

import numpy as np
import pytest

import mojagg


def _numbagg_fill(op_name: str, a: np.ndarray, limit: int | None = None, axis: int = -1):
    import numbagg

    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        fn = getattr(numbagg, op_name)
        return fn(a, limit=limit, axis=axis)
    finally:
        if registered:
            mojagg.register()


@pytest.mark.parametrize("op_name", ["ffill", "bfill"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("length", [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100])
@pytest.mark.parametrize("limit", [None, 0, 1, 3, 32, 150])
def test_fill_1d_boundary_lengths_and_limits(op_name, dtype, length, limit):
    rng = np.random.RandomState(42 + length)
    values = rng.randn(length).astype(dtype)
    if length > 0:
        # Inject NaNs at various indices
        nan_mask = rng.rand(length) < 0.3
        values[nan_mask] = np.nan

    mojagg_fn = getattr(mojagg, op_name)
    actual = mojagg_fn(values, limit=limit)
    expected = _numbagg_fill(op_name, values, limit=limit, axis=-1)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("op_name", ["ffill", "bfill"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fill_large_array_throughput_and_parity(op_name, dtype):
    rng = np.random.RandomState(123)
    n = 100_000
    values = rng.randn(n).astype(dtype)
    # Inject 20% NaNs in bursts
    for _ in range(200):
        start = rng.randint(0, n - 50)
        span = rng.randint(1, 50)
        values[start : start + span] = np.nan

    mojagg_fn = getattr(mojagg, op_name)
    actual = mojagg_fn(values, limit=10)
    expected = _numbagg_fill(op_name, values, limit=10, axis=-1)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("op_name", ["ffill", "bfill"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "pattern", ["all_valid", "all_nan", "alternating", "leading_nan", "trailing_nan"]
)
@pytest.mark.parametrize("limit", [None, 0, 1, 5])
def test_fill_data_patterns(op_name, dtype, pattern, limit):
    length = 40
    if pattern == "all_valid":
        values = np.arange(1, length + 1, dtype=dtype)
    elif pattern == "all_nan":
        values = np.full(length, np.nan, dtype=dtype)
    elif pattern == "alternating":
        values = np.arange(1, length + 1, dtype=dtype)
        values[1::2] = np.nan
    elif pattern == "leading_nan":
        values = np.arange(1, length + 1, dtype=dtype)
        values[:10] = np.nan
    elif pattern == "trailing_nan":
        values = np.arange(1, length + 1, dtype=dtype)
        values[-10:] = np.nan

    mojagg_fn = getattr(mojagg, op_name)
    actual = mojagg_fn(values, limit=limit)
    expected = _numbagg_fill(op_name, values, limit=limit, axis=-1)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("op_name", ["ffill", "bfill"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_fill_2d_axes(op_name, dtype, axis):
    rng = np.random.RandomState(999)
    values = rng.randn(20, 35).astype(dtype)
    nan_mask = rng.rand(*values.shape) < 0.25
    values[nan_mask] = np.nan

    mojagg_fn = getattr(mojagg, op_name)
    actual = mojagg_fn(values, limit=2, axis=axis)
    expected = _numbagg_fill(op_name, values, limit=2, axis=axis)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("op_name", ["ffill", "bfill"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_fill_integer_dtypes_bulk_copy(op_name, dtype):
    values = np.arange(50, dtype=dtype)
    mojagg_fn = getattr(mojagg, op_name)
    actual = mojagg_fn(values, limit=3)
    expected = _numbagg_fill(op_name, values, limit=3, axis=-1)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    # Ensure fresh copy
    assert actual is not values


@pytest.mark.parametrize("op_name", ["ffill", "bfill"])
def test_fill_multi_axis_tuple(op_name):
    values = np.array(
        [
            [[np.nan, 1.0, np.nan, 4.0], [10.0, np.nan, 12.0, np.nan]],
            [[np.nan, np.nan, 22.0, np.nan], [30.0, np.nan, np.nan, 33.0]],
        ],
        dtype=np.float64,
    )
    mojagg_fn = getattr(mojagg, op_name)
    actual = mojagg_fn(values, axis=(0, 2))

    # Reference computation by flattening the selected axes
    execution = np.moveaxis(values, (0, 2), (-2, -1))
    flattened = execution.reshape(execution.shape[:-2] + (-1,))
    expected_execution = np.empty_like(flattened)
    for outer, row in enumerate(flattened):
        expected_execution[outer] = _numbagg_fill(op_name, row, limit=None, axis=-1)
    expected = np.moveaxis(
        expected_execution.reshape(execution.shape),
        (-2, -1),
        (0, 2),
    )

    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.shape == values.shape
