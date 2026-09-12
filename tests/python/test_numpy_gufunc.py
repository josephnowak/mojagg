from __future__ import annotations

import numpy as np
import pytest

from examples.gufunc_design.numpy_gufunc import apply_gufunc


def test_scalar_output_accepts_heterogeneous_core_lengths():
    left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    right = np.arange(30, dtype=np.int32).reshape(2, 3, 5)
    seen = []

    def reduce_pair(x, y):
        seen.append((x.shape, y.shape, x.dtype, y.dtype))
        return x.sum(dtype=np.float64) - y.sum(dtype=np.float64)

    result = apply_gufunc(reduce_pair, [left, right])

    expected = np.empty((2, 3), dtype=np.float64)
    for index in np.ndindex(expected.shape):
        expected[index] = left[index].sum(dtype=np.float64) - right[index].sum(
            dtype=np.float64
        )
    np.testing.assert_array_equal(result, expected)
    assert len(seen) == 6
    assert all(shape == ((4,), (5,), np.dtype(np.float32), np.dtype(np.int32)) for shape in seen)


def test_multiple_outputs_and_nontrailing_core_axes():
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    weights = np.ones((2, 3, 4), dtype=np.float32)

    def transform(x, weight):
        return x * weight, x + weight

    first, second = apply_gufunc(
        transform,
        [values, weights],
        core_axes=(1,),
    )

    expected_values = np.moveaxis(values, 1, -1)
    expected_weights = np.moveaxis(weights, 1, -1)
    np.testing.assert_array_equal(first, expected_values * expected_weights)
    np.testing.assert_array_equal(second, expected_values + expected_weights)
    assert first.dtype == np.float64
    assert second.dtype == np.float64
    assert first.shape == (2, 4, 3)


def test_non_affine_multidimensional_core_requires_explicit_copy():
    values = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5)

    with pytest.raises(ValueError, match="zero-copy"):
        apply_gufunc(lambda x: x.sum(), [values], core_axes=(1, 3))

    result = apply_gufunc(
        lambda x: x.sum(),
        [values],
        core_axes=(1, 3),
        allow_copy=True,
    )
    expected = values.sum(axis=(1, 3))
    np.testing.assert_array_equal(result, expected)


def test_preallocated_output_supports_empty_outer_domain():
    values = np.empty((0, 5), dtype=np.float32)
    destination = np.empty((0,), dtype=np.float64)

    result = apply_gufunc(lambda x: x.sum(dtype=np.float64), [values], out=destination)

    assert result is destination
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_outer_shape_must_match():
    left = np.empty((2, 3), dtype=np.float32)
    right = np.empty((4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="outer shapes must match"):
        apply_gufunc(lambda x, y: x.sum() + y.sum(), [left, right])
