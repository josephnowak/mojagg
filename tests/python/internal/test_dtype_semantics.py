from __future__ import annotations

import numpy as np
import pytest

import mojagg


def _numbagg_result(function_name, values, *args, **kwargs):
    import numbagg

    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        return getattr(numbagg, function_name)(values, *args, **kwargs)
    finally:
        if registered:
            mojagg.register()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.int64])
def test_nanquantile_result_dtype_matches_numbagg(dtype):
    values = np.array([1, 2, 3], dtype=dtype)

    expected = _numbagg_result("nanquantile", values, 0.5)
    actual = mojagg.nanquantile(values, 0.5)

    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.int32, np.int64])
def test_nancovmatrix_integer_promotion_matches_numbagg(dtype):
    values = np.arange(8, dtype=dtype).reshape(2, 4)

    expected = _numbagg_result("nancovmatrix", values)
    actual = mojagg.nancovmatrix(values)

    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, equal_nan=True)
