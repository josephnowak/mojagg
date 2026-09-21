from __future__ import annotations

import numpy as np
import pytest

import mojagg


def _numbagg_move_sum(values, *, window, min_count, axis):
    import numbagg

    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        return numbagg.move_sum(
            values,
            window=window,
            min_count=min_count,
            axis=axis,
        )
    finally:
        if registered:
            mojagg.register()


def _numbagg_move(name, *arrays, window, min_count, axis):
    import numbagg

    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        return getattr(numbagg, name)(
            *arrays,
            window=window,
            min_count=min_count,
            axis=axis,
        )
    finally:
        if registered:
            mojagg.register()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window", [1, 3, 8, 9])
@pytest.mark.parametrize("min_count", [None, 0, 1, 3])
def test_move_sum_matches_numbagg(dtype, window, min_count):
    values = np.arange(23, dtype=dtype) - dtype(7)
    values[[1, 8, 17]] = np.nan

    actual = mojagg.move_sum(
        values,
        window=window,
        min_count=min_count,
    )
    expected = _numbagg_move_sum(
        values,
        window=window,
        min_count=min_count,
        axis=-1,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_move_sum_axis_and_strided_input(axis):
    values = np.arange(3 * 19, dtype=np.float64).reshape(3, 19)
    values[0, 2] = np.nan
    values[2, 14] = np.nan
    if axis == -1:
        values = values[:, ::-1]

    actual = mojagg.move_sum(values, window=3, min_count=2, axis=axis)
    expected = _numbagg_move_sum(
        values,
        window=3,
        min_count=2,
        axis=axis,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_move_sum_axis_empty_returns_input():
    values = np.arange(5, dtype=np.float64)
    result = mojagg.move_sum(values, window=0, axis=())
    assert result is values


def test_move_sum_validates_window_and_min_count():
    values = np.ones(5, dtype=np.float64)
    with pytest.raises(ValueError, match="min_count"):
        mojagg.move_sum(values, window=3, min_count=-1)
    with pytest.raises(ValueError, match="window"):
        mojagg.move_sum(values, window=0)
    with pytest.raises(ValueError, match="window"):
        mojagg.move_sum(values, window=6)
    with pytest.raises(ValueError, match="one axis"):
        mojagg.move_sum(values, window=3, axis=(0, 0))


def test_move_sum_float32_repeated_window_matches_numbagg():
    random = np.random.RandomState(0)
    values = np.tile((random.rand(10) * 1e13).astype(np.float32), 100)
    result = mojagg.move_sum(values, window=10)
    expected = _numbagg_move_sum(
        values,
        window=10,
        min_count=None,
        axis=-1,
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_move_sum_float16_promotes_to_float32():
    values = np.arange(17, dtype=np.float16)
    values[[2, 11]] = np.nan
    actual = mojagg.move_sum(values, window=5, min_count=2)
    expected = _numbagg_move_sum(
        values,
        window=5,
        min_count=2,
        axis=-1,
    )
    assert actual.dtype == np.dtype(np.float32)
    assert expected.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("name", ["move_mean", "move_var", "move_std"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window", [1, 3, 8, 9])
@pytest.mark.parametrize("min_count", [None, 0, 1, 3])
def test_moving_unary_matches_numbagg(name, dtype, window, min_count):
    values = np.arange(23, dtype=dtype) - dtype(7)
    values[[1, 8, 17]] = np.nan

    actual = getattr(mojagg, name)(
        values,
        window=window,
        min_count=min_count,
    )
    expected = _numbagg_move(
        name,
        values,
        window=window,
        min_count=min_count,
        axis=-1,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    tolerance = (2e-5, 2e-6) if dtype == np.float32 else (1e-10, 1e-12)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance[0],
        atol=tolerance[1],
        equal_nan=True,
    )


@pytest.mark.parametrize("name", ["move_cov", "move_corr"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window", [1, 3, 8, 9])
@pytest.mark.parametrize("min_count", [None, 0, 1, 3])
def test_moving_binary_matches_numbagg(name, dtype, window, min_count):
    first = np.arange(23, dtype=dtype) - dtype(7)
    second = first * dtype(0.5) + dtype(3)
    first[[1, 8, 17]] = np.nan
    second[[4, 8, 19]] = np.nan

    actual = getattr(mojagg, name)(
        first,
        second,
        window=window,
        min_count=min_count,
    )
    expected = _numbagg_move(
        name,
        first,
        second,
        window=window,
        min_count=min_count,
        axis=-1,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    tolerance = (2e-5, 2e-6) if dtype == np.float32 else (1e-10, 1e-12)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance[0],
        atol=tolerance[1],
        equal_nan=True,
    )


@pytest.mark.parametrize("name", ["move_mean", "move_var", "move_std"])
def test_moving_unary_axis_and_strided_input(name):
    values = np.arange(3 * 19, dtype=np.float64).reshape(3, 19)
    values[0, 2] = np.nan
    values[2, 14] = np.nan
    values = values[:, ::-1]

    actual = getattr(mojagg, name)(values, window=3, min_count=2, axis=1)
    expected = _numbagg_move(
        name,
        values,
        window=3,
        min_count=2,
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("name", ["move_cov", "move_corr"])
def test_moving_binary_broadcast_and_axis(name):
    first = np.arange(2 * 19, dtype=np.float64).reshape(2, 19)
    second = np.arange(19, dtype=np.float64) + 1
    first[0, 2] = np.nan
    second[14] = np.nan

    actual = getattr(mojagg, name)(
        first,
        second,
        window=3,
        min_count=2,
        axis=-1,
    )
    expected = _numbagg_move(
        name,
        first,
        second,
        window=3,
        min_count=2,
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_moving_binary_empty_axis_is_rejected():
    values = np.arange(5, dtype=np.float64)
    with pytest.raises(ValueError, match="empty tuple"):
        mojagg.move_cov(values, values, window=3, axis=())
