"""Mean parity, especially unfinished sum/count merging and promotion."""

import warnings

import numbagg
import numpy as np
import pytest

import mojagg
from mojagg import _native
from mojagg._reduce import _resolve_axes


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("threshold", [0, 2**60])
def test_weighted_multi(dtype, strided, threshold):
    a = np.array(
        [
            [[1, np.nan, 3], [np.nan, np.nan, np.nan]],
            [[10, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        ],
        dtype=dtype,
    )
    if strided:
        storage = np.full((2, 2, 6), np.nan, dtype=dtype)
        storage[:, :, ::2] = a
        a = storage[:, :, ::2]
    with mojagg.config(parallel_threshold=threshold):
        result = mojagg.nanmean(a, axis=(0, 2))
    np.testing.assert_allclose(result, [14 / 3, np.nan], rtol=1e-6, equal_nan=True)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mean_tails_and_layouts(dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for n in range(130):
            a = (np.arange(n) % 13 - 6).astype(dtype)
            a[::7] = np.nan
            for view in (a, a[::2], a[::-1], a[::-3]):
                np.testing.assert_allclose(
                    mojagg.nanmean(view),
                    numbagg.nanmean(view),
                    rtol=1e-6,
                    atol=1e-8,
                    equal_nan=True,
                )


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_mean_dtype_promotion(dtype):
    a = np.array([0, 1, 1, 0], dtype=dtype)
    before = a.copy()
    expected = np.asarray(numbagg.nanmean(a))
    result = mojagg.nanmean(a)
    assert result.dtype == expected.dtype
    np.testing.assert_equal(result, expected)
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3, 4), (4, 3, 0), (4, 0, 3)])
def test_mean_empty_and_all_nan(dtype, shape):
    a = np.full(shape, np.nan, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for axis in (None, 0, -1, tuple(range(a.ndim))):
            np.testing.assert_array_equal(
                mojagg.nanmean(a, axis=axis), numbagg.nanmean(a, axis=axis)
            )
    assert np.isnan(mojagg.nanmean(np.full(129, np.nan, dtype=dtype)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mean_broadcast_and_axis_order(dtype):
    a = np.arange(3 * 5 * 17, dtype=dtype).reshape(3, 5, 17)
    a[::2, ::2, ::3] = np.nan
    for view in (a, a.T, a[::-1, :, ::-1], np.broadcast_to(a[:1], a.shape)):
        for axis in (None, 0, 1, -1, (0, 2), (2, 0), (-1, -3), (0, 1), ()):
            np.testing.assert_allclose(
                mojagg.nanmean(view, axis=axis),
                numbagg.nanmean(view, axis=axis),
                rtol=1e-6,
                equal_nan=True,
            )


def test_mean_float32_accumulates_in_float64():
    for a in (
        np.full(65, np.finfo(np.float32).max, dtype=np.float32),
        np.array([1e8, 1, -1e8], dtype=np.float32),
    ):
        result = mojagg.nanmean(a)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, numbagg.nanmean(a))


@pytest.mark.parametrize(
    "a",
    [
        np.array(["1", "2"]),
        np.array([1, 2], dtype=object),
        np.array([1 + 2j], dtype=np.complex64),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[D]"),
    ],
)
def test_mean_unsupported_dtype(a):
    with pytest.raises(TypeError, match="nanmean does not support dtype.*supported:"):
        mojagg.nanmean(a)


@pytest.mark.parametrize("dtype,suffix", [(np.float32, "f32"), (np.float64, "f64")])
def test_mean_native_dtype_validation(dtype, suffix):
    a = np.array([1, 2], dtype=dtype)
    entry = getattr(_native, f"nanmean_{suffix}")
    with pytest.raises(Exception, match="output must be"):
        entry(a, _resolve_axes(None, a), np.empty((), dtype=np.int64), 2**60)
    with pytest.raises(Exception, match="expected dtype"):
        entry(a.astype(np.int64), (0,), np.empty((), dtype=dtype), 2**60)
