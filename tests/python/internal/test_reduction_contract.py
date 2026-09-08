"""SIMD masks/tails, dtype specializations and slice-local termination."""

import numpy as np
import pytest

import mojagg
from mojagg import _native
from mojagg._reduce import _resolve_axes


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_allnan_every_lane_and_tail(dtype):
    for n in (0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129):
        a = np.full(n, np.nan, dtype=dtype)
        assert bool(mojagg.allnan(a))
        for i in range(n):
            for value in (0.0, np.inf, -np.inf):
                a[i] = value
                assert not bool(mojagg.allnan(a)), (dtype, n, i, value)
            a[i] = np.nan


@pytest.mark.parametrize(
    "float_dtype,uint_dtype,bits",
    [
        (
            np.float32,
            np.uint32,
            [0x7FC00001, 0xFFC00002, 0x7F800001, 0xFF800001],
        ),
        (
            np.float64,
            np.uint64,
            [
                0x7FF8000000000001,
                0xFFF8000000000002,
                0x7FF0000000000001,
                0xFFF0000000000001,
            ],
        ),
    ],
)
def test_nan_signs_and_payloads(float_dtype, uint_dtype, bits):
    a = np.tile(np.array(bits, dtype=uint_dtype).view(float_dtype), 33)
    assert bool(mojagg.allnan(a))
    assert mojagg.nansum(a) == 0
    a[-1] = np.inf
    assert not bool(mojagg.allnan(a))
    assert mojagg.nansum(a) == np.inf


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_sum_block_boundaries_and_strides(dtype):
    for n in range(130):
        a = (np.arange(n) % 13 - 6).astype(dtype)
        if np.issubdtype(dtype, np.floating):
            a[::7] = np.nan
        for view in (a, a[::2], a[::-1], a[::-3]):
            np.testing.assert_equal(mojagg.nansum(view), np.nansum(view))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("threshold", [0, 2**60])
@pytest.mark.parametrize("strided", [False, True])
def test_multi_output_independence(dtype, threshold, strided):
    floating = np.issubdtype(dtype, np.floating)
    a = np.full((9, 32, 65), np.nan if floating else 1, dtype=dtype)
    if floating:
        a[0, ::3, 0] = np.inf
        a[-1, 1::3, -1] = -np.inf
    if strided:
        a = a[:, :, ::2]
    axes = _resolve_axes((0, 2), a)
    suffix = np.dtype(dtype).name.replace("float", "f").replace("int", "i")
    result = np.empty(32, dtype=np.bool_)
    getattr(_native, f"allnan_{suffix}")(a, axes, result, threshold)
    np.testing.assert_array_equal(result, np.isnan(a).all(axis=(0, 2)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("shape", [(0, 3, 17), (4, 3, 0), (4, 0, 17)])
def test_empty_multi_slices(dtype, shape):
    a = np.empty(shape, dtype=dtype)
    for op, ref in (
        (mojagg.allnan, lambda x: np.isnan(x).all(axis=(0, 2))),
        (mojagg.nansum, lambda x: np.nansum(x, axis=(0, 2))),
    ):
        np.testing.assert_array_equal(op(a, axis=(0, 2)), ref(a))
