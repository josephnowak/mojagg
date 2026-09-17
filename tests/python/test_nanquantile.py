"""Parity tests for nanquantile, including the multi-kth selection path.

``NanQuantileKernel`` answers several quantiles either by ordering the whole
core span or by selecting only the requested order statistics.  The threshold
between the two is a performance decision, so every case below is checked
against numbagg to prove both paths are numerically identical.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojagg

# Lengths straddle the selection thresholds in src/mojagg/nanfuncs/nanquantile.mojo:
# below 2048 always sorts, 2048+ selects for few indices, 262144+ selects for up
# to 20 indices.
SHORT = 512
SMALL_TIER = 4096
MEDIUM_TIER = 20_000
LARGE_TIER = 300_000


def _numbagg_result(function_name, values, *args, **kwargs):
    import numbagg

    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        return getattr(numbagg, function_name)(values, *args, **kwargs)
    finally:
        if registered:
            mojagg.register()


def _random_values(length, *, nan_fraction=0.0, seed=0, shape=None):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(length)
    if nan_fraction:
        mask = rng.random(length) < nan_fraction
        values[mask] = np.nan
    if shape is not None:
        values = values.reshape(shape)
    return values


def _assert_matches_numbagg(values, quantiles, **kwargs):
    expected = _numbagg_result("nanquantile", values, quantiles, **kwargs)
    actual = mojagg.nanquantile(values, quantiles, **kwargs)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("length", [SHORT, 2048, SMALL_TIER, MEDIUM_TIER])
@pytest.mark.parametrize(
    "quantiles",
    [
        [0.25, 0.75],
        [0.25, 0.5, 0.75],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        list(np.linspace(0.05, 0.95, 9)),
        list(np.linspace(0.02, 0.98, 13)),
        list(np.linspace(0.01, 0.99, 21)),
    ],
)
def test_multiple_quantiles_match_numbagg(length, quantiles):
    values = _random_values(length, seed=length)

    _assert_matches_numbagg(values, quantiles)


@pytest.mark.parametrize("nan_fraction", [0.0, 0.3, 0.9])
def test_selection_path_with_nans_matches_numbagg(nan_fraction):
    values = _random_values(SMALL_TIER, nan_fraction=nan_fraction, seed=7)

    _assert_matches_numbagg(values, [0.25, 0.5, 0.75])


def test_large_span_with_many_quantiles_matches_numbagg():
    values = _random_values(LARGE_TIER, nan_fraction=0.1, seed=11)

    _assert_matches_numbagg(values, list(np.linspace(0.05, 0.95, 9)))


def test_endpoint_quantiles_match_numbagg():
    values = _random_values(SMALL_TIER, seed=3)

    _assert_matches_numbagg(values, [0.0, 0.5, 1.0])


def test_duplicate_values_match_numbagg():
    values = np.repeat(np.arange(8, dtype=np.float64), SMALL_TIER // 8)

    _assert_matches_numbagg(values, [0.0, 0.125, 0.5, 0.875, 1.0])


def test_constant_values_match_numbagg():
    values = np.full(SMALL_TIER, 4.0)

    _assert_matches_numbagg(values, [0.1, 0.5, 0.9])


def test_infinities_match_numbagg():
    values = _random_values(SMALL_TIER, seed=5)
    values[0] = np.inf
    values[1] = -np.inf

    _assert_matches_numbagg(values, [0.0, 0.5, 1.0])


def test_repeated_quantiles_match_numbagg():
    values = _random_values(SMALL_TIER, seed=13)

    _assert_matches_numbagg(values, [0.5, 0.5, 0.25, 0.25])


def test_unsorted_quantiles_match_numbagg():
    values = _random_values(SMALL_TIER, seed=17)

    _assert_matches_numbagg(values, [0.9, 0.1, 0.5, 0.2])


def test_nan_quantiles_match_numbagg():
    values = _random_values(SMALL_TIER, seed=19)

    _assert_matches_numbagg(values, [0.25, np.nan, 0.75])


def test_all_nan_input_matches_numbagg():
    values = np.full(SMALL_TIER, np.nan)

    _assert_matches_numbagg(values, [0.25, 0.5, 0.75])


def test_single_valid_value_matches_numbagg():
    values = np.full(SMALL_TIER, np.nan)
    values[SMALL_TIER // 2] = 1.5

    _assert_matches_numbagg(values, [0.0, 0.5, 1.0])


@pytest.mark.parametrize("axis", [0, 1, -1, None, (0, 1)])
def test_axis_variants_match_numbagg(axis):
    values = _random_values(4 * SMALL_TIER, nan_fraction=0.2, seed=23, shape=(4, SMALL_TIER))

    _assert_matches_numbagg(values, [0.25, 0.5, 0.75], axis=axis)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_dtypes_match_numbagg(dtype):
    values = np.arange(SMALL_TIER, dtype=dtype)

    _assert_matches_numbagg(values, [0.25, 0.5, 0.75])


def test_empty_input_matches_numbagg():
    values = np.empty(0, dtype=np.float64)

    _assert_matches_numbagg(values, [0.25, 0.75])


def test_nanmedian_matches_numbagg():
    values = _random_values(LARGE_TIER, nan_fraction=0.15, seed=29)

    expected = _numbagg_result("nanmedian", values)
    actual = mojagg.nanmedian(values)

    np.testing.assert_allclose(actual, expected, equal_nan=True)
