from __future__ import annotations

import numpy as np
import pytest

import mojagg


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("n", [0, 1, 3, 7, 16, 31, 32, 33, 65])
def test_group_nansum_dense_labels(dtype, n):
    values = np.arange(n, dtype=dtype)
    labels = (np.arange(n, dtype=np.int64) % 3).astype(np.int32)
    if np.issubdtype(dtype, np.floating) and n:
        values[::5] = np.nan

    expected = np.zeros(3, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        np.add.at(expected, labels, np.nan_to_num(values, nan=0))
    else:
        np.add.at(expected, labels, values)

    np.testing.assert_equal(mojagg.group_nansum(values, labels, num_labels=3), expected)


@pytest.mark.parametrize(
    "function_name",
    [
        "group_nansum",
        "group_nanprod",
        "group_nancount",
        "group_nansum_of_squares",
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("label_dtype", [np.int32, np.int64])
def test_group_identity_and_count_parity(function_name, dtype, label_dtype):
    import numbagg

    values = np.array([1, 2, -3, 4, 5, 6], dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        values[[1, 4]] = np.nan
    labels = np.array([0, 0, 1, -1, 2, 1], dtype=label_dtype)
    function = getattr(mojagg, function_name)

    actual = function(values, labels, num_labels=4)
    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        expected = getattr(numbagg, function_name)(values, labels, num_labels=4)
    finally:
        if registered:
            mojagg.register()

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "function_name",
    [
        "group_nansum",
        "group_nanprod",
        "group_nancount",
        "group_nansum_of_squares",
    ],
)
def test_group_identity_and_count_empty_groups(function_name):
    import numbagg

    values = np.array([np.nan, np.nan], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int32)
    function = getattr(mojagg, function_name)

    actual = function(values, labels, num_labels=4)
    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        expected = getattr(numbagg, function_name)(values, labels, num_labels=4)
    finally:
        if registered:
            mojagg.register()

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "function_name",
    [
        "group_nansum",
        "group_nanprod",
        "group_nancount",
        "group_nansum_of_squares",
    ],
)
def test_group_identity_and_count_axis_strided_parity(function_name):
    import numbagg

    values = np.arange(2 * 3 * 8, dtype=np.float64).reshape(2, 3, 8)
    values[0, 1, 3] = np.nan
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)
    function = getattr(mojagg, function_name)
    strided_values = values[..., ::-1]
    strided_labels = labels[::-1]

    actual = function(
        strided_values,
        strided_labels,
        axis=-1,
        num_labels=4,
    )
    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        expected = getattr(numbagg, function_name)(
            strided_values,
            strided_labels,
            axis=-1,
            num_labels=4,
        )
    finally:
        if registered:
            mojagg.register()

    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_group_nansum_axis():
    values = np.arange(48.0).reshape(2, 3, 8)
    values[0, 1, 3] = np.nan
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32)

    grouped = mojagg.group_nansum(values, labels, axis=-1)
    expected = np.stack(
        [
            np.nansum(values[..., 0::3], axis=-1),
            np.nansum(values[..., 1::3], axis=-1),
            np.nansum(values[..., 2::3], axis=-1),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(grouped, expected, equal_nan=True)


def test_group_nansum_multiaxis_and_strided():
    values = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    labels = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    expected = np.zeros((4, 2), dtype=np.float64)
    for i in range(2):
        for j in range(3):
            expected[:, labels[i, j]] += values[i, j, :]

    result = mojagg.group_nansum(values, labels, axis=(0, 1))
    np.testing.assert_equal(result, expected)

    strided_values = values[:, :, ::-1]
    strided_labels = np.array([0, 1, 0, 1], dtype=np.int32)
    expected_strided = np.stack(
        [
            np.nansum(strided_values[..., 0::2], axis=-1),
            np.nansum(strided_values[..., 1::2], axis=-1),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(
        mojagg.group_nansum(strided_values, strided_labels, axis=-1),
        expected_strided,
        equal_nan=True,
    )


def test_group_nansum_requires_dense_supported_labels():
    values = np.ones(4, dtype=np.float64)
    with pytest.raises(TypeError, match="group labels"):
        mojagg.group_nansum(values, np.array([0, 1, 2, 3], dtype=np.int16))


def test_group_nansum_skips_negative_labels():
    values = np.ones(3)
    labels = np.array([0, -1, 2])
    np.testing.assert_equal(
        mojagg.group_nansum(values, labels),
        np.array([1.0, 0.0, 1.0]),
    )


def test_group_nansum_rejects_undersized_output():
    with pytest.raises(ValueError, match="maximum label"):
        mojagg.group_nansum(np.ones(2), np.array([0, 3]), num_labels=3)


def test_group_nansum_empty_labels():
    result = mojagg.group_nansum(np.empty(0), np.empty(0, dtype=np.int64))
    assert result.shape == (0,)


def test_group_nansum_empty_outer_axis_preserves_group_count():
    result = mojagg.group_nansum(np.empty((0, 3)), np.array([0, 2, 1]), axis=-1)
    assert result.shape == (0, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("label_dtype", [np.int32, np.int64])
def test_group_nansum_repeated_labels_parity(dtype, label_dtype):
    import numbagg

    values = np.arange(130, dtype=dtype).reshape(2, 65)[:, ::-1]
    values[:, ::7] = np.nan
    labels = (np.arange(65) % 3).astype(label_dtype)
    actual = mojagg.group_nansum(values, labels, axis=-1, num_labels=5)
    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        expected = numbagg.group_nansum(values, labels, axis=-1, num_labels=5)
    finally:
        if registered:
            mojagg.register()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True)


GROUPED_REST_FUNCTIONS = [
    "group_nanmean",
    "group_nanmin",
    "group_nanmax",
    "group_nanargmin",
    "group_nanargmax",
    "group_nanfirst",
    "group_nanlast",
    "group_nanany",
    "group_nanall",
    "group_nanvar",
    "group_nanstd",
]


def _numbagg_group_result(function_name, values, labels, **kwargs):
    import numbagg

    registered = mojagg.is_registered()
    mojagg.unregister()
    try:
        return getattr(numbagg, function_name)(values, labels, **kwargs)
    finally:
        if registered:
            mojagg.register()


@pytest.mark.parametrize("function_name", GROUPED_REST_FUNCTIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("label_dtype", [np.int32, np.int64])
def test_group_remaining_functions_parity(function_name, dtype, label_dtype):
    if np.issubdtype(dtype, np.floating):
        values = np.array([3, np.nan, -2, 4, np.nan, 1], dtype=dtype)
    else:
        values = np.array([3, 2, -2, 4, 5, 1], dtype=dtype)
    labels = np.array([0, 0, 1, -1, 2, 1], dtype=label_dtype)
    kwargs = {"num_labels": 4}
    if function_name in {"group_nanvar", "group_nanstd"}:
        kwargs["ddof"] = 1

    actual = getattr(mojagg, function_name)(values, labels, **kwargs)
    expected = _numbagg_group_result(function_name, values, labels, **kwargs)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("function_name", GROUPED_REST_FUNCTIONS)
def test_group_remaining_functions_axis_and_empty_groups(function_name):
    values = np.array(
        [[3.0, np.nan, -2.0, 4.0], [5.0, 1.0, np.nan, -6.0]],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 0, 2], dtype=np.int64)
    kwargs = {"axis": -1, "num_labels": 4}
    if function_name in {"group_nanvar", "group_nanstd"}:
        kwargs["ddof"] = 1

    actual = getattr(mojagg, function_name)(values, labels, **kwargs)
    expected = _numbagg_group_result(function_name, values, labels, **kwargs)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("function_name", GROUPED_REST_FUNCTIONS)
def test_group_remaining_functions_empty_input(function_name):
    values = np.empty(0, dtype=np.float64)
    labels = np.empty(0, dtype=np.int32)
    kwargs = {"num_labels": 3}
    if function_name in {"group_nanvar", "group_nanstd"}:
        kwargs["ddof"] = 1

    actual = getattr(mojagg, function_name)(values, labels, **kwargs)
    expected = _numbagg_group_result(function_name, values, labels, **kwargs)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("function_name", GROUPED_REST_FUNCTIONS)
def test_group_remaining_functions_multiaxis_parity(function_name):
    values = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    values[0, 0, 1] = np.nan
    values[1, 2, 3] = np.nan
    labels = (np.arange(3 * 4, dtype=np.int64).reshape(3, 4) % 3).copy()
    labels[0, 2] = -1
    kwargs = {"axis": (1, 2), "num_labels": 4}
    if function_name in {"group_nanvar", "group_nanstd"}:
        kwargs["ddof"] = 0

    actual = getattr(mojagg, function_name)(values, labels, **kwargs)
    expected = _numbagg_group_result(function_name, values, labels, **kwargs)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True)
