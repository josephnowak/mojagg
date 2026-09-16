from __future__ import annotations

import numpy as np

import mojagg


def test_matrix_places_selected_core_at_the_end():
    values = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)

    actual = mojagg.nancovmatrix(values, axis=(0, 2))
    moved = np.moveaxis(values, (0, 2), (-2, -1))
    expected = np.stack(
        [np.cov(moved[outer], rowvar=True, ddof=1) for outer in range(4)],
        axis=0,
    )

    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.shape == (4, 2, 2)


def test_fill_flattens_multiple_selected_axes_and_restores_shape():
    values = np.array(
        [
            [[np.nan, 1.0, np.nan, 4.0], [10.0, np.nan, 12.0, np.nan]],
            [[np.nan, np.nan, 22.0, np.nan], [30.0, np.nan, np.nan, 33.0]],
        ]
    )
    original = values.copy()

    actual = mojagg.ffill(values, axis=(0, 2))
    execution = np.moveaxis(values, (0, 2), (-2, -1))
    flattened = execution.reshape(execution.shape[:-2] + (-1,))
    expected_execution = np.empty_like(flattened)
    for outer, row in enumerate(flattened):
        expected_execution[outer] = row
        last = np.nan
        for position, value in enumerate(row):
            if np.isnan(value):
                if not np.isnan(last):
                    expected_execution[outer, position] = last
            else:
                last = value
    expected = np.moveaxis(
        expected_execution.reshape(execution.shape),
        (-2, -1),
        (0, 2),
    )

    np.testing.assert_equal(actual, expected)
    np.testing.assert_equal(values, original)
    assert actual.shape == values.shape
