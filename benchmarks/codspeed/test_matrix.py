"""Coverage of the pairwise matrix kernels.

These are O(vars^2 * obs), so a single deliberately small input per operation
is enough to detect a regression.
"""

from __future__ import annotations

import pytest

import mojagg

ALPHA = 0.15
WINDOW = 64
MIN_COUNT = 32


@pytest.mark.parametrize("op", ("nancorrmatrix", "nancovmatrix"))
def test_matrix(benchmark, matrix_values, op):
    benchmark(getattr(mojagg, op), matrix_values, axis=(0, 1))


@pytest.mark.parametrize("op", ("move_corrmatrix", "move_covmatrix"))
def test_move_matrix(benchmark, moving_matrix_values, op):
    benchmark(getattr(mojagg, op), moving_matrix_values, WINDOW, MIN_COUNT)


@pytest.mark.parametrize("op", ("move_exp_nancorrmatrix", "move_exp_nancovmatrix"))
def test_move_exp_matrix(benchmark, moving_matrix_values, op):
    benchmark(getattr(mojagg, op), moving_matrix_values, ALPHA)
