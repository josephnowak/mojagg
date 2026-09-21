"""Per-operation coverage of the trailing and exponential window kernels.

Trailing windows are measured with a short and a long window because the two
take different amounts of work per emitted value once NaNs enter and leave the
window. The exponentially weighted kernels have no window, so they run on a
single deterministic alpha.
"""

from __future__ import annotations

import pytest
from benchmarks.codspeed.conftest import MOVING_CASES, MOVING_WINDOWS

import mojagg

MOVE_UNARY_OPS = ("move_mean", "move_std", "move_sum", "move_var")
MOVE_BINARY_OPS = ("move_corr", "move_cov")

MOVE_EXP_UNARY_OPS = (
    "move_exp_nancount",
    "move_exp_nanmean",
    "move_exp_nanstd",
    "move_exp_nansum",
    "move_exp_nanvar",
)
MOVE_EXP_BINARY_OPS = ("move_exp_nancorr", "move_exp_nancov")

ALPHA = 0.15
PAIRWISE_WINDOW = 64
PAIRWISE_MIN_COUNT = 32


@pytest.mark.parametrize("op", MOVE_UNARY_OPS)
@pytest.mark.parametrize("case", MOVING_CASES)
def test_move_unary(benchmark, moving_pair, op, case):
    values, _ = moving_pair
    window, min_count = MOVING_WINDOWS[case]
    benchmark(getattr(mojagg, op), values, window=window, min_count=min_count, axis=1)


@pytest.mark.parametrize("op", MOVE_BINARY_OPS)
def test_move_binary(benchmark, moving_pair, op):
    first, second = moving_pair
    benchmark(
        getattr(mojagg, op),
        first,
        second,
        window=PAIRWISE_WINDOW,
        min_count=PAIRWISE_MIN_COUNT,
        axis=1,
    )


@pytest.mark.parametrize("op", MOVE_EXP_UNARY_OPS)
def test_move_exp_unary(benchmark, moving_pair, op):
    values, _ = moving_pair
    benchmark(getattr(mojagg, op), values, alpha=ALPHA, axis=1)


@pytest.mark.parametrize("op", MOVE_EXP_BINARY_OPS)
def test_move_exp_binary(benchmark, moving_pair, op):
    first, second = moving_pair
    benchmark(getattr(mojagg, op), first, second, alpha=ALPHA, axis=1)
