"""Per-operation coverage of the NaN-aware reductions.

One benchmark per (operation, layout) so a regression points at a single
kernel instead of the whole family. Dtype variants are kept to the contiguous
layout: they exist to cover the distinct kernel instantiations, not to widen
the matrix.
"""

from __future__ import annotations

import pytest
from benchmarks.codspeed.conftest import REDUCTION_CASES

import mojagg

# Reductions whose kernels are instantiated for integers as well as floats.
INT_SUPPORTING_OPS = (
    "allnan",
    "anynan",
    "nanargmax",
    "nanargmin",
    "nancount",
    "nanmax",
    "nanmin",
    "nanprod",
    "nansum",
)

# Reductions that always accumulate in floating point.
FLOAT_ONLY_OPS = ("nanmean", "nanstd", "nanvar")

ALL_OPS = tuple(sorted(INT_SUPPORTING_OPS + FLOAT_ONLY_OPS))


@pytest.mark.parametrize("op", ALL_OPS)
@pytest.mark.parametrize("case", REDUCTION_CASES)
def test_reduction(benchmark, reduction_cases, op, case):
    values, axis = reduction_cases[case]
    benchmark(getattr(mojagg, op), values, axis=axis)


@pytest.mark.parametrize("op", ALL_OPS)
def test_reduction_float32(benchmark, float32_case, op):
    values, axis = float32_case
    benchmark(getattr(mojagg, op), values, axis=axis)


@pytest.mark.parametrize("op", INT_SUPPORTING_OPS)
def test_reduction_int64(benchmark, int64_case, op):
    values, axis = int64_case
    benchmark(getattr(mojagg, op), values, axis=axis)
