"""Per-operation coverage of the grouped reductions.

Both cases run the same values through the scatter path at a cardinality that
fits in cache and at one that does not, and they also cover the int32 and
int64 label kernels.
"""

from __future__ import annotations

import pytest
from benchmarks.codspeed.conftest import GROUP_CASES

import mojagg

GROUP_OPS = (
    "group_nanall",
    "group_nanany",
    "group_nanargmax",
    "group_nanargmin",
    "group_nancount",
    "group_nanfirst",
    "group_nanlast",
    "group_nanmax",
    "group_nanmean",
    "group_nanmin",
    "group_nanprod",
    "group_nanstd",
    "group_nansum",
    "group_nansum_of_squares",
    "group_nanvar",
)


@pytest.mark.parametrize("op", GROUP_OPS)
@pytest.mark.parametrize("case", GROUP_CASES)
def test_grouped(benchmark, group_cases, op, case):
    values, labels, num_labels = group_cases[case]
    benchmark(
        getattr(mojagg, op),
        values,
        labels,
        axis=1,
        num_labels=num_labels,
    )
