"""Coverage of the fill and quantile-selection kernels.

Filling along the strided axis is the layout that forces the driver onto its
worker-local scratch path, so both layouts are tracked. nanquantile and
nanmedian share the non-streaming selection kernel, which has a different cost
profile from the reductions.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.codspeed.conftest import FILL_CASES, REDUCTION_CASES

import mojagg

QUANTILES = np.array([0.1, 0.5, 0.9])


@pytest.mark.parametrize("op", ("bfill", "ffill"))
@pytest.mark.parametrize("case", FILL_CASES)
def test_fill(benchmark, fill_cases, op, case):
    values, axis = fill_cases[case]
    benchmark(getattr(mojagg, op), values, None, axis)


@pytest.mark.parametrize("case", REDUCTION_CASES)
def test_nanmedian(benchmark, reduction_cases, case):
    values, axis = reduction_cases[case]
    benchmark(mojagg.nanmedian, values, axis=axis)


@pytest.mark.parametrize("case", REDUCTION_CASES)
def test_nanquantile(benchmark, reduction_cases, case):
    values, axis = reduction_cases[case]
    benchmark(mojagg.nanquantile, values, QUANTILES, axis=axis)
