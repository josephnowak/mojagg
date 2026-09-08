"""CodSpeed benchmarks: nanfuncs vs numpy vs numbagg.

Every benchmark is tagged by op + case so CodSpeed tracks per-op regressions
across PRs. numpy reference only where a direct equivalent exists.
"""

from __future__ import annotations

import numbagg
import numpy as np
import pytest

import mojagg


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nansum_mojagg(cases, case_idx):
    _, a, axis = cases[case_idx]
    mojagg.nansum(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nansum_numbagg(cases, case_idx):
    _, a, axis = cases[case_idx]
    numbagg.nansum(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nansum_numpy(cases, case_idx):
    _, a, axis = cases[case_idx]
    np.nansum(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nanmean_mojagg(mean_cases, case_idx):
    _, a, axis = mean_cases[case_idx]
    mojagg.nanmean(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nanmean_numbagg(mean_cases, case_idx):
    _, a, axis = mean_cases[case_idx]
    numbagg.nanmean(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nanmean_numpy(mean_cases, case_idx):
    _, a, axis = mean_cases[case_idx]
    np.nanmean(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_allnan_mojagg(cases, case_idx):
    _, a, axis = cases[case_idx]
    mojagg.allnan(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_allnan_numbagg(cases, case_idx):
    _, a, axis = cases[case_idx]
    numbagg.allnan(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_anynan_mojagg(cases, case_idx):
    _, a, axis = cases[case_idx]
    mojagg.anynan(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_anynan_numbagg(cases, case_idx):
    _, a, axis = cases[case_idx]
    numbagg.anynan(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nanprod_mojagg(cases, case_idx):
    _, a, axis = cases[case_idx]
    mojagg.nanprod(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_nanprod_numpy(cases, case_idx):
    _, a, axis = cases[case_idx]
    np.nanprod(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("op", ["nanmin", "nanmax", "nanargmin", "nanargmax", "nanvar", "nanstd"])
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_remaining_mojagg(mean_cases, op, case_idx):
    _, a, axis = mean_cases[case_idx]
    getattr(mojagg, op)(a, axis=axis)


@pytest.mark.benchmark
@pytest.mark.parametrize("op", ["nanmin", "nanmax", "nanargmin", "nanargmax", "nanvar", "nanstd"])
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_bench_remaining_numbagg(mean_cases, op, case_idx):
    _, a, axis = mean_cases[case_idx]
    getattr(numbagg, op)(a, axis=axis)
