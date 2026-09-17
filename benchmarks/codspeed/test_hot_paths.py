"""Representative native hot paths for CodSpeed regression tracking.

These inputs are deliberately moderate and fixed. The publication benchmark
owns the broad size/dtype/NaN/cardinality matrix and is never imported here.
"""

from __future__ import annotations

import mojagg


def test_nansum(benchmark, hot_path_inputs):
    values = hot_path_inputs["values"]
    benchmark(mojagg.nansum, values, axis=1)


def test_group_nansum(benchmark, hot_path_inputs):
    values = hot_path_inputs["values"]
    labels = hot_path_inputs["labels"]
    benchmark(mojagg.group_nansum, values, labels, axis=1, num_labels=256)


def test_nancorrmatrix(benchmark, hot_path_inputs):
    values = hot_path_inputs["matrix"]
    benchmark(mojagg.nancorrmatrix, values, axis=(0, 1))


def test_move_mean(benchmark, hot_path_inputs):
    values = hot_path_inputs["moving"]
    benchmark(mojagg.move_mean, values, window=64, min_count=32, axis=1)


def test_move_exp_nanmean(benchmark, hot_path_inputs):
    values = hot_path_inputs["moving"]
    benchmark(mojagg.move_exp_nanmean, values, alpha=0.15, axis=1)


def test_ffill(benchmark, hot_path_inputs):
    values = hot_path_inputs["moving"]
    benchmark(mojagg.ffill, values, limit=64, axis=1)
