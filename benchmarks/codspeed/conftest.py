"""Shared deterministic inputs for the small per-change CodSpeed suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def hot_path_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    values = rng.normal(1.0, 0.25, size=(16, 8_192)).astype(np.float64)
    values[:, ::97] = np.nan

    labels = np.arange(values.shape[1], dtype=np.int32) % 256

    matrix = rng.normal(1.0, 0.25, size=(16, 1_024)).astype(np.float64)
    matrix[:, ::113] = np.nan

    moving = rng.normal(1.0, 0.25, size=(8, 8_192)).astype(np.float64)
    moving[:, ::89] = np.nan

    return {
        "values": values,
        "labels": labels,
        "matrix": matrix,
        "moving": moving,
    }
