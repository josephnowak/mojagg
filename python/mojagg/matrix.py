"""Covariance and correlation matrix functions matching numbagg semantics."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from mojagg import _native
from mojagg.config import get_config

_COV_KERNELS = {
    np.dtype(np.float64): _native.nancovmatrix_f64,
    np.dtype(np.float32): _native.nancovmatrix_f32,
}

_CORR_KERNELS = {
    np.dtype(np.float64): _native.nancorrmatrix_f64,
    np.dtype(np.float32): _native.nancorrmatrix_f32,
}


def _normalize_matrix_axes(axis, ndim: int, func_name: str) -> tuple[int, int]:
    if axis is None:
        return (ndim - 2, ndim - 1)
    if not isinstance(axis, tuple):
        raise TypeError(f"{func_name} axis must be a tuple of two axes")
    axes = np.lib.array_utils.normalize_axis_tuple(axis, ndim)
    if len(axes) != 2:
        raise ValueError(f"{func_name} requires exactly two axes")
    return axes


def _matrix_func(
    arr: Any,
    is_corr: bool,
    func_name: str,
    axis=None,
) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim < 2:
        raise ValueError(
            f"{func_name} requires at least a 2D array with shape (..., vars, obs). "
            "For 1D arrays, use nanvar for variance calculations."
        )

    orig_dtype = a.dtype
    if orig_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        if not np.issubdtype(orig_dtype, np.number):
            raise TypeError(f"Unsupported dtype for matrix operation: {orig_dtype}")
        if np.can_cast(orig_dtype, np.dtype(np.float32), casting="safe"):
            target_dtype = np.dtype(np.float32)
        elif np.can_cast(orig_dtype, np.dtype(np.float64), casting="same_kind"):
            target_dtype = np.dtype(np.float64)
        else:
            raise TypeError(f"Unsupported dtype for matrix operation: {orig_dtype}")
        a = a.astype(target_dtype)
    else:
        target_dtype = orig_dtype

    axes = _normalize_matrix_axes(axis, a.ndim, func_name)

    cfg = get_config()
    workers = cfg.threads if cfg.threads > 0 else (os.cpu_count() or 4)
    threshold = cfg.parallel_threshold
    min_groups = cfg.parallel_min_groups

    kernel_dict = _CORR_KERNELS if is_corr else _COV_KERNELS
    kernel = kernel_dict[target_dtype]
    return kernel(a, axes, threshold, workers, min_groups)


def nancovmatrix(a: Any, axis=None) -> np.ndarray:
    """Compute covariance matrix treating NaN as missing values.

    Parameters
    ----------
    a : array_like
        Input array with shape (..., vars, obs).

    Returns
    -------
    ndarray
        Square covariance matrix with shape (..., vars, vars).
    """
    return _matrix_func(
        a,
        is_corr=False,
        func_name="nancovmatrix",
        axis=axis,
    )


def nancorrmatrix(a: Any, axis=None) -> np.ndarray:
    """Compute correlation matrix treating NaN as missing values.

    Parameters
    ----------
    a : array_like
        Input array with shape (..., vars, obs).

    Returns
    -------
    ndarray
        Square correlation matrix with shape (..., vars, vars).
    """
    return _matrix_func(
        a,
        is_corr=True,
        func_name="nancorrmatrix",
        axis=axis,
    )
