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


def _matrix_func(arr: Any, is_corr: bool, func_name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim < 2:
        raise ValueError(
            f"{func_name} requires at least a 2D array with shape (..., vars, obs). "
            "For 1D arrays, use nanvar for variance calculations."
        )

    orig_dtype = a.dtype
    if orig_dtype not in (np.float32, np.float64):
        if not np.issubdtype(orig_dtype, np.number):
            raise TypeError(f"Unsupported dtype for matrix operation: {orig_dtype}")
        a = a.astype(np.float64)
        target_dtype = np.float64
    else:
        target_dtype = orig_dtype

    n_vars = a.shape[-2]
    n_obs = a.shape[-1]
    batch_shape = a.shape[:-2]
    batch = int(np.prod(batch_shape)) if batch_shape else 1

    out_shape = batch_shape + (n_vars, n_vars)
    out = np.empty(out_shape, dtype=target_dtype)

    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)

    cfg = get_config()
    workers = cfg.threads if cfg.threads > 0 else (os.cpu_count() or 4)
    threshold = cfg.parallel_threshold

    kernel_dict = _CORR_KERNELS if is_corr else _COV_KERNELS
    kernel = kernel_dict[target_dtype]
    kernel(a, out, batch, n_vars, n_obs, threshold, workers)

    return out


def nancovmatrix(a: Any) -> np.ndarray:
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
    return _matrix_func(a, is_corr=False, func_name="nancovmatrix")


def nancorrmatrix(a: Any) -> np.ndarray:
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
    return _matrix_func(a, is_corr=True, func_name="nancorrmatrix")
