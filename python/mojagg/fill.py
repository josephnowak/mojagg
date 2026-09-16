"""Forward and backward fill functions matching numbagg semantics."""

from __future__ import annotations

from operator import index
from typing import Any

import numpy as np

from mojagg import _native
from mojagg.config import get_config

_FFILL_KERNELS = {
    np.dtype(np.float64): _native.ffill_f64,
    np.dtype(np.float32): _native.ffill_f32,
    np.dtype(np.int64): _native.ffill_i64,
    np.dtype(np.int32): _native.ffill_i32,
}

_BFILL_KERNELS = {
    np.dtype(np.float64): _native.bfill_f64,
    np.dtype(np.float32): _native.bfill_f32,
    np.dtype(np.int64): _native.bfill_i64,
    np.dtype(np.int32): _native.bfill_i32,
}


def _normalize_fill_axes(axis, ndim: int) -> tuple[int, ...]:
    if isinstance(axis, tuple):
        axes = np.lib.array_utils.normalize_axis_tuple(axis, ndim)
        if not axes:
            raise ValueError("fill requires at least one axis")
        return axes
    return (np.lib.array_utils.normalize_axis_index(axis, ndim),)


def _fill(
    arr: Any,
    limit: int | None = None,
    axis: int | tuple[int, ...] = -1,
    is_backward: bool = False,
) -> np.ndarray:
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.number):
        raise TypeError(f"Unsupported dtype for fill operation: {a.dtype}")

    axes = _normalize_fill_axes(axis, a.ndim)

    if limit is None:
        limit = int(np.prod([a.shape[axis] for axis in axes], dtype=np.int64))
    else:
        try:
            limit = index(limit)
        except TypeError:
            raise TypeError("`limit` must be an integer or None") from None
        if limit < 0:
            raise ValueError(f"`limit` must be positive: {limit}")

    if a.size == 0 or any(a.shape[axis] == 0 for axis in axes):
        return a.copy()

    # Integers cannot contain NaNs. Preserve numbagg's fresh-output behavior.
    if np.issubdtype(a.dtype, np.integer):
        return a.copy()
    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(f"Unsupported dtype for fill operation: {a.dtype}")

    orig_dtype = a.dtype
    if orig_dtype not in (np.float32, np.float64):
        target_dtype = np.float32 if orig_dtype.itemsize <= 4 else np.float64
        a = a.astype(target_dtype)
    else:
        target_dtype = orig_dtype

    kernel_dict = _BFILL_KERNELS if is_backward else _FFILL_KERNELS
    kernel = kernel_dict[target_dtype]

    if not a.dtype.isnative:
        a = a.byteswap().view(a.dtype.newbyteorder("="))

    cfg = get_config()
    out_execution = kernel(a, axes, limit, cfg)
    output_axes = tuple(range(out_execution.ndim - len(axes), out_execution.ndim))
    out = np.moveaxis(out_execution, output_axes, axes)

    if out.dtype != orig_dtype:
        out = out.astype(orig_dtype)
    return out


def ffill(
    a: Any,
    limit: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs,
) -> np.ndarray:
    """Forward fill missing values.

    Parameters
    ----------
    a : array_like
        Input array.
    limit : int, optional
        Maximum number of consecutive NaN values to forward fill.
        Defaults to the length of the axis.
    axis : int, default -1
        Axis along which to fill missing values.

    Returns
    -------
    ndarray
        Array with NaN values forward-filled.
    """
    return _fill(a, limit=limit, axis=axis, is_backward=False)


def bfill(
    a: Any,
    limit: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs,
) -> np.ndarray:
    """Backward fill missing values.

    Parameters
    ----------
    a : array_like
        Input array.
    limit : int, optional
        Maximum number of consecutive NaN values to backward fill.
        Defaults to the length of the axis.
    axis : int, default -1
        Axis along which to fill missing values.

    Returns
    -------
    ndarray
        Array with NaN values backward-filled.
    """
    return _fill(a, limit=limit, axis=axis, is_backward=True)
