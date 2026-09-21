"""Numbagg-compatible moving-window operations."""

from __future__ import annotations

from operator import index
from typing import Any

import numpy as np

from mojagg import _native
from mojagg.config import get_config
from mojagg.moving_matrix import move_corrmatrix, move_covmatrix

_MOVE_CORR_KERNELS = {
    np.dtype(np.float64): _native.move_corr_f64,
    np.dtype(np.float32): _native.move_corr_f32,
}
_MOVE_COV_KERNELS = {
    np.dtype(np.float64): _native.move_cov_f64,
    np.dtype(np.float32): _native.move_cov_f32,
}
_MOVE_MEAN_KERNELS = {
    np.dtype(np.float64): _native.move_mean_f64,
    np.dtype(np.float32): _native.move_mean_f32,
}
_MOVE_STD_KERNELS = {
    np.dtype(np.float64): _native.move_std_f64,
    np.dtype(np.float32): _native.move_std_f32,
}
_MOVE_SUM_KERNELS = {
    np.dtype(np.float64): _native.move_sum_f64,
    np.dtype(np.float32): _native.move_sum_f32,
}
_MOVE_VAR_KERNELS = {
    np.dtype(np.float64): _native.move_var_f64,
    np.dtype(np.float32): _native.move_var_f32,
}

_SUPPORTED_MOVE_DTYPES = ", ".join(str(dtype) for dtype in _MOVE_SUM_KERNELS)


def _normalize_axis(
    axis: Any,
    ndim: int,
    op: str,
    n_arrays: int = 1,
) -> int | None:
    if isinstance(axis, tuple):
        if axis == ():
            if n_arrays > 1:
                raise ValueError(
                    "`axis` cannot be an empty tuple when passing more than one array; "
                    "since we default to returning the input."
                )
            return None
        if len(axis) != 1:
            raise ValueError(f"only one axis can be passed to {op}; got {axis}")
        axis = axis[0]
    if axis is None:
        raise TypeError(f"{op} axis must be an integer")
    return int(np.lib.array_utils.normalize_axis_index(axis, ndim))


def _parse_window_min_count(
    window: Any,
    min_count: Any,
) -> tuple[int, int]:
    window_int = index(window)
    if min_count is None:
        min_count_int = window_int
    else:
        min_count_int = index(min_count)
        if min_count_int < 0:
            raise ValueError(f"min_count must be positive: {min_count_int}")
    return window_int, min_count_int


def _prepare_move_array(a: Any, op: str) -> np.ndarray:
    a_arr = np.asarray(a)
    if not a_arr.dtype.isnative:
        a_arr = a_arr.byteswap().view(a_arr.dtype.newbyteorder("="))
    if a_arr.dtype == np.dtype(np.float16):
        a_arr = a_arr.astype(np.float32)
    if np.issubdtype(a_arr.dtype, np.integer) or a_arr.dtype == np.dtype(np.bool_):
        a_arr = a_arr.astype(np.float64)
    if a_arr.dtype not in _MOVE_SUM_KERNELS:
        raise TypeError(
            f"{op} does not support dtype {a_arr.dtype}; supported: {_SUPPORTED_MOVE_DTYPES}"
        )
    return a_arr


def _validate_window(a_arr: np.ndarray, window: int, axis: int) -> None:
    if not 0 < window <= a_arr.shape[axis]:
        raise ValueError(f"window not in valid range: {window}")


def _move_unary(
    a: Any,
    *,
    window: Any,
    min_count: Any,
    axis: Any,
    op: str,
    kernels: dict[np.dtype, Any],
) -> np.ndarray:
    a_arr = np.asarray(a)
    window_int, min_count_int = _parse_window_min_count(window, min_count)
    normalized_axis = _normalize_axis(axis, a_arr.ndim, op)
    if normalized_axis is None:
        return a_arr
    _validate_window(a_arr, window_int, normalized_axis)

    a_arr = _prepare_move_array(a_arr, op)
    if op == "move_mean" and window_int == 1:
        if min_count_int <= 1:
            return a_arr.copy()
        return np.full_like(a_arr, np.nan, dtype=a_arr.dtype)
    moved = np.moveaxis(a_arr, normalized_axis, -1)
    core_axis = moved.ndim - 1
    out_moved = kernels[a_arr.dtype](
        moved,
        (core_axis,),
        window_int,
        min_count_int,
        get_config(),
    )
    return np.moveaxis(out_moved, -1, normalized_axis)


def _move_binary(
    a: Any,
    b: Any,
    *,
    window: Any,
    min_count: Any,
    axis: Any,
    op: str,
    kernels: dict[np.dtype, Any],
) -> np.ndarray:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    window_int, min_count_int = _parse_window_min_count(window, min_count)
    normalized_axis = _normalize_axis(axis, a_arr.ndim, op, n_arrays=2)
    if normalized_axis is None:
        return a_arr
    _validate_window(a_arr, window_int, normalized_axis)

    a_arr = _prepare_move_array(a_arr, op)
    b_arr = _prepare_move_array(b_arr, op)
    if a_arr.dtype != b_arr.dtype:
        raise TypeError(
            f"{op} requires inputs with the same dtype; got {a_arr.dtype} and {b_arr.dtype}"
        )

    requested_axis = axis[0] if isinstance(axis, tuple) else axis
    b_axis = requested_axis if requested_axis < 0 else normalized_axis
    moved_a = np.moveaxis(a_arr, normalized_axis, -1)
    moved_b = np.moveaxis(b_arr, b_axis, -1)
    moved_a, moved_b = np.broadcast_arrays(moved_a, moved_b)
    if moved_a.shape[-1] != moved_b.shape[-1]:
        raise ValueError(
            f"{op} inputs must have the same length along axis {normalized_axis}; "
            f"got {moved_a.shape[-1]} and {moved_b.shape[-1]}"
        )
    core_axis = moved_a.ndim - 1
    out_moved = kernels[a_arr.dtype](
        moved_a,
        moved_b,
        (core_axis,),
        window_int,
        min_count_int,
        get_config(),
    )
    output_axis = out_moved.ndim - a_arr.ndim + normalized_axis
    return np.moveaxis(out_moved, -1, output_axis)


def move_sum(
    a: Any,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing NaN-aware moving sum.

    The result has the same shape and floating dtype as the input, except that
    float16 inputs promote to float32. Partial windows are emitted during
    warm-up; ``min_count`` controls when a value is valid and defaults to
    ``window``.
    """

    return _move_unary(
        a,
        window=window,
        min_count=min_count,
        axis=axis,
        op="move_sum",
        kernels=_MOVE_SUM_KERNELS,
    )


def move_mean(
    a: Any,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing NaN-aware moving mean."""

    return _move_unary(
        a,
        window=window,
        min_count=min_count,
        axis=axis,
        op="move_mean",
        kernels=_MOVE_MEAN_KERNELS,
    )


def move_var(
    a: Any,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing NaN-aware moving sample variance."""

    return _move_unary(
        a,
        window=window,
        min_count=min_count,
        axis=axis,
        op="move_var",
        kernels=_MOVE_VAR_KERNELS,
    )


def move_std(
    a: Any,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing NaN-aware moving sample standard deviation."""

    return _move_unary(
        a,
        window=window,
        min_count=min_count,
        axis=axis,
        op="move_std",
        kernels=_MOVE_STD_KERNELS,
    )


def move_cov(
    a: Any,
    b: Any,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing pairwise NaN-aware moving sample covariance."""

    return _move_binary(
        a,
        b,
        window=window,
        min_count=min_count,
        axis=axis,
        op="move_cov",
        kernels=_MOVE_COV_KERNELS,
    )


def move_corr(
    a: Any,
    b: Any,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing pairwise NaN-aware moving correlation."""

    return _move_binary(
        a,
        b,
        window=window,
        min_count=min_count,
        axis=axis,
        op="move_corr",
        kernels=_MOVE_CORR_KERNELS,
    )


__all__ = [
    "move_mean",
    "move_sum",
    "move_std",
    "move_var",
    "move_cov",
    "move_corr",
    "move_corrmatrix",
    "move_covmatrix",
]
