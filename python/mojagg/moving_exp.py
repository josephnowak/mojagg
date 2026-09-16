"""Numbagg-compatible exponentially weighted moving operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from mojagg import _native
from mojagg.config import get_config
from mojagg.moving import _normalize_axis

_MOVE_EXP_NANCOUNT_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nancount_f64,
    np.dtype(np.float32): _native.move_exp_nancount_f32,
}
_MOVE_EXP_NANMEAN_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nanmean_f64,
    np.dtype(np.float32): _native.move_exp_nanmean_f32,
}
_MOVE_EXP_NANSUM_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nansum_f64,
    np.dtype(np.float32): _native.move_exp_nansum_f32,
}
_MOVE_EXP_NANVAR_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nanvar_f64,
    np.dtype(np.float32): _native.move_exp_nanvar_f32,
}
_MOVE_EXP_NANSTD_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nanstd_f64,
    np.dtype(np.float32): _native.move_exp_nanstd_f32,
}
_MOVE_EXP_NANCOV_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nancov_f64,
    np.dtype(np.float32): _native.move_exp_nancov_f32,
}
_MOVE_EXP_NANCORR_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nancorr_f64,
    np.dtype(np.float32): _native.move_exp_nancorr_f32,
}

_SUPPORTED_MOVE_EXP_DTYPES = ", ".join(str(dtype) for dtype in _MOVE_EXP_NANSUM_KERNELS)


def _prepare_exp_array(a: Any, op: str) -> np.ndarray:
    """Prepare an exp-moving value array for one of the float kernels."""

    a_arr = np.asarray(a)
    if not a_arr.dtype.isnative:
        a_arr = a_arr.byteswap().view(a_arr.dtype.newbyteorder("="))
    if a_arr.dtype == np.dtype(np.float16):
        a_arr = a_arr.astype(np.float32)
    if a_arr.dtype in _MOVE_EXP_NANSUM_KERNELS:
        return a_arr
    if np.issubdtype(a_arr.dtype, np.integer) or np.issubdtype(a_arr.dtype, np.bool_):
        return a_arr.astype(np.float64)
    raise TypeError(
        f"{op} does not support dtype {a_arr.dtype}; supported: {_SUPPORTED_MOVE_EXP_DTYPES}"
    )


def _prepare_alpha(
    alpha: Any,
    shape: tuple[int, ...],
    axis: int,
    dtype: np.dtype,
    op: str,
) -> np.ndarray:
    """Make alpha a zero-copy view with the same physical rank as values."""

    alpha_arr = np.asarray(alpha)
    if not alpha_arr.dtype.isnative:
        alpha_arr = alpha_arr.byteswap().view(alpha_arr.dtype.newbyteorder("="))
    if alpha_arr.dtype == np.dtype(np.float16):
        alpha_arr = alpha_arr.astype(np.float32)
    if not np.issubdtype(alpha_arr.dtype, np.number) or np.issubdtype(
        alpha_arr.dtype, np.complexfloating
    ):
        raise TypeError(
            f"{op} alpha does not support dtype {alpha_arr.dtype}; "
            f"supported: {_SUPPORTED_MOVE_EXP_DTYPES}"
        )
    try:
        alpha_arr = alpha_arr.astype(dtype, copy=False)
    except (TypeError, ValueError):
        raise TypeError(
            f"{op} alpha does not support dtype {alpha_arr.dtype}; "
            f"supported: {_SUPPORTED_MOVE_EXP_DTYPES}"
        ) from None

    if alpha_arr.ndim == 0:
        return np.broadcast_to(alpha_arr, shape)
    if alpha_arr.ndim == 1:
        if alpha_arr.shape[0] != shape[axis]:
            raise ValueError(
                f"{op} alpha must have length {shape[axis]} along the moving axis; "
                f"got {alpha_arr.shape[0]}"
            )
        alpha_shape = [1] * len(shape)
        alpha_shape[axis] = alpha_arr.shape[0]
        return np.broadcast_to(alpha_arr.reshape(tuple(alpha_shape)), shape)
    try:
        return np.broadcast_to(alpha_arr, shape)
    except ValueError:
        raise ValueError(
            f"{op} alpha shape {alpha_arr.shape} cannot broadcast to {shape}"
        ) from None


def _min_weight(value: Any, op: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{op} min_weight must be a real number") from None


def _move_exp_unary(
    a: Any,
    alpha: Any,
    *,
    min_weight: Any,
    axis: Any,
    op: str,
    kernels: dict[np.dtype, Any],
) -> np.ndarray:
    values = np.asarray(a)
    normalized_axis = _normalize_axis(axis, values.ndim, op)
    if normalized_axis is None:
        return values

    values = _prepare_exp_array(values, op)
    alpha_arr = _prepare_alpha(
        alpha,
        values.shape,
        normalized_axis,
        values.dtype,
        op,
    )
    result = kernels[values.dtype](
        values,
        alpha_arr,
        (normalized_axis,),
        _min_weight(min_weight, op),
        get_config(),
    )
    return np.moveaxis(result, -1, normalized_axis)


def _move_exp_binary(
    a1: Any,
    a2: Any,
    alpha: Any,
    *,
    min_weight: Any,
    axis: Any,
    op: str,
    kernels: dict[np.dtype, Any],
) -> np.ndarray:
    values1 = np.asarray(a1)
    values2 = np.asarray(a2)
    normalized_axis = _normalize_axis(axis, values1.ndim, op, n_arrays=2)
    if normalized_axis is None:
        raise ValueError(
            "`axis` cannot be an empty tuple when passing more than one array; "
            "since we default to returning the input."
        )

    values1 = _prepare_exp_array(values1, op)
    values2 = _prepare_exp_array(values2, op)
    if values1.dtype != values2.dtype:
        raise TypeError(
            f"{op} requires inputs with the same dtype; got {values1.dtype} and {values2.dtype}"
        )

    requested_axis = axis[0] if isinstance(axis, tuple) else axis
    axis1 = normalized_axis
    axis2 = int(np.lib.array_utils.normalize_axis_index(requested_axis, values2.ndim))
    if values1.shape[axis1] != values2.shape[axis2]:
        raise ValueError(
            f"{op} inputs must have the same length along the moving axis; "
            f"got {values1.shape[axis1]} and {values2.shape[axis2]}"
        )

    values1_ndim = values1.ndim
    values1, values2 = np.broadcast_arrays(values1, values2)
    common_axis = values1.ndim - values1_ndim + normalized_axis
    alpha_arr = _prepare_alpha(
        alpha,
        values1.shape,
        common_axis,
        values1.dtype,
        op,
    )
    result = kernels[values1.dtype](
        values1,
        values2,
        alpha_arr,
        (common_axis,),
        _min_weight(min_weight, op),
        get_config(),
    )
    return np.moveaxis(result, -1, common_axis)


def move_exp_nancount(
    a: Any,
    /,
    *,
    alpha: Any,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving count of non-NaN values."""

    return _move_exp_unary(
        a,
        alpha,
        min_weight=min_weight,
        axis=axis,
        op="move_exp_nancount",
        kernels=_MOVE_EXP_NANCOUNT_KERNELS,
    )


def move_exp_nanmean(
    a: Any,
    /,
    *,
    alpha: Any,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving mean ignoring NaN values."""

    return _move_exp_unary(
        a,
        alpha,
        min_weight=min_weight,
        axis=axis,
        op="move_exp_nanmean",
        kernels=_MOVE_EXP_NANMEAN_KERNELS,
    )


def move_exp_nansum(
    a: Any,
    /,
    *,
    alpha: Any,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving sum ignoring NaN values."""

    return _move_exp_unary(
        a,
        alpha,
        min_weight=min_weight,
        axis=axis,
        op="move_exp_nansum",
        kernels=_MOVE_EXP_NANSUM_KERNELS,
    )


def move_exp_nanvar(
    a: Any,
    /,
    *,
    alpha: Any,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving sample variance ignoring NaNs."""

    return _move_exp_unary(
        a,
        alpha,
        min_weight=min_weight,
        axis=axis,
        op="move_exp_nanvar",
        kernels=_MOVE_EXP_NANVAR_KERNELS,
    )


def move_exp_nanstd(
    a: Any,
    /,
    *,
    alpha: Any,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving sample standard deviation."""

    return _move_exp_unary(
        a,
        alpha,
        min_weight=min_weight,
        axis=axis,
        op="move_exp_nanstd",
        kernels=_MOVE_EXP_NANSTD_KERNELS,
    )


def move_exp_nancov(
    a1: Any,
    a2: Any,
    /,
    *,
    alpha: Any,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving sample covariance."""

    return _move_exp_binary(
        a1,
        a2,
        alpha,
        min_weight=min_weight,
        axis=axis,
        op="move_exp_nancov",
        kernels=_MOVE_EXP_NANCOV_KERNELS,
    )


def move_exp_nancorr(
    a1: Any,
    a2: Any,
    /,
    *,
    alpha: Any,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving correlation."""

    return _move_exp_binary(
        a1,
        a2,
        alpha,
        min_weight=min_weight,
        axis=axis,
        op="move_exp_nancorr",
        kernels=_MOVE_EXP_NANCORR_KERNELS,
    )


__all__ = [
    "move_exp_nancorr",
    "move_exp_nancount",
    "move_exp_nancov",
    "move_exp_nanmean",
    "move_exp_nanstd",
    "move_exp_nansum",
    "move_exp_nanvar",
]
