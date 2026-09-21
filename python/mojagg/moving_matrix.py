"""Moving covariance and correlation matrices matching numbagg semantics."""

from __future__ import annotations

from operator import index
from typing import Any

import numpy as np

from mojagg import _native
from mojagg.config import get_config
from mojagg.matrix import nancorrmatrix, nancovmatrix

_MOVE_CORR_MATRIX_KERNELS = {
    np.dtype(np.float64): _native.move_corrmatrix_f64,
    np.dtype(np.float32): _native.move_corrmatrix_f32,
}
_MOVE_COV_MATRIX_KERNELS = {
    np.dtype(np.float64): _native.move_covmatrix_f64,
    np.dtype(np.float32): _native.move_covmatrix_f32,
}
_MOVE_EXP_NANCORR_MATRIX_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nancorrmatrix_f64,
    np.dtype(np.float32): _native.move_exp_nancorrmatrix_f32,
}
_MOVE_EXP_NANCOV_MATRIX_KERNELS = {
    np.dtype(np.float64): _native.move_exp_nancovmatrix_f64,
    np.dtype(np.float32): _native.move_exp_nancovmatrix_f32,
}

_SUPPORTED_MOVE_MATRIX_DTYPES = ", ".join(str(dtype) for dtype in _MOVE_COV_MATRIX_KERNELS)


def _prepare_matrix_array(a: Any, op: str) -> np.ndarray:
    """Prepare a matrix input for a float kernel."""

    a_arr = np.asarray(a)
    if not a_arr.dtype.isnative:
        a_arr = a_arr.byteswap().view(a_arr.dtype.newbyteorder("="))
    if a_arr.dtype == np.dtype(np.float16):
        a_arr = a_arr.astype(np.float32)
    if a_arr.dtype in _MOVE_COV_MATRIX_KERNELS:
        return a_arr
    if np.issubdtype(a_arr.dtype, np.integer) or np.issubdtype(a_arr.dtype, np.bool_):
        return a_arr.astype(np.float64)
    raise TypeError(
        f"{op} does not support dtype {a_arr.dtype}; supported: {_SUPPORTED_MOVE_MATRIX_DTYPES}"
    )


def _matrix_shape_error(a: np.ndarray, op: str) -> None:
    if a.ndim < 2:
        raise ValueError(f"{op} requires at least a 2D array with shape (..., obs, vars).")


def _parse_window_min_count(window: Any, min_count: Any) -> tuple[int, int]:
    window_int = index(window)
    if min_count is None:
        min_count_int = window_int
    else:
        min_count_int = index(min_count)
        if min_count_int < 0:
            raise ValueError(f"min_count must be positive: {min_count_int}")
    return window_int, min_count_int


def _validate_window(a: np.ndarray, window: int, op: str) -> None:
    if not 0 < window <= a.shape[-2]:
        raise ValueError(f"window not in valid range: {window}")


def _prepare_matrix_alpha(
    alpha: Any,
    shape: tuple[int, ...],
    dtype: np.dtype,
    op: str,
) -> np.ndarray:
    """Broadcast observation weights to the full ``(obs, vars)`` core.

    The native signature uses the same two dimensional core for values and
    weights.  Repeating an observation weight along ``vars`` preserves the
    upstream ``(m,n),(m),()->(m,n,n)`` gufunc contract while leaving outer
    broadcasting to the gufunc planner.
    """

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
            f"supported: {_SUPPORTED_MOVE_MATRIX_DTYPES}"
        )
    alpha_arr = alpha_arr.astype(dtype, copy=False)

    batch_shape = shape[:-2]
    observation_shape = batch_shape + (shape[-2],)
    if alpha_arr.ndim == 0:
        observation_alpha = np.broadcast_to(alpha_arr, observation_shape)
    elif alpha_arr.ndim == 1:
        if alpha_arr.shape[0] != shape[-2]:
            raise ValueError(
                f"{op} alpha must have length {shape[-2]} along the observation axis; "
                f"got {alpha_arr.shape[0]}"
            )
        alpha_shape = (1,) * len(batch_shape) + (shape[-2],)
        observation_alpha = np.broadcast_to(alpha_arr.reshape(alpha_shape), observation_shape)
    elif alpha_arr.ndim == len(shape) - 1:
        try:
            observation_alpha = np.broadcast_to(alpha_arr, observation_shape)
        except ValueError:
            raise ValueError(
                f"{op} alpha shape {alpha_arr.shape} cannot broadcast to {observation_shape}"
            ) from None
    else:
        try:
            return np.broadcast_to(alpha_arr, shape)
        except ValueError:
            raise ValueError(
                f"{op} alpha shape {alpha_arr.shape} cannot broadcast to {shape}"
            ) from None

    return np.broadcast_to(observation_alpha[..., None], shape)


def _move_matrix(
    a: Any,
    *,
    window: Any,
    min_count: Any,
    op: str,
    kernels: dict[np.dtype, Any],
) -> np.ndarray:
    values = np.asarray(a)
    _matrix_shape_error(values, op)
    window_int, min_count_int = _parse_window_min_count(window, min_count)
    _validate_window(values, window_int, op)
    values = _prepare_matrix_array(values, op)
    axes = (values.ndim - 2, values.ndim - 1)
    result = kernels[values.dtype](
        values,
        axes,
        window_int,
        min_count_int,
        get_config(),
    )
    if window_int == values.shape[-2] and min_count_int == window_int:
        reference_input = np.swapaxes(values, -1, -2)
        if op == "move_corrmatrix":
            result[..., -1, :, :] = nancorrmatrix(reference_input)
        elif op == "move_covmatrix":
            result[..., -1, :, :] = nancovmatrix(reference_input)
    return result


def _move_exp_matrix(
    a: Any,
    alpha: Any,
    *,
    min_weight: Any,
    op: str,
    kernels: dict[np.dtype, Any],
) -> np.ndarray:
    values = np.asarray(a)
    _matrix_shape_error(values, op)
    values = _prepare_matrix_array(values, op)
    alpha_arr = _prepare_matrix_alpha(alpha, values.shape, values.dtype, op)
    try:
        min_weight_float = float(min_weight)
    except (TypeError, ValueError):
        raise TypeError(f"{op} min_weight must be a real number") from None
    axes = (values.ndim - 2, values.ndim - 1)
    return kernels[values.dtype](
        values,
        alpha_arr,
        axes,
        min_weight_float,
        get_config(),
    )


def move_corrmatrix(
    a: Any,
    window: int,
    min_count: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing NaN-aware moving correlation matrices."""

    return _move_matrix(
        a,
        window=window,
        min_count=min_count,
        op="move_corrmatrix",
        kernels=_MOVE_CORR_MATRIX_KERNELS,
    )


def move_covmatrix(
    a: Any,
    window: int,
    min_count: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Trailing NaN-aware moving covariance matrices."""

    return _move_matrix(
        a,
        window=window,
        min_count=min_count,
        op="move_covmatrix",
        kernels=_MOVE_COV_MATRIX_KERNELS,
    )


def move_exp_nancorrmatrix(
    a: Any,
    alpha: Any,
    min_weight: float = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving correlation matrices."""

    return _move_exp_matrix(
        a,
        alpha,
        min_weight=min_weight,
        op="move_exp_nancorrmatrix",
        kernels=_MOVE_EXP_NANCORR_MATRIX_KERNELS,
    )


def move_exp_nancovmatrix(
    a: Any,
    alpha: Any,
    min_weight: float = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Exponentially weighted moving covariance matrices."""

    return _move_exp_matrix(
        a,
        alpha,
        min_weight=min_weight,
        op="move_exp_nancovmatrix",
        kernels=_MOVE_EXP_NANCOV_MATRIX_KERNELS,
    )


__all__ = [
    "move_corrmatrix",
    "move_covmatrix",
    "move_exp_nancorrmatrix",
    "move_exp_nancovmatrix",
]
