"""Grouped NaN-aware reductions.

The facade owns public validation, axis normalization, label sizing, output
initialization, and zero-copy label broadcasting. Native kernels receive one
normalized call with dense, zero-based group labels.
"""

from __future__ import annotations

from operator import index

import numpy as np

from mojagg import _native
from mojagg._reduce import _resolve_axes
from mojagg.config import get_config

_GROUP_VALUE_TYPES = (
    (np.dtype(np.float64), "f64"),
    (np.dtype(np.float32), "f32"),
    (np.dtype(np.int64), "i64"),
    (np.dtype(np.int32), "i32"),
)
_GROUP_LABEL_TYPES = (
    (np.dtype(np.int64), "i64"),
    (np.dtype(np.int32), "i32"),
)


def _make_group_kernels(op_name: str):
    return {
        value_dtype: {
            label_dtype: getattr(
                _native,
                f"{op_name}_{value_suffix}_{label_suffix}",
            )
            for label_dtype, label_suffix in _GROUP_LABEL_TYPES
        }
        for value_dtype, value_suffix in _GROUP_VALUE_TYPES
    }


_GROUP_KERNELS = {
    op_name: _make_group_kernels(op_name)
    for op_name in (
        "group_nansum",
        "group_nanmean",
        "group_nanprod",
        "group_nancount",
        "group_nanmin",
        "group_nanmax",
        "group_nanargmin",
        "group_nanargmax",
        "group_nanfirst",
        "group_nanlast",
        "group_nanany",
        "group_nanall",
        "group_nanvar",
        "group_nanstd",
        "group_nansum_of_squares",
    )
}

_GROUP_FLOAT64_PROMOTIONS = {"group_nanvar", "group_nanstd"}


def _normalize_group_axes(values: np.ndarray, labels: np.ndarray, axis) -> tuple[int, ...]:
    if axis is None:
        if values.shape != labels.shape:
            raise ValueError(
                "axis required if values and labels have different "
                f"shapes: {values.shape} vs {labels.shape}"
            )
        return tuple(range(values.ndim))

    if isinstance(axis, tuple):
        axes = _resolve_axes(axis, values, sort_axes=False)
    else:
        axes = _resolve_axes(axis, values, sort_axes=False)

    expected = tuple(values.shape[d] for d in axes)
    if labels.shape != expected:
        raise ValueError(
            f"values must have same shape along axis as labels: {expected} vs {labels.shape}"
        )
    return axes


def _prepare_group_call(values, labels, axis, num_labels, op_name):
    values_arr = np.asarray(values)
    labels_arr = np.asarray(labels)
    if not values_arr.dtype.isnative or not labels_arr.dtype.isnative:
        raise TypeError("grouped operations require native-endian values and labels")
    if labels_arr.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise TypeError(
            f"group labels do not support dtype {labels_arr.dtype}; supported: int32, int64"
        )
    if values_arr.dtype not in _GROUP_KERNELS[op_name]:
        raise TypeError(
            f"{op_name} does not support dtype {values_arr.dtype}; "
            "supported: float32, float64, int32, int64"
        )
    if values_arr.ndim == 0:
        values_arr = values_arr.reshape(1)
    if labels_arr.ndim == 0:
        labels_arr = labels_arr.reshape(1)
    axes = _normalize_group_axes(values_arr, labels_arr, axis)
    nlabels = _resolve_num_labels(labels_arr, num_labels)
    if op_name in _GROUP_FLOAT64_PROMOTIONS and np.issubdtype(values_arr.dtype, np.integer):
        # These operations intentionally follow numbagg's supports_ints=False
        # path. The promotion is visible at the Python boundary; kernels only
        # see their actual accumulation dtype.
        values_arr = values_arr.astype(np.float64)
    # The tuple GUFunc receives one runtime layout per operand.  Expand labels
    # to the value rank as a zero-copy broadcast view so outer dimensions
    # align even when the public grouped API receives labels shaped only like
    # the reduced axes (including non-trailing axes).
    if labels_arr.shape != values_arr.shape:
        full_shape = [1] * values_arr.ndim
        for position, value_axis in enumerate(axes):
            full_shape[value_axis] = labels_arr.shape[position]
        labels_arr = np.broadcast_to(labels_arr.reshape(tuple(full_shape)), values_arr.shape)
    reduced = frozenset(axes)
    outer_shape = tuple(values_arr.shape[d] for d in range(values_arr.ndim) if d not in reduced)
    return values_arr, labels_arr, axes, outer_shape, nlabels


def _resolve_num_labels(labels: np.ndarray, num_labels) -> int:
    required = 0
    if labels.size:
        max_label = int(np.max(labels, initial=-1))
        if max_label >= 0:
            required = max_label + 1
    if num_labels is not None:
        try:
            result = index(num_labels)
        except TypeError:
            raise TypeError("num_labels must be an integer") from None
        if result < 0:
            raise ValueError("num_labels must be non-negative")
        if result < required:
            raise ValueError("num_labels must be greater than the maximum label")
        return result
    return required


def _group_reduce(op_name, values, labels, axis=None, num_labels=None, *, ddof=1):
    values_arr, labels_arr, axes, outer_shape, nlabels = _prepare_group_call(
        values, labels, axis, num_labels, op_name
    )
    internal_shape = outer_shape + (nlabels,)
    result = np.zeros(internal_shape, dtype=values_arr.dtype)
    auxiliary: list[np.ndarray] = []

    if op_name == "group_nanprod" or op_name == "group_nanall":
        result.fill(1)
    elif op_name in {
        "group_nanmin",
        "group_nanmax",
    }:
        if np.issubdtype(result.dtype, np.floating):
            result.fill(np.nan)
        else:
            result.fill(np.iinfo(result.dtype).min)
    elif op_name in {
        "group_nanargmin",
        "group_nanargmax",
        "group_nanfirst",
        "group_nanlast",
    } and np.issubdtype(result.dtype, np.floating):
        result.fill(np.nan)

    if op_name == "group_nanmean" or op_name in {"group_nanmin", "group_nanmax"}:
        auxiliary.append(np.zeros(internal_shape, dtype=np.int64))
    elif op_name in {"group_nanargmin", "group_nanargmax"}:
        best_values = np.zeros(internal_shape, dtype=values_arr.dtype)
        if np.issubdtype(best_values.dtype, np.floating):
            best_values.fill(np.nan)
        auxiliary.extend([best_values, np.zeros(internal_shape, dtype=np.int64)])
    elif op_name == "group_nanfirst" or op_name == "group_nanlast":
        auxiliary.append(np.zeros(internal_shape, dtype=np.int64))
    elif op_name in {"group_nanvar", "group_nanstd"}:
        auxiliary.extend(
            [
                np.zeros(internal_shape, dtype=values_arr.dtype),
                np.zeros(internal_shape, dtype=np.int64),
            ]
        )

    if result.size == 0:
        return result
    cfg = get_config()
    _GROUP_KERNELS[op_name][values_arr.dtype][labels_arr.dtype](
        values_arr,
        labels_arr,
        axes,
        result,
        (tuple(auxiliary), cfg, int(ddof)),
    )
    return result


def group_nansum(values, labels, *, axis=None, num_labels=None):
    """Sum non-NaN values into dense, factorized groups."""
    return _group_reduce("group_nansum", values, labels, axis=axis, num_labels=num_labels)


def group_nanprod(values, labels, *, axis=None, num_labels=None):
    """Multiply non-NaN values into dense groups."""
    return _group_reduce("group_nanprod", values, labels, axis=axis, num_labels=num_labels)


def group_nancount(values, labels, *, axis=None, num_labels=None):
    """Count non-NaN values in each group."""
    return _group_reduce("group_nancount", values, labels, axis=axis, num_labels=num_labels)


def group_nanmean(values, labels, *, axis=None, num_labels=None):
    """Compute the mean of non-NaN values in each dense group."""
    return _group_reduce("group_nanmean", values, labels, axis=axis, num_labels=num_labels)


def group_nanmin(values, labels, *, axis=None, num_labels=None):
    """Compute the minimum of non-NaN values in each dense group."""
    return _group_reduce("group_nanmin", values, labels, axis=axis, num_labels=num_labels)


def group_nanmax(values, labels, *, axis=None, num_labels=None):
    """Compute the maximum of non-NaN values in each dense group."""
    return _group_reduce("group_nanmax", values, labels, axis=axis, num_labels=num_labels)


def group_nanargmin(values, labels, *, axis=None, num_labels=None):
    """Return the local flattened index of each group's minimum."""
    return _group_reduce("group_nanargmin", values, labels, axis=axis, num_labels=num_labels)


def group_nanargmax(values, labels, *, axis=None, num_labels=None):
    """Return the local flattened index of each group's maximum."""
    return _group_reduce("group_nanargmax", values, labels, axis=axis, num_labels=num_labels)


def group_nanfirst(values, labels, *, axis=None, num_labels=None):
    """Return the first non-NaN value in each dense group."""
    return _group_reduce("group_nanfirst", values, labels, axis=axis, num_labels=num_labels)


def group_nanlast(values, labels, *, axis=None, num_labels=None):
    """Return the last non-NaN value in each dense group."""
    return _group_reduce("group_nanlast", values, labels, axis=axis, num_labels=num_labels)


def group_nanany(values, labels, *, axis=None, num_labels=None):
    """Return whether any non-NaN value in each group is truthy."""
    return _group_reduce("group_nanany", values, labels, axis=axis, num_labels=num_labels)


def group_nanall(values, labels, *, axis=None, num_labels=None):
    """Return whether all non-NaN values in each group are truthy."""
    return _group_reduce("group_nanall", values, labels, axis=axis, num_labels=num_labels)


def group_nanvar(values, labels, *, ddof=1, axis=None, num_labels=None):
    """Compute grouped sample variance with the requested degrees of freedom."""
    return _group_reduce(
        "group_nanvar", values, labels, axis=axis, num_labels=num_labels, ddof=ddof
    )


def group_nanstd(values, labels, *, ddof=1, axis=None, num_labels=None):
    """Compute grouped sample standard deviation with the requested ddof."""
    return _group_reduce(
        "group_nanstd", values, labels, axis=axis, num_labels=num_labels, ddof=ddof
    )


def group_nansum_of_squares(values, labels, *, axis=None, num_labels=None):
    """Sum squares of non-NaN values into dense groups."""
    return _group_reduce(
        "group_nansum_of_squares",
        values,
        labels,
        axis=axis,
        num_labels=num_labels,
    )


__all__ = [
    "group_nansum",
    "group_nanmean",
    "group_nanprod",
    "group_nancount",
    "group_nanmin",
    "group_nanmax",
    "group_nanargmin",
    "group_nanargmax",
    "group_nanfirst",
    "group_nanlast",
    "group_nanany",
    "group_nanall",
    "group_nanvar",
    "group_nanstd",
    "group_nansum_of_squares",
]
