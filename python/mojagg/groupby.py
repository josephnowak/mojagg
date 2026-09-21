"""Grouped NaN-aware reductions.

The facade owns public validation, axis normalization, label sizing, and
zero-copy label broadcasting. The native binding derives and initializes the
complete output signature before executing the kernel.
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
# Native kernels for these reductions operate on floating-point accumulators.
# Integer means are cast back after reduction to match numbagg's integer result.
_GROUP_FLOAT64_PROMOTIONS = {"group_nanmean", "group_nanvar", "group_nanstd"}
_GROUP_FLOAT_VALUE_TYPES = _GROUP_VALUE_TYPES[:2]
_GROUP_LABEL_TYPES = (
    (np.dtype(np.int64), "i64"),
    (np.dtype(np.int32), "i32"),
)


def _make_group_kernels(op_name: str):
    if op_name in _GROUP_FLOAT64_PROMOTIONS:
        value_types = _GROUP_FLOAT_VALUE_TYPES
    else:
        value_types = _GROUP_VALUE_TYPES
    return {
        value_dtype: {
            label_dtype: getattr(
                _native,
                f"{op_name}_{value_suffix}_{label_suffix}",
            )
            for label_dtype, label_suffix in _GROUP_LABEL_TYPES
        }
        for value_dtype, value_suffix in value_types
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

_GROUP_BOOL_UNSUPPORTED = {"group_nanvar", "group_nanstd"}
_GROUP_BOOL_SUPPORTED = set(_GROUP_KERNELS) - _GROUP_BOOL_UNSUPPORTED


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
    if not np.issubdtype(labels_arr.dtype, np.integer):
        raise TypeError(f"group labels do not support dtype {labels_arr.dtype}; supported: integer")
    if labels_arr.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        labels_arr = labels_arr.astype(np.int64)
    if values_arr.dtype == np.dtype(np.bool_):
        if op_name not in _GROUP_BOOL_SUPPORTED:
            raise TypeError(
                f"{op_name} does not support boolean input. Convert to a numeric type first."
            )
        # numbagg converts supported boolean grouped inputs to int32 before
        # selecting the gufunc signature. Keep that conversion at the public
        # boundary so native kernels only handle their registered dtypes.
        values_arr = values_arr.astype(np.int32)
    if op_name in _GROUP_FLOAT64_PROMOTIONS and np.issubdtype(values_arr.dtype, np.integer):
        # These operations intentionally follow numbagg's supports_ints=False
        # path. The promotion is visible at the Python boundary; kernels only
        # see their actual accumulation dtype.
        values_arr = values_arr.astype(np.float64)
    elif np.issubdtype(values_arr.dtype, np.integer) and values_arr.dtype not in (
        np.dtype(np.int32),
        np.dtype(np.int64),
    ):
        values_arr = values_arr.astype(np.int32)
    if values_arr.dtype not in _GROUP_KERNELS[op_name]:
        supported = ", ".join(str(dtype) for dtype in _GROUP_KERNELS[op_name])
        raise TypeError(
            f"{op_name} does not support dtype {values_arr.dtype}; supported: {supported}"
        )
    if values_arr.ndim == 0:
        values_arr = values_arr.reshape(1)
    if labels_arr.ndim == 0:
        labels_arr = labels_arr.reshape(1)
    axes = _normalize_group_axes(values_arr, labels_arr, axis)
    nlabels = _resolve_num_labels(labels_arr, num_labels)
    # The tuple GUFunc receives one runtime layout per operand.  Expand labels
    # to the value rank as a zero-copy broadcast view so outer dimensions
    # align even when the public grouped API receives labels shaped only like
    # the reduced axes (including non-trailing axes).
    if labels_arr.shape != values_arr.shape:
        full_shape = [1] * values_arr.ndim
        for position, value_axis in enumerate(axes):
            full_shape[value_axis] = labels_arr.shape[position]
        labels_arr = np.broadcast_to(labels_arr.reshape(tuple(full_shape)), values_arr.shape)
    return values_arr, labels_arr, axes, nlabels


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
    original_dtype = np.asarray(values).dtype
    values_arr, labels_arr, axes, nlabels = _prepare_group_call(
        values, labels, axis, num_labels, op_name
    )
    cfg = get_config()
    result = _GROUP_KERNELS[op_name][values_arr.dtype][labels_arr.dtype](
        values_arr,
        labels_arr,
        axes,
        nlabels,
        (cfg, int(ddof)),
    )
    if op_name == "group_nanmean" and np.issubdtype(original_dtype, np.integer):
        integer_min = np.iinfo(original_dtype).min
        with np.errstate(invalid="ignore", over="ignore"):
            return np.where(np.isnan(result), integer_min, result).astype(original_dtype)
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


# Match the metadata exposed by numbagg's ``groupndreduce`` wrappers.  The
# upstream grouped test suite uses these attributes during collection and for
# dtype capability checks.
for _group_function in (
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nansum,
    group_nansum_of_squares,
):
    _group_function.supports_bool = True
    _group_function.supports_ints = True
    _group_function.supports_ddof = False

for _group_function in (group_nanvar, group_nanstd):
    _group_function.supports_bool = False
    _group_function.supports_ints = False
    _group_function.supports_ddof = True


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
