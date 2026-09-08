"""Public nanfuncs API — numbagg-compatible signatures.

Each op is a native-binding table + one `reduce_op` call on the shared
thin-wrapper facade (see `_reduce.py`). Adding a new nanfunc = add its
bindings to the native module, then add a table + one line here. No
axis/promotion logic is duplicated per op.

dtype/result matrix (mirrors numbagg's numba signatures):
- nansum:            f64→f64, f32→f32, i64→i64, i32→i32
- nanprod:           f64→f64, f32→f32, i64→i64, i32→i32
- nanmean/nanvar/nanstd: f64→f64, f32→f32 (other numeric types promote)
- nanmin/nanmax:     f64→f64, f32→f32, i64→i64, i32→i64
- nancount/count:    any supported → i64
- allnan/anynan:     any supported → bool
- nanargmin/argmax:  any supported → i64
- nanquantile/nanmedian: numeric inputs → f64 selection results
"""

from __future__ import annotations

import numpy as np

from mojagg import _native
from mojagg._reduce import (
    _promote_nanmean_like,
    _promote_nansum_like,
    _resolve_axes,
    reduce_op,
    reduce_op_with_ddof,
)

_allnan_kernels = {
    np.dtype(np.float64): _native.allnan_f64,
    np.dtype(np.float32): _native.allnan_f32,
    np.dtype(np.int64): _native.allnan_i64,
    np.dtype(np.int32): _native.allnan_i32,
}

_anynan_kernels = {
    np.dtype(np.float64): _native.anynan_f64,
    np.dtype(np.float32): _native.anynan_f32,
    np.dtype(np.int64): _native.anynan_i64,
    np.dtype(np.int32): _native.anynan_i32,
}

_nansum_kernels = {
    np.dtype(np.float64): _native.nansum_f64,
    np.dtype(np.float32): _native.nansum_f32,
    np.dtype(np.int64): _native.nansum_i64,
    np.dtype(np.int32): _native.nansum_i32,
}

_nanmean_kernels = {
    np.dtype(np.float64): _native.nanmean_f64,
    np.dtype(np.float32): _native.nanmean_f32,
}

_nanprod_kernels = {
    np.dtype(np.float64): _native.nanprod_f64,
    np.dtype(np.float32): _native.nanprod_f32,
    np.dtype(np.int64): _native.nanprod_i64,
    np.dtype(np.int32): _native.nanprod_i32,
}

_nanmin_kernels = {
    np.dtype(np.float64): _native.nanmin_f64,
    np.dtype(np.float32): _native.nanmin_f32,
    np.dtype(np.int64): _native.nanmin_i64,
    np.dtype(np.int32): _native.nanmin_i32,
}

_nanmax_kernels = {
    np.dtype(np.float64): _native.nanmax_f64,
    np.dtype(np.float32): _native.nanmax_f32,
    np.dtype(np.int64): _native.nanmax_i64,
    np.dtype(np.int32): _native.nanmax_i32,
}

_nanargmin_kernels = {
    np.dtype(np.float64): _native.nanargmin_f64,
    np.dtype(np.float32): _native.nanargmin_f32,
    np.dtype(np.int64): _native.nanargmin_i64,
    np.dtype(np.int32): _native.nanargmin_i32,
}

_nanargmax_kernels = {
    np.dtype(np.float64): _native.nanargmax_f64,
    np.dtype(np.float32): _native.nanargmax_f32,
    np.dtype(np.int64): _native.nanargmax_i64,
    np.dtype(np.int32): _native.nanargmax_i32,
}

_nanvar_kernels = {
    np.dtype(np.float64): _native.nanvar_f64,
    np.dtype(np.float32): _native.nanvar_f32,
}

_nanstd_kernels = {
    np.dtype(np.float64): _native.nanstd_f64,
    np.dtype(np.float32): _native.nanstd_f32,
}

_nancount_kernels = {
    np.dtype(np.float64): _native.nancount_f64,
    np.dtype(np.float32): _native.nancount_f32,
    np.dtype(np.int64): _native.nancount_i64,
    np.dtype(np.int32): _native.nancount_i32,
}

allnan = reduce_op(
    "allnan",
    _allnan_kernels,
    _promote_nansum_like,
    out_dtype=np.bool_,
    doc="""True iff all elements along a given axis are NaN (empty → True).

Matches numbagg.allnan semantics.""",
)

anynan = reduce_op(
    "anynan",
    _anynan_kernels,
    _promote_nansum_like,
    out_dtype=np.bool_,
    doc="""True iff any element along a given axis is NaN (empty → False).

Matches numbagg.anynan semantics.""",
)

nansum = reduce_op(
    "nansum",
    _nansum_kernels,
    _promote_nansum_like,
    doc="""Sum of array elements over a given axis, treating NaN as zero.

Matches numbagg.nansum / numpy.nansum semantics.""",
)

nanmean = reduce_op(
    "nanmean",
    _nanmean_kernels,
    _promote_nanmean_like,
    doc="""Mean of non-NaN elements over a given axis.

Empty/all-NaN slices return NaN. Float32/float64 inputs retain their result
dtype; the running sum uses float64 to match numbagg. Float16, bool and small
integers promote to float32; 32/64-bit integers promote to float64.""",
)

nanprod = reduce_op(
    "nanprod",
    _nanprod_kernels,
    _promote_nansum_like,
    doc="""Product of array elements over a given axis, treating NaN as one.

NaN/empty semantics match numpy.nanprod; native integer result dtype is
retained. This is an extension: numbagg has no standalone nanprod.""",
)

nanmin = reduce_op(
    "nanmin",
    _nanmin_kernels,
    _promote_nansum_like,
    out_dtype=lambda dtype: np.int64 if dtype == np.int32 else dtype,
    empty_error="zero-size array to reduction operation fmin which has no identity",
    doc="""Minimum of non-NaN elements over a given axis.

Matches numbagg.nanmin semantics.""",
)

nanmax = reduce_op(
    "nanmax",
    _nanmax_kernels,
    _promote_nansum_like,
    out_dtype=lambda dtype: np.int64 if dtype == np.int32 else dtype,
    empty_error="zero-size array to reduction operation fmax which has no identity",
    doc="""Maximum of non-NaN elements over a given axis.

Matches numbagg.nanmax semantics.""",
)

nanargmin = reduce_op(
    "nanargmin",
    _nanargmin_kernels,
    _promote_nansum_like,
    out_dtype=np.int64,
    invalid_output_error="All-NaN slice encountered",
    sort_axes=False,
    doc="""Index of the first minimum non-NaN element along an axis.

Matches numbagg.nanargmin semantics.""",
)

nanargmax = reduce_op(
    "nanargmax",
    _nanargmax_kernels,
    _promote_nansum_like,
    out_dtype=np.int64,
    invalid_output_error="All-NaN slice encountered",
    sort_axes=False,
    doc="""Index of the first maximum non-NaN element along an axis.

Matches numbagg.nanargmax semantics.""",
)

nanvar = reduce_op_with_ddof(
    "nanvar",
    _nanvar_kernels,
    _promote_nanmean_like,
    doc="""Variance of non-NaN elements, with ddof=1 by default.

Matches numbagg.nanvar semantics.""",
)

nanstd = reduce_op_with_ddof(
    "nanstd",
    _nanstd_kernels,
    _promote_nanmean_like,
    doc="""Standard deviation of non-NaN elements, with ddof=1 by default.

Matches numbagg.nanstd semantics.""",
)

nancount = reduce_op(
    "nancount",
    _nancount_kernels,
    _promote_nansum_like,
    out_dtype=np.int64,
    doc="""Count non-NaN elements along a given axis.

Matches numbagg.nancount semantics.""",
)

count = nancount


_NANQUANTILE_KERNELS = {
    np.dtype(np.float64): _native.nanquantile_f64,
    np.dtype(np.float32): _native.nanquantile_f32,
}


def _selection_input(arr) -> np.ndarray:
    """Normalize supported numeric inputs for the f64 selection kernel."""
    a = np.asarray(arr)
    if not a.dtype.isnative:
        a = a.byteswap().view(a.dtype.newbyteorder("="))
    if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)):
        raise TypeError(f"nanquantile does not support dtype {a.dtype}; expected a numeric array")
    return a.astype(np.float64, copy=False)


def nanquantile(a, quantiles, axis=None, **kwargs):
    """Quantiles of non-NaN values using numbagg's linear interpolation."""
    from collections.abc import Iterable

    from mojagg.config import get_config

    if not isinstance(quantiles, Iterable):
        squeeze = True
        q_arr = np.array([quantiles], dtype=np.float64)
    else:
        squeeze = False
        q_arr = np.asarray(quantiles, dtype=np.float64)

    try:
        invalid = np.any(q_arr < 0) or np.any(q_arr > 1)
    except TypeError:
        raise ValueError(
            f"quantiles must be in the range [0, 1], inclusive. Got {quantiles}."
        ) from None
    if invalid:
        raise ValueError(f"quantiles must be in the range [0, 1], inclusive. Got {quantiles}.")

    arr_val = np.asarray(a)
    if not arr_val.dtype.isnative:
        arr_val = arr_val.byteswap().view(arr_val.dtype.newbyteorder("="))

    if arr_val.dtype not in (np.float32, np.float64):
        if np.issubdtype(arr_val.dtype, np.floating) or np.issubdtype(arr_val.dtype, np.integer):
            arr_val = arr_val.astype(np.float64)
        else:
            raise TypeError(
                f"nanquantile does not support dtype {arr_val.dtype}; expected a numeric array"
            )

    if arr_val.ndim == 0:
        arr_val = arr_val.reshape(1)

    axes = _resolve_axes(axis, arr_val)
    axes_set = frozenset(axes)
    outer_shape = tuple(arr_val.shape[d] for d in range(arr_val.ndim) if d not in axes_set)

    num_q = len(q_arr)
    out_shape = outer_shape + (num_q,)
    out = np.empty(out_shape, dtype=arr_val.dtype)

    cfg = get_config()
    kernel = _NANQUANTILE_KERNELS[arr_val.dtype]
    kernel(arr_val, axes, q_arr, out, cfg)

    result = np.moveaxis(out, -1, 0)
    if squeeze:
        result = result.squeeze(axis=0)
    return result


def nanmedian(a, *, axis=None, **kwargs):
    """Median of non-NaN values using numbagg's quantile implementation."""
    return nanquantile(a, quantiles=0.5, axis=axis, **kwargs)
