"""Shared reduce-axis driver for the nanfuncs family.

One public helper, `_reduce_axis`, holds ALL the op-independent machinery:
dtype normalization + numbagg promotion, axis normalization, moveaxis,
outer-slice looping, and output assembly. Each op only supplies a
`{dtype: flat_1d_binding}` table. This mirrors numbagg's ndaggregate
decorator: the gufunc-wrapping logic is written once, kernels stay pure 1-D.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Flat 1-D binding signature: contiguous 1-D ndarray -> python scalar.
FlatKernel = Callable[[np.ndarray], object]


def _promote_nansum_like(a: np.ndarray) -> np.ndarray:
    """numbagg promotion for additive reductions (nansum, nancount-like).

    int64->int64, int32->int32, small-int/bool->int32, f16->f32, f32/f64 stay.
    """
    if a.dtype == np.float16:
        return a.astype(np.float32)
    if a.dtype in (np.bool_, np.uint8, np.int8, np.int16, np.uint16):
        return a.astype(np.int32)
    return a


def _promote_nanmean_like(a: np.ndarray) -> np.ndarray:
    """numbagg promotion for float-only reductions (nanmean/var/std).

    Everything non-float promotes to float64; f16->f32.
    """
    if a.dtype == np.float16:
        return a.astype(np.float32)
    if a.dtype not in (np.float64, np.float32):
        return a.astype(np.float64)
    return a


def _reduce_axis(
    arr,
    axis,
    flat_kernels: dict[np.dtype, FlatKernel],
    promote: Callable[[np.ndarray], np.ndarray],
    op: str,
):
    """Apply a flat 1-D reduction over `axis` of `arr` (numbagg semantics).

    axis=None  -> reduce all axes, return 0-d result
    axis=int   -> reduce that axis, output shape = input minus that axis
    """
    a = np.asarray(arr)
    if not a.dtype.isnative:
        # big-endian ">f4" is logically float32; byteswap rather than promote.
        a = a.byteswap().view(a.dtype.newbyteorder("="))
    a = promote(a)

    flat_fn = flat_kernels.get(a.dtype)
    if flat_fn is None:
        raise TypeError(
            f"{op} does not support dtype {a.dtype}; "
            f"supported: {sorted(str(d) for d in flat_kernels)}"
        )

    if axis is None:
        return np.asarray(flat_fn(np.ascontiguousarray(a).ravel()), dtype=a.dtype)

    axis = np.lib.array_utils.normalize_axis_index(axis, a.ndim)
    moved = np.ascontiguousarray(np.moveaxis(a, axis, -1))
    outer_shape, n = moved.shape[:-1], moved.shape[-1]

    # 2-D (outer, reduce-len); explicit product since reshape(-1, 0) is
    # ambiguous for zero-size inputs.
    outer = 1
    for d in outer_shape:
        outer *= d
    rows = moved.reshape(outer, n)

    out = np.empty(outer, dtype=a.dtype)
    for i in range(outer):
        out[i] = flat_fn(rows[i])
    return out.reshape(outer_shape)
