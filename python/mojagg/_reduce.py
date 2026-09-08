"""Shared reduce-axis facade — THIN Python wrapper over the Mojo driver.

ALL N-D iteration lives in the Mojo `reduce_axis` driver (one FFI call per
public op). Python's only jobs here, per the zero-copy contract (SKILL.md
§3.1 — nothing else may live outside Mojo):

- dtype promotion following numbagg's rules (visible, never inside kernels)
- big-endian → native byteswap (the ONLY allowed copy, documented)
- axis normalization to a validated tuple (stride-sorted for value reductions)
- output allocation + error translation

Adding a new nanfunc is one line:
    nanmean = reduce_op("nanmean", _NANMEAN_KERNELS, _promote_nanmean_like)
"""

from __future__ import annotations

from collections.abc import Callable
from operator import index

import numpy as np

# Native binding signature: (ndarray, axes: tuple, out: ndarray, threshold).
NativeKernel = Callable[..., object]


def _promote_nansum_like(a: np.ndarray) -> np.ndarray:
    """numbagg promotion for int-supporting reductions (nansum, nancount,
    nanmin/max, nanargmin/max, allnan, anynan).

    int64->int64, int32->int32, uint32->int64, small-int/bool->int32,
    f16->f32, f32/f64 stay.
    """
    if a.dtype == np.float16:
        return a.astype(np.float32)
    if a.dtype in (np.bool_, np.uint8, np.int8, np.int16, np.uint16):
        return a.astype(np.int32)
    if a.dtype == np.uint32:
        return a.astype(np.int64)
    return a


def _promote_nanmean_like(a: np.ndarray) -> np.ndarray:
    """numbagg promotion for float-only reductions (nanmean/var/std).

    Match the reference gufunc's safe-casting order: float16, bool and small
    integers -> float32; int32/int64/uint32/uint64 -> float64. Unsupported
    nonnumeric or wider floating types reach the normal dtype error.
    """
    if a.dtype in (np.float32, np.float64):
        return a
    for dtype in (np.float32, np.float64):
        if np.can_cast(a.dtype, dtype, casting="safe"):
            return a.astype(dtype)
    return a


def _resolve_axes(axis, a: np.ndarray, *, sort_axes: bool = True) -> tuple[int, ...]:
    """Normalize axis (None/int/tuple) to a validated, stride-sorted tuple.

    numbagg semantics: None → all axes; int → 1-tuple; tuple validated for
    duplicates/bounds. Value reductions sort axes by stride DESCENDING so the
    contiguous (stride-1) axis lands innermost; arg reductions preserve tuple
    order because their flat index is part of the public result.
    The Mojo driver only ever receives this tuple form.
    """
    if a.ndim == 0:
        return ()  # caller reshapes to (1,) first
    if axis is None:
        axes = tuple(range(a.ndim))
    elif isinstance(axis, tuple):
        axes = np.lib.array_utils.normalize_axis_tuple(axis, a.ndim)
    else:
        axes = (np.lib.array_utils.normalize_axis_index(axis, a.ndim),)
    if len(axes) == 0:
        # numbagg quirk: axis=() flattens via move_axes and reduces
        # everything (its shape[:-0] slice is empty). Mirror it.
        axes = tuple(range(a.ndim))
    if not sort_axes:
        return axes
    return tuple(sorted(axes, key=lambda ax: -a.strides[ax]))


def _translate_errors(op: str, e: Exception) -> Exception:
    """Map native-kernel failures to the exact exceptions numbagg raises."""
    msg = str(e)
    if "All-NaN slice" in msg or "zero-size array" in msg:
        return ValueError(msg)
    return e


def _reduce_axis(
    arr,
    axis,
    kernels: dict[np.dtype, NativeKernel],
    promote: Callable[[np.ndarray], np.ndarray],
    op: str,
    out_dtype: np.dtype | Callable[[np.dtype], np.dtype] | None = None,
    *,
    native_parameter: int | None = None,
    empty_error: str | None = None,
    invalid_output_error: str | None = None,
    sort_axes: bool = True,
):
    """One FFI call; the Mojo driver owns all N-D iteration.

    No moveaxis, no ascontiguousarray, no per-slice Python loop — strides are
    handed to the driver and handled without copies (any layout, any axis).
    """
    from mojagg.config import get_config

    a = np.asarray(arr)
    if not a.dtype.isnative:
        # big-endian ">f4" is logically float32; byteswap rather than promote.
        a = a.byteswap().view(a.dtype.newbyteorder("="))
    a = promote(a)
    if a.ndim == 0:
        a = a.reshape(1)  # view; numbagg reduces 0-d as a single element

    entry = kernels.get(a.dtype)
    if entry is None:
        raise TypeError(
            f"{op} does not support dtype {a.dtype}; supported: {sorted(str(d) for d in kernels)}"
        )

    axes = _resolve_axes(axis, a, sort_axes=sort_axes)
    axes_set = frozenset(axes)
    out_shape = tuple(a.shape[d] for d in range(a.ndim) if d not in axes_set)
    if out_dtype is None:
        res_dtype = a.dtype
    elif callable(out_dtype) and not isinstance(out_dtype, (np.dtype, type)):
        res_dtype = np.dtype(out_dtype(a.dtype))
    else:
        res_dtype = np.dtype(out_dtype)
    out = np.empty(out_shape, dtype=res_dtype)
    if empty_error is not None and out.size > 0 and any(a.shape[d] == 0 for d in axes):
        raise ValueError(empty_error)
    try:
        cfg = get_config()
        if native_parameter is None:
            entry(a, axes, out, cfg)
        else:
            entry(a, axes, out, native_parameter, cfg)
    except Exception as e:
        raise _translate_errors(op, e) from None
    if invalid_output_error is not None and out.size > 0 and np.any(out < 0):
        raise ValueError(invalid_output_error)
    return out


def reduce_op(
    name: str,
    kernels: dict[np.dtype, NativeKernel],
    promote: Callable[[np.ndarray], np.ndarray],
    *,
    out_dtype: np.dtype | Callable[[np.dtype], np.dtype] | None = None,
    doc: str | None = None,
    empty_error: str | None = None,
    invalid_output_error: str | None = None,
    sort_axes: bool = True,
) -> Callable:
    """Build a public reduction on the native N-D driver.

    `out_dtype`: result dtype — None (= promoted input dtype), a fixed
    np.dtype (e.g. int64 for nancount), or a callable on the promoted dtype
    (e.g. nanmin: int32 -> int64, floats unchanged).
    """

    def fn(a, axis=None, **kwargs):
        return _reduce_axis(
            a,
            axis,
            kernels,
            promote,
            name,
            out_dtype,
            empty_error=empty_error,
            invalid_output_error=invalid_output_error,
            sort_axes=sort_axes,
        )

    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = doc
    return fn


def reduce_op_with_ddof(
    name: str,
    kernels: dict[np.dtype, NativeKernel],
    promote: Callable[[np.ndarray], np.ndarray],
    *,
    out_dtype: np.dtype | Callable[[np.dtype], np.dtype] | None = None,
    doc: str | None = None,
    sort_axes: bool = True,
) -> Callable:
    """Build a reduction whose native kernel receives an integer ``ddof``."""

    def fn(a, axis=None, ddof=1, **kwargs):
        try:
            ddof_int = index(ddof)
        except TypeError:
            raise TypeError(f"{name}: ddof must be an integer") from None
        return _reduce_axis(
            a,
            axis,
            kernels,
            promote,
            name,
            out_dtype,
            native_parameter=ddof_int,
            sort_axes=sort_axes,
        )

    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = doc
    return fn
