"""Python bindings for the nanfuncs family — THIN layer.

One native call handles all axes and layouts through gufunc driver. The facade
normalizes axes and allocates output; these bindings validate dtype/metadata
and borrow pointers without copying. Operations implement GUFuncKernel or
Reduction1D directly.
"""

from std.collections import Span
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.core.ndview import DimArray, MAX_NDIM, NDView, _dtype_name
from mojagg.drivers.gufunc import apply_gufunc
from mojagg.nanfuncs.fill import BFill, FFill
from mojagg.nanfuncs.allnan import AllNan
from mojagg.nanfuncs.anynan import AnyNan
from mojagg.nanfuncs.nanargmax import NanArgMax
from mojagg.nanfuncs.nanargmin import NanArgMin
from mojagg.nanfuncs.nancount import NanCount
from mojagg.nanfuncs.nanmatrix import (
    NanCorrOp,
    NanCovOp,
    matrix_batch,
)
from mojagg.nanfuncs.nanmax import NanMax
from mojagg.nanfuncs.nanmean import NanMean
from mojagg.nanfuncs.nanmin import NanMin
from mojagg.nanfuncs.nanprod import NanProd
from mojagg.nanfuncs.nanquantile import nanquantile_driver
from mojagg.nanfuncs.nansum import NanSum
from mojagg.nanfuncs.nanvar import NanVar


# --- N-D axis bindings --------------------------------------------------------
# One generic binding per op, instantiated per dtype at module registration.
# Contract: `axes` is a NORMALIZED Python tuple from the facade (deduped,
# negatives resolved, stride-sorted descending, k >= 1); `out_arr` is a
# preallocated numpy array of the op's result dtype; `threshold` is the
# resolved MojaggConfig.parallel_threshold or MojaggConfig object.


@always_inline
def _cfg_params(cfg: PythonObject) raises -> Tuple[Int, Int]:
    try:
        var t = Int(py=cfg.parallel_threshold)
        var w = Int(py=cfg.threads)
        return (t, w)
    except:
        return (Int(py=cfg), 0)


def _out_addr[
    out_dtype: DType
](out_arr: PythonObject, op: String) raises -> Int:
    """Validate the preallocated output array's dtype, return its address.

    Drivers store the address as an `Int` (struct fields cannot expose
    MutAnyOrigin pointers) and reconstitute the typed pointer on demand.
    """
    comptime want = _dtype_name[out_dtype]()
    var got = String(py=out_arr.dtype.name)
    if got != want:
        raise Error(op + ": output must be " + want + ", got " + got)
    return Int(py=out_arr.ctypes.data)


def _axes_from_py(
    axes: PythonObject, op: String
) raises -> Tuple[DimArray, Int]:
    """Read a normalized axes tuple into a stack array (ndim <= MAX_NDIM)."""
    var k = Int(py=axes.__len__())
    if k < 1 or k > MAX_NDIM:
        raise Error(
            op + ": expected 1.." + String(MAX_NDIM) + " axes, got " + String(k)
        )
    var ax = DimArray(fill=0)
    for i in range(k):
        ax[i] = Int(py=axes[i])
    return (ax^, k)


def allnan_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """Allnan over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "allnan")
    var t = _axes_from_py(axes, "allnan")
    var out_addr = _out_addr[DType.bool](out_arr, "allnan")
    var op = AllNan[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def anynan_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """AnyNan over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "anynan")
    var t = _axes_from_py(axes, "anynan")
    var out_addr = _out_addr[DType.bool](out_arr, "anynan")
    var op = AnyNan[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nansum_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanSum over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "nansum")
    var t = _axes_from_py(axes, "nansum")
    var out_addr = _out_addr[dtype](out_arr, "nansum")
    var op = NanSum[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nanmean_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanMean over axes, merging sum/count before final division."""
    var view = NDView[dtype].from_numpy(arr, "nanmean")
    var t = _axes_from_py(axes, "nanmean")
    var out_addr = _out_addr[dtype](out_arr, "nanmean")
    var op = NanMean[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nanprod_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanProd over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "nanprod")
    var t = _axes_from_py(axes, "nanprod")
    var out_addr = _out_addr[dtype](out_arr, "nanprod")
    var op = NanProd[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nanmin_binding[
    dtype: DType,
    result_dtype: DType,
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanMin over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "nanmin")
    var t = _axes_from_py(axes, "nanmin")
    var out_addr = _out_addr[result_dtype](out_arr, "nanmin")
    var op = NanMin[dtype, result_dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nanmax_binding[
    dtype: DType,
    result_dtype: DType,
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanMax over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "nanmax")
    var t = _axes_from_py(axes, "nanmax")
    var out_addr = _out_addr[result_dtype](out_arr, "nanmax")
    var op = NanMax[dtype, result_dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nanargmin_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanArgMin over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "nanargmin")
    var t = _axes_from_py(axes, "nanargmin")
    var out_addr = _out_addr[DType.int64](out_arr, "nanargmin")
    var op = NanArgMin[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nanargmax_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanArgMax over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "nanargmax")
    var t = _axes_from_py(axes, "nanargmax")
    var out_addr = _out_addr[DType.int64](out_arr, "nanargmax")
    var op = NanArgMax[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def nanvar_binding[
    dtype: DType,
    take_sqrt: Bool,
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    ddof: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanVar/NanStd over `axes`, with a runtime ddof parameter."""
    var op = "nanstd"
    comptime if not take_sqrt:
        op = "nanvar"
    var view = NDView[dtype].from_numpy(arr, op)
    var t = _axes_from_py(axes, op)
    var out_addr = _out_addr[dtype](out_arr, op)
    var kernel = NanVar[dtype, take_sqrt](ddof=Int(py=ddof))
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(
        kernel,
        view,
        t[0],
        t[1],
        out_addr,
        cfg_p[0],
        cfg_p[1],
    )
    return out_arr


def nancount_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanCount over `axes` of an N-D array — one FFI call, zero copies."""
    var view = NDView[dtype].from_numpy(arr, "nancount")
    var t = _axes_from_py(axes, "nancount")
    var out_addr = _out_addr[DType.int64](out_arr, "nancount")
    var op = NanCount[dtype]()
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(op, view, t[0], t[1], out_addr, cfg_p[0], cfg_p[1])
    return out_arr


def ffill_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    limit: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    var view = NDView[dtype].from_numpy(arr, "ffill")
    var out_view = NDView[dtype].from_numpy(out_arr, "ffill")
    var t = _axes_from_py(axes, "ffill")
    var op = FFill[dtype](limit=Int(py=limit))
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(
        op,
        view,
        out_view,
        t[0],
        t[1],
        cfg_p[0],
        cfg_p[1],
    )
    return out_arr


def bfill_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    limit: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    var view = NDView[dtype].from_numpy(arr, "bfill")
    var out_view = NDView[dtype].from_numpy(out_arr, "bfill")
    var t = _axes_from_py(axes, "bfill")
    var op = BFill[dtype](limit=Int(py=limit))
    var cfg_p = _cfg_params(threshold)
    apply_gufunc(
        op,
        view,
        out_view,
        t[0],
        t[1],
        cfg_p[0],
        cfg_p[1],
    )
    return out_arr


def nancovmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    out_arr: PythonObject,
    batch: PythonObject,
    n_vars: PythonObject,
    n_obs: PythonObject,
    threshold: PythonObject,
    workers: PythonObject,
) raises -> PythonObject:
    var src_addr = Int(py=arr.ctypes.data)
    var dst_addr = Int(py=out_arr.ctypes.data)
    matrix_batch[NanCovOp[dtype]](
        src_addr,
        dst_addr,
        Int(py=batch),
        Int(py=n_vars),
        Int(py=n_obs),
        Int(py=threshold),
        Int(py=workers),
    )
    return out_arr


def nancorrmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    out_arr: PythonObject,
    batch: PythonObject,
    n_vars: PythonObject,
    n_obs: PythonObject,
    threshold: PythonObject,
    workers: PythonObject,
) raises -> PythonObject:
    var src_addr = Int(py=arr.ctypes.data)
    var dst_addr = Int(py=out_arr.ctypes.data)
    matrix_batch[NanCorrOp[dtype]](
        src_addr,
        dst_addr,
        Int(py=batch),
        Int(py=n_vars),
        Int(py=n_obs),
        Int(py=threshold),
        Int(py=workers),
    )
    return out_arr


def nanquantile_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    quantiles_arr: PythonObject,
    out_arr: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    var view = NDView[dtype].from_numpy(arr, "nanquantile")
    var t = _axes_from_py(axes, "nanquantile")
    var q_addr = Int(py=quantiles_arr.ctypes.data)
    var num_q = Int(py=quantiles_arr.__len__())
    var out_addr = _out_addr[dtype](out_arr, "nanquantile")
    var cfg_p = _cfg_params(cfg)
    nanquantile_driver[dtype](
        view,
        t[0],
        t[1],
        q_addr,
        num_q,
        out_addr,
        cfg_p[0],
        cfg_p[1],
    )
    return out_arr


@export
def PyInit_nanfuncs_native() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("nanfuncs_native")
        m.def_function[allnan_binding[DType.float64]]("allnan_f64")
        m.def_function[allnan_binding[DType.float32]]("allnan_f32")
        m.def_function[allnan_binding[DType.int64]]("allnan_i64")
        m.def_function[allnan_binding[DType.int32]]("allnan_i32")

        m.def_function[anynan_binding[DType.float64]]("anynan_f64")
        m.def_function[anynan_binding[DType.float32]]("anynan_f32")
        m.def_function[anynan_binding[DType.int64]]("anynan_i64")
        m.def_function[anynan_binding[DType.int32]]("anynan_i32")

        m.def_function[nansum_binding[DType.float64]]("nansum_f64")
        m.def_function[nansum_binding[DType.float32]]("nansum_f32")
        m.def_function[nansum_binding[DType.int64]]("nansum_i64")
        m.def_function[nansum_binding[DType.int32]]("nansum_i32")

        m.def_function[nanmean_binding[DType.float64]]("nanmean_f64")
        m.def_function[nanmean_binding[DType.float32]]("nanmean_f32")

        m.def_function[nanprod_binding[DType.float64]]("nanprod_f64")
        m.def_function[nanprod_binding[DType.float32]]("nanprod_f32")
        m.def_function[nanprod_binding[DType.int64]]("nanprod_i64")
        m.def_function[nanprod_binding[DType.int32]]("nanprod_i32")

        m.def_function[nanmin_binding[DType.float64, DType.float64]](
            "nanmin_f64"
        )
        m.def_function[nanmin_binding[DType.float32, DType.float32]](
            "nanmin_f32"
        )
        m.def_function[nanmin_binding[DType.int64, DType.int64]]("nanmin_i64")
        m.def_function[nanmin_binding[DType.int32, DType.int64]]("nanmin_i32")

        m.def_function[nanmax_binding[DType.float64, DType.float64]](
            "nanmax_f64"
        )
        m.def_function[nanmax_binding[DType.float32, DType.float32]](
            "nanmax_f32"
        )
        m.def_function[nanmax_binding[DType.int64, DType.int64]]("nanmax_i64")
        m.def_function[nanmax_binding[DType.int32, DType.int64]]("nanmax_i32")

        m.def_function[nanargmin_binding[DType.float64]]("nanargmin_f64")
        m.def_function[nanargmin_binding[DType.float32]]("nanargmin_f32")
        m.def_function[nanargmin_binding[DType.int64]]("nanargmin_i64")
        m.def_function[nanargmin_binding[DType.int32]]("nanargmin_i32")

        m.def_function[nanargmax_binding[DType.float64]]("nanargmax_f64")
        m.def_function[nanargmax_binding[DType.float32]]("nanargmax_f32")
        m.def_function[nanargmax_binding[DType.int64]]("nanargmax_i64")
        m.def_function[nanargmax_binding[DType.int32]]("nanargmax_i32")

        m.def_function[nanvar_binding[DType.float64, False]]("nanvar_f64")
        m.def_function[nanvar_binding[DType.float32, False]]("nanvar_f32")
        m.def_function[nanvar_binding[DType.float64, True]]("nanstd_f64")
        m.def_function[nanvar_binding[DType.float32, True]]("nanstd_f32")

        m.def_function[nancount_binding[DType.float64]]("nancount_f64")
        m.def_function[nancount_binding[DType.float32]]("nancount_f32")
        m.def_function[nancount_binding[DType.int64]]("nancount_i64")
        m.def_function[nancount_binding[DType.int32]]("nancount_i32")

        m.def_function[ffill_binding[DType.float64]]("ffill_f64")
        m.def_function[ffill_binding[DType.float32]]("ffill_f32")
        m.def_function[ffill_binding[DType.int64]]("ffill_i64")
        m.def_function[ffill_binding[DType.int32]]("ffill_i32")

        m.def_function[bfill_binding[DType.float64]]("bfill_f64")
        m.def_function[bfill_binding[DType.float32]]("bfill_f32")
        m.def_function[bfill_binding[DType.int64]]("bfill_i64")
        m.def_function[bfill_binding[DType.int32]]("bfill_i32")

        m.def_function[nancovmatrix_binding[DType.float64]]("nancovmatrix_f64")
        m.def_function[nancovmatrix_binding[DType.float32]]("nancovmatrix_f32")

        m.def_function[nancorrmatrix_binding[DType.float64]](
            "nancorrmatrix_f64"
        )
        m.def_function[nancorrmatrix_binding[DType.float32]](
            "nancorrmatrix_f32"
        )

        m.def_function[nanquantile_binding[DType.float64]]("nanquantile_f64")
        m.def_function[nanquantile_binding[DType.float32]]("nanquantile_f32")

        return m.finalize()
    except e:
        abort(String("failed to create nanfuncs_native module: ", e))
