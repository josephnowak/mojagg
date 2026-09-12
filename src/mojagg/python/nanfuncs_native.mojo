"""Python bindings for the nanfuncs family.

The binding validates the Python boundary once, borrows NumPy storage into
typed ``LayoutTensor`` views, and invokes the canonical tuple-based GUFunc.
All axis traversal, scratch materialization, and scheduling happen in Mojo.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.core.tensor_view import (
    DimArray as TensorDimArray,
    MAX_RANK,
    TensorArg,
    _dtype_name,
)
from mojagg.python.tensor_view_binding import (
    read_tensor_from_numpy,
    write_tensor_from_numpy,
)
from mojagg.drivers.gufunc import (
    DispatchPolicy,
    GUFunc,
    GUFuncOperation,
)
from mojagg.nanfuncs.fill import FillKernel
from mojagg.nanfuncs.allnan import AllNan
from mojagg.nanfuncs.anynan import AnyNan
from mojagg.nanfuncs.nanargmax import NanArgMax
from mojagg.nanfuncs.nanargmin import NanArgMin
from mojagg.nanfuncs.nancount import NanCount
from mojagg.nanfuncs.nanmatrix import (
    NanCorrOp,
    NanCovOp,
)
from mojagg.nanfuncs.nanmax import NanMax
from mojagg.nanfuncs.nanmean import NanMean
from mojagg.nanfuncs.nanmin import NanMin
from mojagg.nanfuncs.nanprod import NanProd
from mojagg.nanfuncs.nanquantile import NanQuantileKernel
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


def _axes_from_py(
    axes: PythonObject, op: String
) raises -> Tuple[TensorDimArray, Int]:
    """Read a normalized axes tuple into a fixed-capacity stack array."""
    var k = Int(py=axes.__len__())
    if k < 1 or k > MAX_RANK:
        raise Error(
            op + ": expected 1.." + String(MAX_RANK) + " axes, got " + String(k)
        )
    var ax = TensorDimArray(fill=0)
    for i in range(k):
        ax[i] = Int(py=axes[i])
    return (ax^, k)


def _validate_dtype[dtype: DType](arr: PythonObject, op: String) raises:
    var expected = _dtype_name[dtype]()
    var actual = String(py=arr.dtype.name)
    if expected != actual:
        raise Error(op + ": expected dtype " + expected + ", got " + actual)


def _validate_output_dtype[dtype: DType](arr: PythonObject, op: String) raises:
    """Validate a caller-provided output with an output-specific diagnostic."""

    var expected = _dtype_name[dtype]()
    var actual = String(py=arr.dtype.name)
    if expected != actual:
        raise Error(op + ": output must be " + expected + ", got " + actual)


def _apply_reduction[
    value_dtype: DType,
    out_dtype: DType,
    Op: GUFuncOperation,
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run one concrete tuple operation over normalized reduction axes."""

    _validate_dtype[value_dtype](arr, op_name)
    _validate_output_dtype[out_dtype](out_arr, op_name)
    var input_tensor = read_tensor_from_numpy[value_dtype](arr)
    var output_tensor = write_tensor_from_numpy[out_dtype](out_arr)
    var input_ndim = Int(py=arr.ndim)
    var output_ndim = Int(py=out_arr.ndim)
    var parsed = _axes_from_py(axes, op_name)
    var tensors = Tuple(
        TensorArg[value_dtype, False].from_read_tensor(
            input_tensor,
            input_ndim,
            Int(py=arr.ctypes.data),
        ),
        TensorArg[out_dtype, True].from_write_tensor(
            output_tensor,
            output_ndim,
            Int(py=out_arr.ctypes.data),
        ),
    )
    var driver = GUFunc[Op](operation)
    var output_axes = TensorDimArray(fill=0)
    driver.execute(
        tensors,
        parsed[0],
        parsed[1],
        output_axes,
        0,
        _policy(cfg),
    )
    return out_arr


def _apply_matrix[
    value_dtype: DType,
    Op: GUFuncOperation,
](
    arr: PythonObject,
    out_arr: PythonObject,
    n_vars: Int,
    n_obs: Int,
    policy: DispatchPolicy,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run one matrix operation over the two trailing core axes."""

    _validate_dtype[value_dtype](arr, op_name)
    _validate_output_dtype[value_dtype](out_arr, op_name)
    var input_tensor = read_tensor_from_numpy[value_dtype](arr)
    var output_tensor = write_tensor_from_numpy[value_dtype](out_arr)
    var input_ndim = Int(py=arr.ndim)
    var output_ndim = Int(py=out_arr.ndim)
    if input_ndim < 2 or output_ndim < 2:
        raise Error(op_name + " requires at least two dimensions")

    var input_axes = TensorDimArray(fill=0)
    input_axes[0] = input_ndim - 2
    input_axes[1] = input_ndim - 1
    var output_axes = TensorDimArray(fill=0)
    output_axes[0] = output_ndim - 2
    output_axes[1] = output_ndim - 1
    var tensors = Tuple(
        TensorArg[value_dtype, False].from_read_tensor(
            input_tensor,
            input_ndim,
            Int(py=arr.ctypes.data),
        ),
        TensorArg[value_dtype, True].from_write_tensor(
            output_tensor,
            output_ndim,
            Int(py=out_arr.ctypes.data),
        ),
    )
    var driver = GUFunc[Op](operation)
    driver.execute(tensors, input_axes, 2, output_axes, 2, policy)
    return out_arr


def _policy(cfg: PythonObject) raises -> DispatchPolicy:
    var values = _cfg_params(cfg)
    return DispatchPolicy(values[1], values[0])


def allnan_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """Allnan over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.bool, AllNan[dtype]](
        arr, axes, out_arr, threshold, AllNan[dtype](), "allnan"
    )


def anynan_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """AnyNan over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.bool, AnyNan[dtype]](
        arr, axes, out_arr, threshold, AnyNan[dtype](), "anynan"
    )


def nansum_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    return _apply_reduction[dtype, dtype, NanSum[dtype]](
        arr,
        axes,
        out_arr,
        threshold,
        NanSum[dtype](),
        "nansum",
    )


def nanmean_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanMean over axes, merging sum/count before final division."""
    return _apply_reduction[dtype, dtype, NanMean[dtype]](
        arr, axes, out_arr, threshold, NanMean[dtype](), "nanmean"
    )


def nanprod_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanProd over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, dtype, NanProd[dtype]](
        arr, axes, out_arr, threshold, NanProd[dtype](), "nanprod"
    )


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
    return _apply_reduction[dtype, result_dtype, NanMin[dtype, result_dtype]](
        arr,
        axes,
        out_arr,
        threshold,
        NanMin[dtype, result_dtype](),
        "nanmin",
    )


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
    return _apply_reduction[dtype, result_dtype, NanMax[dtype, result_dtype]](
        arr,
        axes,
        out_arr,
        threshold,
        NanMax[dtype, result_dtype](),
        "nanmax",
    )


def nanargmin_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanArgMin over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.int64, NanArgMin[dtype]](
        arr, axes, out_arr, threshold, NanArgMin[dtype](), "nanargmin"
    )


def nanargmax_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanArgMax over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.int64, NanArgMax[dtype]](
        arr, axes, out_arr, threshold, NanArgMax[dtype](), "nanargmax"
    )


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
    var kernel = NanVar[dtype, take_sqrt](ddof=Int(py=ddof))
    return _apply_reduction[dtype, dtype, NanVar[dtype, take_sqrt]](
        arr,
        axes,
        out_arr,
        threshold,
        kernel,
        op,
    )


def nancount_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanCount over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.int64, NanCount[dtype]](
        arr, axes, out_arr, threshold, NanCount[dtype](), "nancount"
    )


def _apply_fill[
    dtype: DType, backward: Bool
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    limit: PythonObject,
    cfg: PythonObject,
    op_name: String,
) raises -> PythonObject:
    _validate_dtype[dtype](arr, op_name)
    _validate_output_dtype[dtype](out_arr, op_name)
    var input_tensor = read_tensor_from_numpy[dtype](arr)
    var output_tensor = write_tensor_from_numpy[dtype](out_arr)
    var input_ndim = Int(py=arr.ndim)
    var output_ndim = Int(py=out_arr.ndim)
    var parsed = _axes_from_py(axes, op_name)
    var tensors = Tuple(
        TensorArg[dtype, False].from_read_tensor(
            input_tensor,
            input_ndim,
            Int(py=arr.ctypes.data),
        ),
        TensorArg[dtype, True].from_write_tensor(
            output_tensor,
            output_ndim,
            Int(py=out_arr.ctypes.data),
        ),
    )
    var operation = FillKernel[dtype, backward](limit=Int(py=limit))
    var driver = GUFunc[FillKernel[dtype, backward]](operation)
    driver.execute(
        tensors,
        parsed[0],
        parsed[1],
        parsed[0],
        parsed[1],
        _policy(cfg),
    )
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
    return _apply_fill[dtype, False](
        arr, axes, out_arr, limit, threshold, "ffill"
    )


def bfill_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    limit: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    return _apply_fill[dtype, True](
        arr, axes, out_arr, limit, threshold, "bfill"
    )


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
    if Int(py=batch) == 0:
        return out_arr
    var policy = DispatchPolicy(Int(py=workers), Int(py=threshold))
    var operation = NanCovOp[dtype](
        Int(py=n_vars),
        Int(py=n_obs),
    )
    return _apply_matrix[dtype, NanCovOp[dtype]](
        arr,
        out_arr,
        Int(py=n_vars),
        Int(py=n_obs),
        policy,
        operation,
        "nancovmatrix",
    )


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
    if Int(py=batch) == 0:
        return out_arr
    var policy = DispatchPolicy(Int(py=workers), Int(py=threshold))
    var operation = NanCorrOp[dtype](
        Int(py=n_vars),
        Int(py=n_obs),
    )
    return _apply_matrix[dtype, NanCorrOp[dtype]](
        arr,
        out_arr,
        Int(py=n_vars),
        Int(py=n_obs),
        policy,
        operation,
        "nancorrmatrix",
    )


def nanquantile_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    quantiles_arr: PythonObject,
    out_arr: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    _validate_dtype[dtype](arr, "nanquantile")
    _validate_output_dtype[dtype](out_arr, "nanquantile")
    var input_tensor = read_tensor_from_numpy[dtype](arr)
    var output_tensor = write_tensor_from_numpy[dtype](out_arr)
    var input_ndim = Int(py=arr.ndim)
    var output_ndim = Int(py=out_arr.ndim)
    var t = _axes_from_py(axes, "nanquantile")
    var q_addr = Int(py=quantiles_arr.ctypes.data)
    var num_q = Int(py=quantiles_arr.__len__())
    var policy = _policy(cfg)
    var tensors = Tuple(
        TensorArg[dtype, False].from_read_tensor(
            input_tensor,
            input_ndim,
            Int(py=arr.ctypes.data),
        ),
        TensorArg[dtype, True].from_write_tensor(
            output_tensor,
            output_ndim,
            Int(py=out_arr.ctypes.data),
        ),
    )
    var operation = NanQuantileKernel[dtype](q_addr, num_q)
    var driver = GUFunc[NanQuantileKernel[dtype]](operation)
    var output_axes = TensorDimArray(fill=0)
    output_axes[0] = output_ndim - 1
    driver.execute(
        tensors,
        t[0],
        t[1],
        output_axes,
        1,
        policy,
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
