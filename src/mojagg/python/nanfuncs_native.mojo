"""Python bindings for the nanfuncs family.

The binding validates the Python boundary once, borrows NumPy input storage
into typed tensor views, plans output metadata, allocates NumPy from that
plan, and invokes the appropriate vectorization driver. The nanfuncs use the
combined-signature ``guvectorize`` driver. All axis traversal, scratch
materialization, and scheduling happen in Mojo.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.drivers.guvectorize import (
    AxisSpec,
    CoreSpec,
    CoreBindings,
    Dim,
    DispatchPolicy as VectorizeDispatchPolicy,
    GUFuncKernel,
    build_signature_plan,
    build_signature_plan_with_bindings,
    GUTensor,
    guvectorize,
)
from mojagg.nanfuncs.fill import FillKernel
from mojagg.nanfuncs.allnan import AllNan
from mojagg.nanfuncs.anynan import AnyNan
from mojagg.nanfuncs.nanargmax import NanArgMax
from mojagg.nanfuncs.nanargmin import NanArgMin
from mojagg.nanfuncs.nancount import NanCount

# Matrix kernels own their concrete operation types; nanmatrix provides the
# shared pairwise accumulator and execution driver.
from mojagg.nanfuncs.nancorrmatrix import NanCorrOp
from mojagg.nanfuncs.nancovmatrix import NanCovOp
from mojagg.nanfuncs.nanmax import NanMax
from mojagg.nanfuncs.nanmean import NanMean
from mojagg.nanfuncs.nanmin import NanMin
from mojagg.nanfuncs.nanprod import NanProd
from mojagg.nanfuncs.nanquantile import NanQuantileKernel, QUANTILE_DIM
from mojagg.nanfuncs.nansum import NanSum
from mojagg.nanfuncs.nanvar import NanVar
from mojagg.python.common import (
    axes_from_py,
    borrow_numpy_tensor,
    dispatch_policy_from_py,
    matrix_axes_from_py,
    validate_dtype,
)
from mojagg.python.signature_materialization import materialize_outputs


# --- N-D axis bindings --------------------------------------------------------
# One generic binding per op, instantiated per dtype at module registration.
# Contract: `axes` is a NORMALIZED Python tuple from the facade (deduped,
def _matrix_dimensions(
    arr: PythonObject, axes: PythonObject, op: String
) raises -> Tuple[Int, Int, Int]:
    var ndim = Int(py=arr.ndim)
    if ndim < 2:
        raise Error(op + " requires at least two dimensions")
    var parsed = matrix_axes_from_py(axes, ndim, op)
    var n_vars = Int(py=arr.shape[parsed[0]])
    var n_obs = Int(py=arr.shape[parsed[1]])
    var batch = 1
    for axis in range(ndim):
        if axis != parsed[0] and axis != parsed[1]:
            batch *= Int(py=arr.shape[axis])
    return (batch, n_vars, n_obs)


@always_inline
def _matrix_schedule(
    batch: Int,
    n_vars: Int,
    n_obs: Int,
    threshold: Int,
    workers: Int,
    parallel_min_groups: Int,
) -> Tuple[VectorizeDispatchPolicy, Int]:
    """Choose outer or pair-tile parallelism for one matrix call.

    Matrix work scales with the upper triangle, not with one input core.  The
    generic driver only sees ``n_vars * n_obs`` as its core length, so the
    matrix binding performs the work gate here and passes a zero driver
    threshold once the pair-work gate is satisfied.
    """

    var pair_count = n_vars * (n_vars + 1) // 2
    var pair_work = pair_count * n_obs
    var required_work = max(threshold, 0)
    # A batched call amortizes one worker-pool launch over every matrix.  Use
    # aggregate pair work for that path; a single matrix still gates on its own
    # pair work before enabling the inner tile pool.
    var dispatch_work = pair_work
    if batch > 1:
        dispatch_work = pair_work * batch
    if dispatch_work < required_work:
        return (VectorizeDispatchPolicy(1, 0, 1), 1)

    var groups = max(parallel_min_groups, 1)
    groups = min(groups, max(batch, 1))
    if batch > 1:
        return (
            VectorizeDispatchPolicy(workers, 0, groups),
            1,
        )

    var requested = workers if workers > 0 else 16
    return (VectorizeDispatchPolicy(1, 0, 1), requested)


def _apply_reduction[
    value_dtype: DType,
    out_dtype: DType,
    Op: GUFuncKernel,
](
    arr: PythonObject,
    axes: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run one scalar reduction through the guvectorize driver."""

    validate_dtype[value_dtype](arr, op_name)
    var input = borrow_numpy_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var parsed = axes_from_py(axes, op_name)
    var signature = Tuple(
        input,
        GUTensor[out_dtype, True, CoreSpec[]].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        parsed,
        AxisSpec.empty(),
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](
        operation,
        signature,
        plan,
        dispatch_policy_from_py(cfg),
    )
    return outputs[0]


def _apply_matrix[
    value_dtype: DType,
    Op: GUFuncKernel,
](
    arr: PythonObject,
    axes: PythonObject,
    policy: VectorizeDispatchPolicy,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run one matrix operation over two selected input core axes.

    The output core is always placed at the end of the execution array.  The
    selected input axes may occur in any physical order; the generic driver
    materializes a contiguous logical input span when needed.
    """

    validate_dtype[value_dtype](arr, op_name)
    var input_ndim = Int(py=arr.ndim)
    var input_axes = matrix_axes_from_py(axes, input_ndim, op_name)

    var input = borrow_numpy_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0], Dim[1]],
    ](arr)
    var output_axes = AxisSpec.empty()
    output_axes.count = 2
    output_axes[0] = input_ndim - 2
    output_axes[1] = input_ndim - 1
    var signature = Tuple(
        input,
        GUTensor[
            value_dtype,
            True,
            CoreSpec[Dim[0], Dim[0]],
        ].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        input_axes,
        output_axes,
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](
        operation,
        signature,
        plan,
        policy,
    )
    return outputs[0]


def allnan_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """Allnan over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.bool, AllNan[dtype]](
        arr, axes, threshold, AllNan[dtype](), "allnan"
    )


def anynan_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """AnyNan over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.bool, AnyNan[dtype]](
        arr, axes, threshold, AnyNan[dtype](), "anynan"
    )


def nansum_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    return _apply_reduction[dtype, dtype, NanSum[dtype]](
        arr,
        axes,
        threshold,
        NanSum[dtype](),
        "nansum",
    )


def nanmean_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanMean over axes, merging sum/count before final division."""
    return _apply_reduction[dtype, dtype, NanMean[dtype]](
        arr, axes, threshold, NanMean[dtype](), "nanmean"
    )


def nanprod_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanProd over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, dtype, NanProd[dtype]](
        arr, axes, threshold, NanProd[dtype](), "nanprod"
    )


def nanmin_binding[
    dtype: DType,
    result_dtype: DType,
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanMin over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, result_dtype, NanMin[dtype, result_dtype]](
        arr,
        axes,
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
    threshold: PythonObject,
) raises -> PythonObject:
    """NanMax over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, result_dtype, NanMax[dtype, result_dtype]](
        arr,
        axes,
        threshold,
        NanMax[dtype, result_dtype](),
        "nanmax",
    )


def nanargmin_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanArgMin over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.int64, NanArgMin[dtype]](
        arr, axes, threshold, NanArgMin[dtype](), "nanargmin"
    )


def nanargmax_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanArgMax over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.int64, NanArgMax[dtype]](
        arr, axes, threshold, NanArgMax[dtype](), "nanargmax"
    )


def nanvar_binding[
    dtype: DType,
    take_sqrt: Bool,
](
    arr: PythonObject,
    axes: PythonObject,
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
        threshold,
        kernel,
        op,
    )


def nancount_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    """NanCount over `axes` of an N-D array — one FFI call, zero copies."""
    return _apply_reduction[dtype, DType.int64, NanCount[dtype]](
        arr, axes, threshold, NanCount[dtype](), "nancount"
    )


def _apply_fill[
    dtype: DType, backward: Bool
](
    arr: PythonObject,
    axes: PythonObject,
    limit: PythonObject,
    cfg: PythonObject,
    op_name: String,
) raises -> PythonObject:
    validate_dtype[dtype](arr, op_name)
    var input = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var parsed = axes_from_py(axes, op_name)
    var operation = FillKernel[dtype, backward](limit=Int(py=limit))
    var signature = Tuple(
        input,
        GUTensor[dtype, True, CoreSpec[Dim[0]]].empty(),
    )
    var output_axes = AxisSpec.empty()
    output_axes.count = parsed.count
    for i in range(parsed.count):
        output_axes[i] = input.ndim - parsed.count + i
    var plan = build_signature_plan[FillKernel[dtype, backward]](
        signature,
        parsed,
        output_axes,
    )
    var outputs = materialize_outputs[FillKernel[dtype, backward]](signature)
    guvectorize[FillKernel[dtype, backward]](
        operation,
        signature,
        plan,
        dispatch_policy_from_py(cfg),
    )
    return outputs[0]


def ffill_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    limit: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    return _apply_fill[dtype, False](arr, axes, limit, threshold, "ffill")


def bfill_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    limit: PythonObject,
    threshold: PythonObject,
) raises -> PythonObject:
    return _apply_fill[dtype, True](arr, axes, limit, threshold, "bfill")


def nancovmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
    workers: PythonObject,
    parallel_min_groups: PythonObject,
) raises -> PythonObject:
    var dimensions = _matrix_dimensions(arr, axes, "nancovmatrix")
    var schedule = _matrix_schedule(
        dimensions[0],
        dimensions[1],
        dimensions[2],
        Int(py=threshold),
        Int(py=workers),
        Int(py=parallel_min_groups),
    )
    var operation = NanCovOp[dtype](
        dimensions[1],
        dimensions[2],
        schedule[1],
    )
    return _apply_matrix[dtype, NanCovOp[dtype]](
        arr,
        axes,
        schedule[0],
        operation,
        "nancovmatrix",
    )


def nancorrmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    threshold: PythonObject,
    workers: PythonObject,
    parallel_min_groups: PythonObject,
) raises -> PythonObject:
    var dimensions = _matrix_dimensions(arr, axes, "nancorrmatrix")
    var schedule = _matrix_schedule(
        dimensions[0],
        dimensions[1],
        dimensions[2],
        Int(py=threshold),
        Int(py=workers),
        Int(py=parallel_min_groups),
    )
    var operation = NanCorrOp[dtype](
        dimensions[1],
        dimensions[2],
        schedule[1],
    )
    return _apply_matrix[dtype, NanCorrOp[dtype]](
        arr,
        axes,
        schedule[0],
        operation,
        "nancorrmatrix",
    )


def nanquantile_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    quantiles_arr: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    validate_dtype[dtype](arr, "nanquantile")
    var input = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var parsed = axes_from_py(axes, "nanquantile")
    var output_ndim = input.ndim - parsed.count + 1
    var q_addr = Int(py=quantiles_arr.ctypes.data)
    var num_q = Int(py=quantiles_arr.__len__())
    var policy = dispatch_policy_from_py(cfg)
    var output_axes = AxisSpec.empty()
    output_axes.count = 1
    output_axes[0] = output_ndim - 1
    var bindings = CoreBindings.empty()
    bindings.bind(QUANTILE_DIM, num_q)
    var signature = Tuple(
        input,
        GUTensor[
            dtype,
            True,
            CoreSpec[Dim[QUANTILE_DIM]],
        ].empty(),
    )
    var plan = build_signature_plan_with_bindings[NanQuantileKernel[dtype]](
        signature,
        parsed,
        output_axes,
        bindings,
    )
    var outputs = materialize_outputs[NanQuantileKernel[dtype]](signature)
    var operation = NanQuantileKernel[dtype](q_addr, num_q)
    guvectorize[NanQuantileKernel[dtype]](
        operation,
        signature,
        plan,
        policy,
    )
    return outputs[0]


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

        return m.finalize()
    except e:
        abort(String("failed to create nanfuncs_native module: ", e))
