"""Python bindings for the nanfuncs family.

The binding validates the Python boundary once, borrows NumPy storage into
typed tensor views, and invokes the appropriate vectorization driver.  The
nanfuncs use the combined-signature ``guvectorize`` driver.  All axis
traversal, scratch materialization, and scheduling happen in Mojo.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.core.numeric import _dtype_name
from mojagg.drivers.guvectorize import (
    AxisSpec,
    CoreSpec,
    CoreSpecProtocol,
    CoreBindings,
    DimArray as VectorizeDimArray,
    Dim,
    DispatchPolicy as VectorizeDispatchPolicy,
    GUFuncKernel,
    build_signature_with_bindings,
    build_signature,
    GUTensor,
    guvectorize,
    MAX_RANK,
    MAX_RANK as VectorizeMaxRank,
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


# --- N-D axis bindings --------------------------------------------------------
# One generic binding per op, instantiated per dtype at module registration.
# Contract: `axes` is a NORMALIZED Python tuple from the facade (deduped,
# negatives resolved, stride-sorted descending, k >= 1); `out_arr` is a
# preallocated numpy array of the op's result dtype; `threshold` is the
# resolved MojaggConfig fields or a legacy integer threshold.


@always_inline
def _cfg_params(cfg: PythonObject) raises -> Tuple[Int, Int, Int]:
    try:
        var t = Int(py=cfg.parallel_threshold)
        var w = Int(py=cfg.threads)
        var g = Int(py=cfg.parallel_min_groups)
        return (t, w, g)
    except:
        return (Int(py=cfg), 0, 1)


def _axes_from_py(
    axes: PythonObject, op: String
) raises -> Tuple[VectorizeDimArray, Int]:
    """Read a normalized axes tuple into a fixed-capacity stack array."""
    var k = Int(py=axes.__len__())
    if k < 1 or k > MAX_RANK:
        raise Error(
            op + ": expected 1.." + String(MAX_RANK) + " axes, got " + String(k)
        )
    var ax = VectorizeDimArray(fill=0)
    for i in range(k):
        ax[i] = Int(py=axes[i])
    return (ax^, k)


def _axis_spec_from_py(axes: PythonObject, op: String) raises -> AxisSpec:
    var parsed = _axes_from_py(axes, op)
    return AxisSpec(parsed[0].copy(), parsed[1])


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


def _guvectorize_tensor[
    dtype: DType,
    writable: Bool,
    core: CoreSpecProtocol,
](arr: PythonObject) raises -> GUTensor[dtype, writable, core]:
    """Borrow one NumPy array as a ``GUTensor`` descriptor."""

    var rank = Int(py=arr.ndim)
    if rank < 0 or rank > VectorizeMaxRank:
        raise Error("tensor rank exceeds guvectorize capacity")

    var shape = VectorizeDimArray(fill=1)
    var stride = VectorizeDimArray(fill=0)
    var itemsize = Int(py=arr.dtype.itemsize)
    if itemsize <= 0:
        raise Error("tensor itemsize must be positive")
    for axis in range(rank):
        shape[axis] = Int(py=arr.shape[axis])
        var byte_stride = Int(py=arr.strides[axis])
        if byte_stride % itemsize != 0:
            raise Error("array stride is not divisible by its itemsize")
        stride[axis] = byte_stride // itemsize

    return GUTensor[dtype, writable, core].borrow(
        Int(py=arr.ctypes.data),
        shape,
        stride,
        rank,
    )


def allocate_signature[
    dtype: DType,
    core: CoreSpecProtocol,
](
    planned: GUTensor[dtype, True, core],
    out_arr: PythonObject,
    op: String,
) raises -> GUTensor[
    dtype,
    True,
    core,
]:
    """Bind one Python-owned NumPy output to a planned signature tensor.

    Allocation remains in the Python facade.  This generic boundary helper
    validates the shape/layout described by ``build_signature`` and installs
    the borrowed address without retaining a Python owner in Mojo.
    """

    var output = planned.copy()
    var rank = Int(py=out_arr.ndim)
    if rank != output.ndim:
        raise Error(op + " output rank does not match its signature")

    var itemsize = Int(py=out_arr.dtype.itemsize)
    var has_zero_extent = False
    for axis in range(rank):
        var extent = Int(py=out_arr.shape[axis])
        if extent != output.shape[axis]:
            raise Error(op + " output shape does not match its signature")
        if extent == 0:
            has_zero_extent = True
        var byte_stride = Int(py=out_arr.strides[axis])
        if byte_stride % itemsize != 0:
            raise Error(op + " output stride is not divisible by its itemsize")
        if (
            not has_zero_extent
            and byte_stride // itemsize != output.stride[axis]
        ):
            raise Error(op + " output must be C-contiguous")

    output.bind_address(Int(py=out_arr.ctypes.data), output.length)
    return output^


def _vectorize_policy(cfg: PythonObject) raises -> VectorizeDispatchPolicy:
    var values = _cfg_params(cfg)
    return VectorizeDispatchPolicy(values[1], values[0], values[2])


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
    out_arr: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run one scalar reduction through the guvectorize driver."""

    _validate_dtype[value_dtype](arr, op_name)
    _validate_output_dtype[out_dtype](out_arr, op_name)
    var input = _guvectorize_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var parsed = _axis_spec_from_py(axes, op_name)
    var template = Tuple(
        input,
        GUTensor[
            out_dtype,
            True,
            CoreSpec[],
        ].empty(),
    )
    var planned = build_signature[Op](
        template,
        parsed,
        AxisSpec.empty(),
    )
    var input_view, planned_output = planned
    var output_view = allocate_signature[
        out_dtype,
        CoreSpec[],
    ](planned_output, out_arr, op_name)
    var signature = Tuple(input_view, output_view)
    guvectorize[Op](
        operation,
        signature,
        parsed,
        AxisSpec.empty(),
        _vectorize_policy(cfg),
    )
    return out_arr


def _apply_matrix[
    value_dtype: DType,
    Op: GUFuncKernel,
](
    arr: PythonObject,
    out_arr: PythonObject,
    n_vars: Int,
    n_obs: Int,
    policy: VectorizeDispatchPolicy,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run one matrix operation over the two trailing core axes."""

    _validate_dtype[value_dtype](arr, op_name)
    _validate_output_dtype[value_dtype](out_arr, op_name)
    var input_ndim = Int(py=arr.ndim)
    var output_ndim = Int(py=out_arr.ndim)
    if input_ndim < 2 or output_ndim < 2:
        raise Error(op_name + " requires at least two dimensions")

    var input = _guvectorize_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0], Dim[1]],
    ](arr)
    var output = _guvectorize_tensor[
        value_dtype,
        True,
        CoreSpec[Dim[0], Dim[0]],
    ](out_arr)
    var input_axes = AxisSpec.empty()
    input_axes.count = 2
    input_axes[0] = input_ndim - 2
    input_axes[1] = input_ndim - 1
    var output_axes = AxisSpec.empty()
    output_axes.count = 2
    output_axes[0] = output_ndim - 2
    output_axes[1] = output_ndim - 1
    guvectorize[Op](
        operation,
        Tuple(input, output),
        input_axes,
        output_axes,
        policy,
    )
    return out_arr


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
    var input = _guvectorize_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var output = _guvectorize_tensor[
        dtype,
        True,
        CoreSpec[Dim[0]],
    ](out_arr)
    var parsed = _axis_spec_from_py(axes, op_name)
    var operation = FillKernel[dtype, backward](limit=Int(py=limit))
    guvectorize[FillKernel[dtype, backward]](
        operation,
        Tuple(input, output),
        parsed,
        parsed,
        _vectorize_policy(cfg),
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
    parallel_min_groups: PythonObject,
) raises -> PythonObject:
    var batch_count = Int(py=batch)
    var variables = Int(py=n_vars)
    var observations = Int(py=n_obs)
    if batch_count == 0:
        return out_arr
    var schedule = _matrix_schedule(
        batch_count,
        variables,
        observations,
        Int(py=threshold),
        Int(py=workers),
        Int(py=parallel_min_groups),
    )
    var operation = NanCovOp[dtype](
        variables,
        observations,
        schedule[1],
    )
    return _apply_matrix[dtype, NanCovOp[dtype]](
        arr,
        out_arr,
        variables,
        observations,
        schedule[0],
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
    parallel_min_groups: PythonObject,
) raises -> PythonObject:
    var batch_count = Int(py=batch)
    var variables = Int(py=n_vars)
    var observations = Int(py=n_obs)
    if batch_count == 0:
        return out_arr
    var schedule = _matrix_schedule(
        batch_count,
        variables,
        observations,
        Int(py=threshold),
        Int(py=workers),
        Int(py=parallel_min_groups),
    )
    var operation = NanCorrOp[dtype](
        variables,
        observations,
        schedule[1],
    )
    return _apply_matrix[dtype, NanCorrOp[dtype]](
        arr,
        out_arr,
        variables,
        observations,
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
    out_arr: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    _validate_dtype[dtype](arr, "nanquantile")
    _validate_output_dtype[dtype](out_arr, "nanquantile")
    var output_ndim = Int(py=out_arr.ndim)
    var input = _guvectorize_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var parsed = _axis_spec_from_py(axes, "nanquantile")
    var q_addr = Int(py=quantiles_arr.ctypes.data)
    var num_q = Int(py=quantiles_arr.__len__())
    var policy = _vectorize_policy(cfg)
    var output_axes = AxisSpec.empty()
    output_axes.count = 1
    output_axes[0] = output_ndim - 1
    var bindings = CoreBindings.empty()
    bindings.bind(QUANTILE_DIM, num_q)
    var template = Tuple(
        input,
        GUTensor[
            dtype,
            True,
            CoreSpec[Dim[QUANTILE_DIM]],
        ].empty(),
    )
    var planned = build_signature_with_bindings[NanQuantileKernel[dtype]](
        template,
        parsed,
        output_axes,
        bindings,
    )
    var input_view, planned_output = planned
    var output_view = allocate_signature[
        dtype,
        CoreSpec[Dim[QUANTILE_DIM]],
    ](planned_output, out_arr, "nanquantile")
    var operation = NanQuantileKernel[dtype](q_addr, num_q)
    guvectorize[NanQuantileKernel[dtype]](
        operation,
        Tuple(input_view, output_view),
        parsed,
        output_axes,
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

        return m.finalize()
    except e:
        abort(String("failed to create nanfuncs_native module: ", e))
