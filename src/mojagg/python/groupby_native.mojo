"""Python bindings for the grouped reduction family.

The facade resolves axes, broadcasts labels, initializes the result and any
operation workspace, and passes one normalized call into this module.  The
typed binding below only performs the NumPy boundary conversion and runs the
operation through the common guvectorize driver.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.core.numeric import _dtype_name
from mojagg.drivers.guvectorize import (
    AxisSpec,
    CoreSpec,
    CoreSpecProtocol,
    Dim,
    DimArray as VectorizeDimArray,
    DispatchPolicy,
    GUTensor,
    GUFuncKernel,
    guvectorize,
    MAX_RANK,
    MAX_RANK as VectorizeMaxRank,
)
from mojagg.groupby.nananyall import GroupNanAnyAll
from mojagg.groupby.nanargminmax import GroupNanArgMinMax
from mojagg.groupby.nancount import GroupNanCount
from mojagg.groupby.nanfirstlast import GroupNanFirst, GroupNanLast
from mojagg.groupby.nanmean import GroupNanMean
from mojagg.groupby.nanminmax import GroupNanMinMax
from mojagg.groupby.nanprod import GroupNanProd
from mojagg.groupby.nansum import GroupNanSum
from mojagg.groupby.nanvarstd import GroupNanVarStd


def _validate_dtype[dtype: DType](arr: PythonObject, op_name: String) raises:
    var expected = _dtype_name[dtype]()
    var actual = String(py=arr.dtype.name)
    if expected != actual:
        raise Error(
            op_name + ": expected dtype " + expected + ", got " + actual
        )


def _policy(cfg: PythonObject) raises -> DispatchPolicy:
    return DispatchPolicy(
        workers=Int(py=cfg.threads),
        parallel_threshold=Int(py=cfg.parallel_threshold),
        parallel_min_groups=Int(py=cfg.parallel_min_groups),
    )


def _guvectorize_tensor[
    dtype: DType,
    writable: Bool,
    core: CoreSpecProtocol,
](arr: PythonObject) raises -> GUTensor[dtype, writable, core]:
    """Borrow a NumPy array as a typed guvectorize tensor descriptor."""

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


def _apply_group[
    value_dtype: DType,
    label_dtype: DType,
    out_dtype: DType,
    aux1_dtype: DType,
    aux2_dtype: DType,
    aux_count: Int,
    Op: GUFuncKernel,
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run one grouped operation over all normalized outer slices."""

    _validate_dtype[value_dtype](values, op_name)
    _validate_dtype[label_dtype](labels, op_name + " labels")
    _validate_dtype[out_dtype](group_out, op_name + " output")

    var auxiliaries = options[0]
    comptime if aux_count > 0:
        _validate_dtype[aux1_dtype](auxiliaries[0], op_name + " auxiliary 0")
    comptime if aux_count > 1:
        _validate_dtype[aux2_dtype](auxiliaries[1], op_name + " auxiliary 1")

    var core_rank = Int(py=axes.__len__())
    if core_rank < 1 or core_rank > VectorizeMaxRank:
        raise Error(
            op_name
            + ": expected 1.."
            + String(VectorizeMaxRank)
            + " grouped axes, got "
            + String(core_rank)
        )

    var values_ndim = Int(py=values.ndim)
    var labels_ndim = Int(py=labels.ndim)
    if labels_ndim != values_ndim:
        raise Error(op_name + ": values and labels must have the same shape")
    for axis in range(values_ndim):
        if Int(py=values.shape[axis]) != Int(py=labels.shape[axis]):
            raise Error(
                op_name + ": values and labels must have the same shape"
            )

    var input_axes = VectorizeDimArray(fill=0)
    for d in range(core_rank):
        input_axes[d] = Int(py=axes[d])

    var output_ndim = Int(py=group_out.ndim)

    var input_axis_spec = AxisSpec(input_axes.copy(), core_rank)
    var output_axis_spec = AxisSpec.empty()
    output_axis_spec.count = 1
    output_axis_spec[0] = output_ndim - 1
    var policy = _policy(options[1])

    var values_tensor = _guvectorize_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0]],
    ](values)
    var labels_tensor = _guvectorize_tensor[
        label_dtype,
        False,
        CoreSpec[Dim[0]],
    ](labels)
    var output_tensor = _guvectorize_tensor[
        out_dtype,
        True,
        CoreSpec[Dim[1]],
    ](group_out)

    comptime if aux_count == 0:
        var tensors = Tuple(
            values_tensor,
            labels_tensor,
            output_tensor,
        )
        guvectorize[Op](
            operation,
            tensors,
            input_axis_spec,
            output_axis_spec,
            policy,
        )
    elif aux_count == 1:
        var aux1 = auxiliaries[0]
        var aux1_tensor = _guvectorize_tensor[
            aux1_dtype,
            True,
            CoreSpec[Dim[1]],
        ](aux1)
        var tensors = Tuple(
            values_tensor,
            labels_tensor,
            output_tensor,
            aux1_tensor,
        )
        guvectorize[Op](
            operation,
            tensors,
            input_axis_spec,
            output_axis_spec,
            policy,
        )
    else:
        var aux1 = auxiliaries[0]
        var aux2 = auxiliaries[1]
        var aux1_tensor = _guvectorize_tensor[
            aux1_dtype,
            True,
            CoreSpec[Dim[1]],
        ](aux1)
        var aux2_tensor = _guvectorize_tensor[
            aux2_dtype,
            True,
            CoreSpec[Dim[1]],
        ](aux2)
        var tensors = Tuple(
            values_tensor,
            labels_tensor,
            output_tensor,
            aux1_tensor,
            aux2_tensor,
        )
        guvectorize[Op](
            operation,
            tensors,
            input_axis_spec,
            output_axis_spec,
            policy,
        )
    return group_out


def group_nansum_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.float64,
        DType.float64,
        0,
        GroupNanSum[value_dtype, label_dtype, 1],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanSum[value_dtype, label_dtype, 1](),
        "group_nansum",
    )


def group_nanmean_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.int64,
        DType.float64,
        1,
        GroupNanMean[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanMean[value_dtype, label_dtype](),
        "group_nanmean",
    )


def group_nanprod_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.float64,
        DType.float64,
        0,
        GroupNanProd[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanProd[value_dtype, label_dtype](),
        "group_nanprod",
    )


def group_nancount_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.float64,
        DType.float64,
        0,
        GroupNanCount[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanCount[value_dtype, label_dtype](),
        "group_nancount",
    )


def group_nanmin_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.int64,
        DType.float64,
        1,
        GroupNanMinMax[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanMinMax[value_dtype, label_dtype, False](),
        "group_nanmin",
    )


def group_nanmax_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.int64,
        DType.float64,
        1,
        GroupNanMinMax[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanMinMax[value_dtype, label_dtype, True](),
        "group_nanmax",
    )


def group_nanargmin_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        value_dtype,
        DType.int64,
        2,
        GroupNanArgMinMax[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanArgMinMax[value_dtype, label_dtype, False](),
        "group_nanargmin",
    )


def group_nanargmax_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        value_dtype,
        DType.int64,
        2,
        GroupNanArgMinMax[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanArgMinMax[value_dtype, label_dtype, True](),
        "group_nanargmax",
    )


def group_nanfirst_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.int64,
        DType.float64,
        1,
        GroupNanFirst[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanFirst[value_dtype, label_dtype](),
        "group_nanfirst",
    )


def group_nanlast_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.int64,
        DType.float64,
        1,
        GroupNanLast[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanLast[value_dtype, label_dtype](),
        "group_nanlast",
    )


def group_nanany_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.float64,
        DType.float64,
        0,
        GroupNanAnyAll[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanAnyAll[value_dtype, label_dtype, False](),
        "group_nanany",
    )


def group_nanall_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.float64,
        DType.float64,
        0,
        GroupNanAnyAll[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanAnyAll[value_dtype, label_dtype, True](),
        "group_nanall",
    )


def group_nanvar_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    var ddof = Int(py=options[2])
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        value_dtype,
        DType.int64,
        2,
        GroupNanVarStd[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanVarStd[value_dtype, label_dtype, False](ddof),
        "group_nanvar",
    )


def group_nanstd_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    var ddof = Int(py=options[2])
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        value_dtype,
        DType.int64,
        2,
        GroupNanVarStd[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanVarStd[value_dtype, label_dtype, True](ddof),
        "group_nanstd",
    )


def group_nansum_of_squares_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    group_out: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group[
        value_dtype,
        label_dtype,
        value_dtype,
        DType.float64,
        DType.float64,
        0,
        GroupNanSum[value_dtype, label_dtype, 2],
    ](
        values,
        labels,
        axes,
        group_out,
        options,
        GroupNanSum[value_dtype, label_dtype, 2](),
        "group_nansum_of_squares",
    )


@export
def PyInit_groupby_native() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("groupby_native")

        m.def_function[group_nansum_binding[DType.float64, DType.int64]](
            "group_nansum_f64_i64"
        )
        m.def_function[group_nansum_binding[DType.float64, DType.int32]](
            "group_nansum_f64_i32"
        )
        m.def_function[group_nansum_binding[DType.float32, DType.int64]](
            "group_nansum_f32_i64"
        )
        m.def_function[group_nansum_binding[DType.float32, DType.int32]](
            "group_nansum_f32_i32"
        )
        m.def_function[group_nansum_binding[DType.int64, DType.int64]](
            "group_nansum_i64_i64"
        )
        m.def_function[group_nansum_binding[DType.int64, DType.int32]](
            "group_nansum_i64_i32"
        )
        m.def_function[group_nansum_binding[DType.int32, DType.int64]](
            "group_nansum_i32_i64"
        )
        m.def_function[group_nansum_binding[DType.int32, DType.int32]](
            "group_nansum_i32_i32"
        )

        m.def_function[group_nanmean_binding[DType.float64, DType.int64]](
            "group_nanmean_f64_i64"
        )
        m.def_function[group_nanmean_binding[DType.float64, DType.int32]](
            "group_nanmean_f64_i32"
        )
        m.def_function[group_nanmean_binding[DType.float32, DType.int64]](
            "group_nanmean_f32_i64"
        )
        m.def_function[group_nanmean_binding[DType.float32, DType.int32]](
            "group_nanmean_f32_i32"
        )
        m.def_function[group_nanmean_binding[DType.int64, DType.int64]](
            "group_nanmean_i64_i64"
        )
        m.def_function[group_nanmean_binding[DType.int64, DType.int32]](
            "group_nanmean_i64_i32"
        )
        m.def_function[group_nanmean_binding[DType.int32, DType.int64]](
            "group_nanmean_i32_i64"
        )
        m.def_function[group_nanmean_binding[DType.int32, DType.int32]](
            "group_nanmean_i32_i32"
        )

        m.def_function[group_nanprod_binding[DType.float64, DType.int64]](
            "group_nanprod_f64_i64"
        )
        m.def_function[group_nanprod_binding[DType.float64, DType.int32]](
            "group_nanprod_f64_i32"
        )
        m.def_function[group_nanprod_binding[DType.float32, DType.int64]](
            "group_nanprod_f32_i64"
        )
        m.def_function[group_nanprod_binding[DType.float32, DType.int32]](
            "group_nanprod_f32_i32"
        )
        m.def_function[group_nanprod_binding[DType.int64, DType.int64]](
            "group_nanprod_i64_i64"
        )
        m.def_function[group_nanprod_binding[DType.int64, DType.int32]](
            "group_nanprod_i64_i32"
        )
        m.def_function[group_nanprod_binding[DType.int32, DType.int64]](
            "group_nanprod_i32_i64"
        )
        m.def_function[group_nanprod_binding[DType.int32, DType.int32]](
            "group_nanprod_i32_i32"
        )

        m.def_function[group_nancount_binding[DType.float64, DType.int64]](
            "group_nancount_f64_i64"
        )
        m.def_function[group_nancount_binding[DType.float64, DType.int32]](
            "group_nancount_f64_i32"
        )
        m.def_function[group_nancount_binding[DType.float32, DType.int64]](
            "group_nancount_f32_i64"
        )
        m.def_function[group_nancount_binding[DType.float32, DType.int32]](
            "group_nancount_f32_i32"
        )
        m.def_function[group_nancount_binding[DType.int64, DType.int64]](
            "group_nancount_i64_i64"
        )
        m.def_function[group_nancount_binding[DType.int64, DType.int32]](
            "group_nancount_i64_i32"
        )
        m.def_function[group_nancount_binding[DType.int32, DType.int64]](
            "group_nancount_i32_i64"
        )
        m.def_function[group_nancount_binding[DType.int32, DType.int32]](
            "group_nancount_i32_i32"
        )

        m.def_function[group_nanmin_binding[DType.float64, DType.int64]](
            "group_nanmin_f64_i64"
        )
        m.def_function[group_nanmin_binding[DType.float64, DType.int32]](
            "group_nanmin_f64_i32"
        )
        m.def_function[group_nanmin_binding[DType.float32, DType.int64]](
            "group_nanmin_f32_i64"
        )
        m.def_function[group_nanmin_binding[DType.float32, DType.int32]](
            "group_nanmin_f32_i32"
        )
        m.def_function[group_nanmin_binding[DType.int64, DType.int64]](
            "group_nanmin_i64_i64"
        )
        m.def_function[group_nanmin_binding[DType.int64, DType.int32]](
            "group_nanmin_i64_i32"
        )
        m.def_function[group_nanmin_binding[DType.int32, DType.int64]](
            "group_nanmin_i32_i64"
        )
        m.def_function[group_nanmin_binding[DType.int32, DType.int32]](
            "group_nanmin_i32_i32"
        )

        m.def_function[group_nanmax_binding[DType.float64, DType.int64]](
            "group_nanmax_f64_i64"
        )
        m.def_function[group_nanmax_binding[DType.float64, DType.int32]](
            "group_nanmax_f64_i32"
        )
        m.def_function[group_nanmax_binding[DType.float32, DType.int64]](
            "group_nanmax_f32_i64"
        )
        m.def_function[group_nanmax_binding[DType.float32, DType.int32]](
            "group_nanmax_f32_i32"
        )
        m.def_function[group_nanmax_binding[DType.int64, DType.int64]](
            "group_nanmax_i64_i64"
        )
        m.def_function[group_nanmax_binding[DType.int64, DType.int32]](
            "group_nanmax_i64_i32"
        )
        m.def_function[group_nanmax_binding[DType.int32, DType.int64]](
            "group_nanmax_i32_i64"
        )
        m.def_function[group_nanmax_binding[DType.int32, DType.int32]](
            "group_nanmax_i32_i32"
        )

        m.def_function[group_nanargmin_binding[DType.float64, DType.int64]](
            "group_nanargmin_f64_i64"
        )
        m.def_function[group_nanargmin_binding[DType.float64, DType.int32]](
            "group_nanargmin_f64_i32"
        )
        m.def_function[group_nanargmin_binding[DType.float32, DType.int64]](
            "group_nanargmin_f32_i64"
        )
        m.def_function[group_nanargmin_binding[DType.float32, DType.int32]](
            "group_nanargmin_f32_i32"
        )
        m.def_function[group_nanargmin_binding[DType.int64, DType.int64]](
            "group_nanargmin_i64_i64"
        )
        m.def_function[group_nanargmin_binding[DType.int64, DType.int32]](
            "group_nanargmin_i64_i32"
        )
        m.def_function[group_nanargmin_binding[DType.int32, DType.int64]](
            "group_nanargmin_i32_i64"
        )
        m.def_function[group_nanargmin_binding[DType.int32, DType.int32]](
            "group_nanargmin_i32_i32"
        )

        m.def_function[group_nanargmax_binding[DType.float64, DType.int64]](
            "group_nanargmax_f64_i64"
        )
        m.def_function[group_nanargmax_binding[DType.float64, DType.int32]](
            "group_nanargmax_f64_i32"
        )
        m.def_function[group_nanargmax_binding[DType.float32, DType.int64]](
            "group_nanargmax_f32_i64"
        )
        m.def_function[group_nanargmax_binding[DType.float32, DType.int32]](
            "group_nanargmax_f32_i32"
        )
        m.def_function[group_nanargmax_binding[DType.int64, DType.int64]](
            "group_nanargmax_i64_i64"
        )
        m.def_function[group_nanargmax_binding[DType.int64, DType.int32]](
            "group_nanargmax_i64_i32"
        )
        m.def_function[group_nanargmax_binding[DType.int32, DType.int64]](
            "group_nanargmax_i32_i64"
        )
        m.def_function[group_nanargmax_binding[DType.int32, DType.int32]](
            "group_nanargmax_i32_i32"
        )

        m.def_function[group_nanfirst_binding[DType.float64, DType.int64]](
            "group_nanfirst_f64_i64"
        )
        m.def_function[group_nanfirst_binding[DType.float64, DType.int32]](
            "group_nanfirst_f64_i32"
        )
        m.def_function[group_nanfirst_binding[DType.float32, DType.int64]](
            "group_nanfirst_f32_i64"
        )
        m.def_function[group_nanfirst_binding[DType.float32, DType.int32]](
            "group_nanfirst_f32_i32"
        )
        m.def_function[group_nanfirst_binding[DType.int64, DType.int64]](
            "group_nanfirst_i64_i64"
        )
        m.def_function[group_nanfirst_binding[DType.int64, DType.int32]](
            "group_nanfirst_i64_i32"
        )
        m.def_function[group_nanfirst_binding[DType.int32, DType.int64]](
            "group_nanfirst_i32_i64"
        )
        m.def_function[group_nanfirst_binding[DType.int32, DType.int32]](
            "group_nanfirst_i32_i32"
        )

        m.def_function[group_nanlast_binding[DType.float64, DType.int64]](
            "group_nanlast_f64_i64"
        )
        m.def_function[group_nanlast_binding[DType.float64, DType.int32]](
            "group_nanlast_f64_i32"
        )
        m.def_function[group_nanlast_binding[DType.float32, DType.int64]](
            "group_nanlast_f32_i64"
        )
        m.def_function[group_nanlast_binding[DType.float32, DType.int32]](
            "group_nanlast_f32_i32"
        )
        m.def_function[group_nanlast_binding[DType.int64, DType.int64]](
            "group_nanlast_i64_i64"
        )
        m.def_function[group_nanlast_binding[DType.int64, DType.int32]](
            "group_nanlast_i64_i32"
        )
        m.def_function[group_nanlast_binding[DType.int32, DType.int64]](
            "group_nanlast_i32_i64"
        )
        m.def_function[group_nanlast_binding[DType.int32, DType.int32]](
            "group_nanlast_i32_i32"
        )

        m.def_function[group_nanany_binding[DType.float64, DType.int64]](
            "group_nanany_f64_i64"
        )
        m.def_function[group_nanany_binding[DType.float64, DType.int32]](
            "group_nanany_f64_i32"
        )
        m.def_function[group_nanany_binding[DType.float32, DType.int64]](
            "group_nanany_f32_i64"
        )
        m.def_function[group_nanany_binding[DType.float32, DType.int32]](
            "group_nanany_f32_i32"
        )
        m.def_function[group_nanany_binding[DType.int64, DType.int64]](
            "group_nanany_i64_i64"
        )
        m.def_function[group_nanany_binding[DType.int64, DType.int32]](
            "group_nanany_i64_i32"
        )
        m.def_function[group_nanany_binding[DType.int32, DType.int64]](
            "group_nanany_i32_i64"
        )
        m.def_function[group_nanany_binding[DType.int32, DType.int32]](
            "group_nanany_i32_i32"
        )

        m.def_function[group_nanall_binding[DType.float64, DType.int64]](
            "group_nanall_f64_i64"
        )
        m.def_function[group_nanall_binding[DType.float64, DType.int32]](
            "group_nanall_f64_i32"
        )
        m.def_function[group_nanall_binding[DType.float32, DType.int64]](
            "group_nanall_f32_i64"
        )
        m.def_function[group_nanall_binding[DType.float32, DType.int32]](
            "group_nanall_f32_i32"
        )
        m.def_function[group_nanall_binding[DType.int64, DType.int64]](
            "group_nanall_i64_i64"
        )
        m.def_function[group_nanall_binding[DType.int64, DType.int32]](
            "group_nanall_i64_i32"
        )
        m.def_function[group_nanall_binding[DType.int32, DType.int64]](
            "group_nanall_i32_i64"
        )
        m.def_function[group_nanall_binding[DType.int32, DType.int32]](
            "group_nanall_i32_i32"
        )

        m.def_function[group_nanvar_binding[DType.float64, DType.int64]](
            "group_nanvar_f64_i64"
        )
        m.def_function[group_nanvar_binding[DType.float64, DType.int32]](
            "group_nanvar_f64_i32"
        )
        m.def_function[group_nanvar_binding[DType.float32, DType.int64]](
            "group_nanvar_f32_i64"
        )
        m.def_function[group_nanvar_binding[DType.float32, DType.int32]](
            "group_nanvar_f32_i32"
        )
        m.def_function[group_nanstd_binding[DType.float64, DType.int64]](
            "group_nanstd_f64_i64"
        )
        m.def_function[group_nanstd_binding[DType.float64, DType.int32]](
            "group_nanstd_f64_i32"
        )
        m.def_function[group_nanstd_binding[DType.float32, DType.int64]](
            "group_nanstd_f32_i64"
        )
        m.def_function[group_nanstd_binding[DType.float32, DType.int32]](
            "group_nanstd_f32_i32"
        )
        m.def_function[
            group_nansum_of_squares_binding[DType.float64, DType.int64]
        ]("group_nansum_of_squares_f64_i64")
        m.def_function[
            group_nansum_of_squares_binding[DType.float64, DType.int32]
        ]("group_nansum_of_squares_f64_i32")
        m.def_function[
            group_nansum_of_squares_binding[DType.float32, DType.int64]
        ]("group_nansum_of_squares_f32_i64")
        m.def_function[
            group_nansum_of_squares_binding[DType.float32, DType.int32]
        ]("group_nansum_of_squares_f32_i32")
        m.def_function[
            group_nansum_of_squares_binding[DType.int64, DType.int64]
        ]("group_nansum_of_squares_i64_i64")
        m.def_function[
            group_nansum_of_squares_binding[DType.int64, DType.int32]
        ]("group_nansum_of_squares_i64_i32")
        m.def_function[
            group_nansum_of_squares_binding[DType.int32, DType.int64]
        ]("group_nansum_of_squares_i32_i64")
        m.def_function[
            group_nansum_of_squares_binding[DType.int32, DType.int32]
        ]("group_nansum_of_squares_i32_i32")

        return m.finalize()
    except e:
        abort(String("failed to create groupby_native module: ", e))
