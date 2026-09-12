"""Python bindings for the grouped reduction family.

The facade resolves axes, broadcasts labels, initializes the result and any
operation workspace, and passes one normalized call into this module.  The
typed binding below only performs the NumPy boundary conversion and runs the
operation through the common GUFunc driver.
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
from mojagg.drivers.gufunc import (
    DispatchPolicy,
    GUFunc,
    GUFuncOperation,
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
from mojagg.python.tensor_view_binding import (
    read_tensor_from_numpy,
    write_tensor_from_numpy,
)


def _validate_dtype[dtype: DType](arr: PythonObject, op_name: String) raises:
    var expected = _dtype_name[dtype]()
    var actual = String(py=arr.dtype.name)
    if expected != actual:
        raise Error(
            op_name + ": expected dtype " + expected + ", got " + actual
        )


def _policy(cfg: PythonObject) raises -> DispatchPolicy:
    return DispatchPolicy(
        Int(py=cfg.threads),
        Int(py=cfg.parallel_threshold),
    )


def _apply_group[
    value_dtype: DType,
    label_dtype: DType,
    out_dtype: DType,
    aux1_dtype: DType,
    aux2_dtype: DType,
    aux_count: Int,
    Op: GUFuncOperation,
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

    var values_tensor = read_tensor_from_numpy[value_dtype](values)
    var labels_tensor = read_tensor_from_numpy[label_dtype](labels)
    var output_tensor = write_tensor_from_numpy[out_dtype](group_out)

    var core_rank = Int(py=axes.__len__())
    if core_rank < 1 or core_rank > MAX_RANK:
        raise Error(
            op_name
            + ": expected 1.."
            + String(MAX_RANK)
            + " grouped axes, got "
            + String(core_rank)
        )

    var input_axes = TensorDimArray(fill=0)
    for d in range(core_rank):
        input_axes[d] = Int(py=axes[d])

    var values_ndim = Int(py=values.ndim)
    var labels_ndim = Int(py=labels.ndim)
    var output_ndim = Int(py=group_out.ndim)

    var output_axes = TensorDimArray(fill=0)
    output_axes[0] = output_ndim - 1
    var policy = _policy(options[1])

    comptime if aux_count == 0:
        var tensors = Tuple(
            TensorArg[value_dtype, False].from_read_tensor(
                values_tensor,
                values_ndim,
                Int(py=values.ctypes.data),
            ),
            TensorArg[label_dtype, False].from_read_tensor(
                labels_tensor,
                labels_ndim,
                Int(py=labels.ctypes.data),
            ),
            TensorArg[out_dtype, True].from_write_tensor(
                output_tensor,
                output_ndim,
                Int(py=group_out.ctypes.data),
            ),
        )
        var driver = GUFunc[Op](operation)
        driver.execute(
            tensors,
            input_axes,
            core_rank,
            output_axes,
            1,
            policy,
        )
    elif aux_count == 1:
        var aux1 = auxiliaries[0]
        var aux1_tensor = write_tensor_from_numpy[aux1_dtype](aux1)
        var aux1_ndim = Int(py=aux1.ndim)
        var tensors = Tuple(
            TensorArg[value_dtype, False].from_read_tensor(
                values_tensor,
                values_ndim,
                Int(py=values.ctypes.data),
            ),
            TensorArg[label_dtype, False].from_read_tensor(
                labels_tensor,
                labels_ndim,
                Int(py=labels.ctypes.data),
            ),
            TensorArg[out_dtype, True].from_write_tensor(
                output_tensor,
                output_ndim,
                Int(py=group_out.ctypes.data),
            ),
            TensorArg[aux1_dtype, True].from_write_tensor(
                aux1_tensor,
                aux1_ndim,
                Int(py=aux1.ctypes.data),
            ),
        )
        var driver = GUFunc[Op](operation)
        driver.execute(
            tensors,
            input_axes,
            core_rank,
            output_axes,
            1,
            policy,
        )
    else:
        var aux1 = auxiliaries[0]
        var aux2 = auxiliaries[1]
        var aux1_tensor = write_tensor_from_numpy[aux1_dtype](aux1)
        var aux2_tensor = write_tensor_from_numpy[aux2_dtype](aux2)
        var aux1_ndim = Int(py=aux1.ndim)
        var aux2_ndim = Int(py=aux2.ndim)
        var tensors = Tuple(
            TensorArg[value_dtype, False].from_read_tensor(
                values_tensor,
                values_ndim,
                Int(py=values.ctypes.data),
            ),
            TensorArg[label_dtype, False].from_read_tensor(
                labels_tensor,
                labels_ndim,
                Int(py=labels.ctypes.data),
            ),
            TensorArg[out_dtype, True].from_write_tensor(
                output_tensor,
                output_ndim,
                Int(py=group_out.ctypes.data),
            ),
            TensorArg[aux1_dtype, True].from_write_tensor(
                aux1_tensor,
                aux1_ndim,
                Int(py=aux1.ctypes.data),
            ),
            TensorArg[aux2_dtype, True].from_write_tensor(
                aux2_tensor,
                aux2_ndim,
                Int(py=aux2.ctypes.data),
            ),
        )
        var driver = GUFunc[Op](operation)
        driver.execute(
            tensors,
            input_axes,
            core_rank,
            output_axes,
            1,
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
        m.def_function[group_nanvar_binding[DType.int64, DType.int64]](
            "group_nanvar_i64_i64"
        )
        m.def_function[group_nanvar_binding[DType.int64, DType.int32]](
            "group_nanvar_i64_i32"
        )
        m.def_function[group_nanvar_binding[DType.int32, DType.int64]](
            "group_nanvar_i32_i64"
        )
        m.def_function[group_nanvar_binding[DType.int32, DType.int32]](
            "group_nanvar_i32_i32"
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
        m.def_function[group_nanstd_binding[DType.int64, DType.int64]](
            "group_nanstd_i64_i64"
        )
        m.def_function[group_nanstd_binding[DType.int64, DType.int32]](
            "group_nanstd_i64_i32"
        )
        m.def_function[group_nanstd_binding[DType.int32, DType.int64]](
            "group_nanstd_i32_i64"
        )
        m.def_function[group_nanstd_binding[DType.int32, DType.int32]](
            "group_nanstd_i32_i32"
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
