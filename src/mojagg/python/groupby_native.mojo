"""Python bindings for the grouped reduction family.

The facade resolves public axes, broadcasts labels, and determines the dense
group count. This binding builds and materializes the complete operation
signature, initializes result/workspace outputs, and runs the operation.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    AnyGUTensor,
    AxisSpec,
    CoreBindings,
    CoreSpec,
    CoreSpecProtocol,
    Dim,
    GUTensor,
    build_signature_plan_with_bindings,
    guvectorize,
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
from mojagg.groupby.group_kernel import (
    GROUP_INIT_NAN_OR_ZERO,
    GROUP_INIT_ONE,
    GROUP_INIT_ZERO,
    GroupKernel,
)
from mojagg.python.common import (
    axes_from_py,
    borrow_numpy_tensor,
    dispatch_policy_from_py,
    validate_dtype,
)
from mojagg.python.signature_materialization import materialize_outputs


def _fill_zero[
    dtype: DType, core: CoreSpecProtocol
](mut output: GUTensor[dtype, True, core],):
    var values = output.write_span()
    for i in range(len(values)):
        values[i] = Scalar[dtype](0)


def _fill_one[
    dtype: DType, core: CoreSpecProtocol
](mut output: GUTensor[dtype, True, core],):
    var values = output.write_span()
    for i in range(len(values)):
        values[i] = Scalar[dtype](1)


def _fill_nan[
    dtype: DType, core: CoreSpecProtocol
](mut output: GUTensor[dtype, True, core],):
    var values = output.write_span()
    var nan_value = nan_or_zero[dtype]()
    for i in range(len(values)):
        values[i] = nan_value


def _initialize_output[
    dtype: DType,
    core: CoreSpecProtocol,
](mut output: GUTensor[dtype, True, core], init_kind: Int):
    """Apply one binding-selected group output identity."""

    if init_kind == GROUP_INIT_ONE:
        _fill_one[dtype, core](output)
    elif init_kind == GROUP_INIT_NAN_OR_ZERO:
        comptime if dtype.is_floating_point():
            _fill_nan[dtype, core](output)
        else:
            _fill_zero[dtype, core](output)
    else:
        _fill_zero[dtype, core](output)


def _initialize_group_outputs[
    *Args: AnyGUTensor,
](mut signature: Tuple[*Args], initializers: List[Int]):
    """Initialize every writable slot declared by a grouped signature."""

    comptime for i in range(len(Args)):
        comptime if Args[i].is_output:
            var output = rebind[
                GUTensor[Args[i].dtype, True, Args[i].core_spec]
            ](signature[i])
            # Group signatures place their inputs in slots zero and one, and
            # their outputs consecutively after them. The output flag still
            # determines which descriptors are initialized.
            _initialize_output[Args[i].dtype, Args[i].core_spec](
                output,
                initializers[i - 2],
            )


def _execute_group[
    Op: GroupKernel,
    *Args: AnyGUTensor,
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
    mut signature: Tuple[*Args],
    initializers: List[Int],
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    comptime assert Op.Signature == Tuple[*Args]
    var input_axis_spec = axes_from_py(axes, op_name)
    var core_rank = input_axis_spec.count

    var values_ndim = Int(py=values.ndim)
    var labels_ndim = Int(py=labels.ndim)
    if labels_ndim != values_ndim:
        raise Error(op_name + ": values and labels must have the same shape")
    for axis in range(values_ndim):
        if Int(py=values.shape[axis]) != Int(py=labels.shape[axis]):
            raise Error(
                op_name + ": values and labels must have the same shape"
            )

    var output_axis_spec = AxisSpec.empty()
    output_axis_spec.count = 1
    output_axis_spec[0] = values_ndim - core_rank
    var policy = dispatch_policy_from_py(options[0])
    var bindings = CoreBindings.empty()
    bindings.bind(1, Int(py=num_labels))
    var plan = build_signature_plan_with_bindings[Op](
        signature,
        input_axis_spec,
        output_axis_spec,
        bindings,
    )
    var outputs = materialize_outputs[Op](signature)
    _initialize_group_outputs(signature, initializers)
    guvectorize[Op](operation, signature, plan, policy)
    return outputs[0]


def _apply_group_one[
    value_dtype: DType,
    label_dtype: DType,
    Op: GroupKernel,
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
    operation: Op,
    op_name: String,
    initializers: List[Int],
) raises -> PythonObject:
    """Build and execute a grouped signature with one output."""

    validate_dtype[value_dtype](values, op_name)
    validate_dtype[label_dtype](labels, op_name + " labels")

    var values_tensor = borrow_numpy_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0]],
    ](values)
    var labels_tensor = borrow_numpy_tensor[
        label_dtype,
        False,
        CoreSpec[Dim[0]],
    ](labels)

    # The concrete tuple is the operation's signature. The allocator infers
    # outputs, dtypes, and shapes directly from this tuple.
    var signature = Tuple(
        values_tensor,
        labels_tensor,
        GUTensor[value_dtype, True, CoreSpec[Dim[1]]].empty(),
    )
    return _execute_group[Op](
        values,
        labels,
        axes,
        num_labels,
        options,
        signature,
        initializers,
        operation,
        op_name,
    )


def _apply_group_two[
    value_dtype: DType,
    label_dtype: DType,
    Op: GroupKernel,
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
    operation: Op,
    op_name: String,
    initializers: List[Int],
) raises -> PythonObject:
    """Build and execute a grouped signature with one workspace output."""

    validate_dtype[value_dtype](values, op_name)
    validate_dtype[label_dtype](labels, op_name + " labels")
    var values_tensor = borrow_numpy_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0]],
    ](values)
    var labels_tensor = borrow_numpy_tensor[
        label_dtype,
        False,
        CoreSpec[Dim[0]],
    ](labels)
    var signature = Tuple(
        values_tensor,
        labels_tensor,
        GUTensor[value_dtype, True, CoreSpec[Dim[1]]].empty(),
        GUTensor[DType.int64, True, CoreSpec[Dim[1]]].empty(),
    )
    return _execute_group[Op](
        values,
        labels,
        axes,
        num_labels,
        options,
        signature,
        initializers,
        operation,
        op_name,
    )


def _apply_group_three[
    value_dtype: DType,
    label_dtype: DType,
    Op: GroupKernel,
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
    operation: Op,
    op_name: String,
    initializers: List[Int],
) raises -> PythonObject:
    """Build and execute a grouped signature with two workspaces."""

    validate_dtype[value_dtype](values, op_name)
    validate_dtype[label_dtype](labels, op_name + " labels")
    var values_tensor = borrow_numpy_tensor[
        value_dtype,
        False,
        CoreSpec[Dim[0]],
    ](values)
    var labels_tensor = borrow_numpy_tensor[
        label_dtype,
        False,
        CoreSpec[Dim[0]],
    ](labels)
    var signature = Tuple(
        values_tensor,
        labels_tensor,
        GUTensor[value_dtype, True, CoreSpec[Dim[1]]].empty(),
        GUTensor[value_dtype, True, CoreSpec[Dim[1]]].empty(),
        GUTensor[DType.int64, True, CoreSpec[Dim[1]]].empty(),
    )
    return _execute_group[Op](
        values,
        labels,
        axes,
        num_labels,
        options,
        signature,
        initializers,
        operation,
        op_name,
    )


def group_nansum_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_one[
        value_dtype,
        label_dtype,
        GroupNanSum[value_dtype, label_dtype, 1],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanSum[value_dtype, label_dtype, 1](),
        "group_nansum",
        [GROUP_INIT_ZERO],
    )


def group_nanmean_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_two[
        value_dtype,
        label_dtype,
        GroupNanMean[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanMean[value_dtype, label_dtype](),
        "group_nanmean",
        [GROUP_INIT_ZERO, GROUP_INIT_ZERO],
    )


def group_nanprod_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_one[
        value_dtype,
        label_dtype,
        GroupNanProd[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanProd[value_dtype, label_dtype](),
        "group_nanprod",
        [GROUP_INIT_ONE],
    )


def group_nancount_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_one[
        value_dtype,
        label_dtype,
        GroupNanCount[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanCount[value_dtype, label_dtype](),
        "group_nancount",
        [GROUP_INIT_ZERO],
    )


def group_nanmin_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_two[
        value_dtype,
        label_dtype,
        GroupNanMinMax[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanMinMax[value_dtype, label_dtype, False](),
        "group_nanmin",
        [GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_ZERO],
    )


def group_nanmax_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_two[
        value_dtype,
        label_dtype,
        GroupNanMinMax[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanMinMax[value_dtype, label_dtype, True](),
        "group_nanmax",
        [GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_ZERO],
    )


def group_nanargmin_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_three[
        value_dtype,
        label_dtype,
        GroupNanArgMinMax[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanArgMinMax[value_dtype, label_dtype, False](),
        "group_nanargmin",
        [GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_ZERO],
    )


def group_nanargmax_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_three[
        value_dtype,
        label_dtype,
        GroupNanArgMinMax[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanArgMinMax[value_dtype, label_dtype, True](),
        "group_nanargmax",
        [GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_ZERO],
    )


def group_nanfirst_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_two[
        value_dtype,
        label_dtype,
        GroupNanFirst[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanFirst[value_dtype, label_dtype](),
        "group_nanfirst",
        [GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_ZERO],
    )


def group_nanlast_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_two[
        value_dtype,
        label_dtype,
        GroupNanLast[value_dtype, label_dtype],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanLast[value_dtype, label_dtype](),
        "group_nanlast",
        [GROUP_INIT_NAN_OR_ZERO, GROUP_INIT_ZERO],
    )


def group_nanany_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_one[
        value_dtype,
        label_dtype,
        GroupNanAnyAll[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanAnyAll[value_dtype, label_dtype, False](),
        "group_nanany",
        [GROUP_INIT_ZERO],
    )


def group_nanall_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_one[
        value_dtype,
        label_dtype,
        GroupNanAnyAll[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanAnyAll[value_dtype, label_dtype, True](),
        "group_nanall",
        [GROUP_INIT_ONE],
    )


def group_nanvar_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    var ddof = Int(py=options[1])
    return _apply_group_three[
        value_dtype,
        label_dtype,
        GroupNanVarStd[value_dtype, label_dtype, False],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanVarStd[value_dtype, label_dtype, False](ddof),
        "group_nanvar",
        [GROUP_INIT_ZERO, GROUP_INIT_ZERO, GROUP_INIT_ZERO],
    )


def group_nanstd_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    var ddof = Int(py=options[1])
    return _apply_group_three[
        value_dtype,
        label_dtype,
        GroupNanVarStd[value_dtype, label_dtype, True],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanVarStd[value_dtype, label_dtype, True](ddof),
        "group_nanstd",
        [GROUP_INIT_ZERO, GROUP_INIT_ZERO, GROUP_INIT_ZERO],
    )


def group_nansum_of_squares_binding[
    value_dtype: DType, label_dtype: DType
](
    values: PythonObject,
    labels: PythonObject,
    axes: PythonObject,
    num_labels: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    return _apply_group_one[
        value_dtype,
        label_dtype,
        GroupNanSum[value_dtype, label_dtype, 2],
    ](
        values,
        labels,
        axes,
        num_labels,
        options,
        GroupNanSum[value_dtype, label_dtype, 2](),
        "group_nansum_of_squares",
        [GROUP_INIT_ZERO],
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
