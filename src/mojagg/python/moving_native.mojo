"""Python binding for the moving-window kernels."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.drivers.guvectorize import (
    AxisSpec,
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
    build_signature_plan,
    guvectorize,
)
from mojagg.moving.move_corr import MoveCorrKernel
from mojagg.moving.move_cov import MoveCovKernel
from mojagg.moving.move_corrmatrix import MoveCorrMatrixKernel
from mojagg.moving.move_covmatrix import MoveCovMatrixKernel
from mojagg.moving.move_exp_nancorr import MoveExpNanCorrKernel
from mojagg.moving.move_exp_nancount import MoveExpNanCountKernel
from mojagg.moving.move_exp_nancorrmatrix import MoveExpNanCorrMatrixKernel
from mojagg.moving.move_exp_nancov import MoveExpNanCovKernel
from mojagg.moving.move_exp_nancovmatrix import MoveExpNanCovMatrixKernel
from mojagg.moving.move_exp_nanmean import MoveExpNanMeanKernel
from mojagg.moving.move_exp_nansum import MoveExpNanSumKernel
from mojagg.moving.move_exp_nanvar import (
    MoveExpNanStdKernel,
    MoveExpNanVarKernel,
)
from mojagg.moving.move_mean import MoveMeanKernel
from mojagg.moving.move_sum import MoveSumKernel
from mojagg.moving.move_var import MoveStdKernel, MoveVarKernel
from mojagg.python.common import (
    axes_from_py,
    borrow_numpy_tensor,
    dispatch_policy_from_py,
    validate_dtype,
)
from mojagg.python.signature_materialization import materialize_outputs


def _apply_unary[
    dtype: DType,
    Op: GUFuncKernel,
](
    arr: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    validate_dtype[dtype](arr, op_name)
    var input = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var axis_spec = axes_from_py(axes, op_name)
    if axis_spec.count != 1:
        raise Error(op_name + ": expected exactly one axis")
    var policy = dispatch_policy_from_py(cfg)
    var signature = Tuple(
        input,
        GUTensor[dtype, True, CoreSpec[Dim[0]]].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        axis_spec,
        axis_spec,
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](operation, signature, plan, policy)
    return outputs[0]


def _apply_binary[
    dtype: DType,
    Op: GUFuncKernel,
](
    a: PythonObject,
    b: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    validate_dtype[dtype](a, op_name)
    validate_dtype[dtype](b, op_name)
    var input_a = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](a)
    var input_b = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](b)
    var axis_spec = axes_from_py(axes, op_name)
    if axis_spec.count != 1:
        raise Error(op_name + ": expected exactly one axis")
    var policy = dispatch_policy_from_py(cfg)
    var signature = Tuple(
        input_a,
        input_b,
        GUTensor[dtype, True, CoreSpec[Dim[0]]].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        axis_spec,
        axis_spec,
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](operation, signature, plan, policy)
    return outputs[0]


def _apply_matrix[
    dtype: DType,
    Op: GUFuncKernel,
](
    arr: PythonObject,
    axes: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run a moving matrix operation over an ``(obs, vars)`` input core."""

    validate_dtype[dtype](arr, op_name)
    var input = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0], Dim[1]],
    ](arr)
    var input_axes = axes_from_py(axes, op_name)
    if input_axes.count != 2:
        raise Error(op_name + ": expected exactly two axes")

    # The moving matrix gufunc keeps observations in the output and appends
    # the two variable axes: (..., obs, vars) -> (..., obs, vars, vars).
    var output_axes = AxisSpec.empty()
    output_axes.count = 3
    output_axes[0] = input.ndim - 2
    output_axes[1] = input.ndim - 1
    output_axes[2] = input.ndim
    var signature = Tuple(
        input,
        GUTensor[
            dtype,
            True,
            CoreSpec[Dim[0], Dim[1], Dim[1]],
        ].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        input_axes,
        output_axes,
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](operation, signature, plan, dispatch_policy_from_py(cfg))
    return outputs[0]


def move_sum_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_unary[dtype, MoveSumKernel[dtype]](
        arr,
        axes,
        window,
        min_count,
        cfg,
        MoveSumKernel[dtype](Int(py=window), Int(py=min_count)),
        "move_sum",
    )


def move_mean_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_unary[dtype, MoveMeanKernel[dtype]](
        arr,
        axes,
        window,
        min_count,
        cfg,
        MoveMeanKernel[dtype](Int(py=window), Int(py=min_count)),
        "move_mean",
    )


def move_var_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_unary[dtype, MoveVarKernel[dtype, False]](
        arr,
        axes,
        window,
        min_count,
        cfg,
        MoveVarKernel[dtype, False](Int(py=window), Int(py=min_count)),
        "move_var",
    )


def move_std_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_unary[dtype, MoveStdKernel[dtype]](
        arr,
        axes,
        window,
        min_count,
        cfg,
        MoveStdKernel[dtype](Int(py=window), Int(py=min_count)),
        "move_std",
    )


def move_cov_binding[
    dtype: DType
](
    a: PythonObject,
    b: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_binary[dtype, MoveCovKernel[dtype]](
        a,
        b,
        axes,
        window,
        min_count,
        cfg,
        MoveCovKernel[dtype](Int(py=window), Int(py=min_count)),
        "move_cov",
    )


def move_corr_binding[
    dtype: DType
](
    a: PythonObject,
    b: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_binary[dtype, MoveCorrKernel[dtype]](
        a,
        b,
        axes,
        window,
        min_count,
        cfg,
        MoveCorrKernel[dtype](Int(py=window), Int(py=min_count)),
        "move_corr",
    )


def move_corrmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    var input_ndim = Int(py=arr.ndim)
    var n_obs = Int(py=arr.shape[input_ndim - 2])
    var n_vars = Int(py=arr.shape[input_ndim - 1])
    return _apply_matrix[dtype, MoveCorrMatrixKernel[dtype]](
        arr,
        axes,
        cfg,
        MoveCorrMatrixKernel[dtype](
            n_vars,
            n_obs,
            Int(py=window),
            Int(py=min_count),
        ),
        "move_corrmatrix",
    )


def move_covmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    axes: PythonObject,
    window: PythonObject,
    min_count: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    var input_ndim = Int(py=arr.ndim)
    var n_obs = Int(py=arr.shape[input_ndim - 2])
    var n_vars = Int(py=arr.shape[input_ndim - 1])
    return _apply_matrix[dtype, MoveCovMatrixKernel[dtype]](
        arr,
        axes,
        cfg,
        MoveCovMatrixKernel[dtype](
            n_vars,
            n_obs,
            Int(py=window),
            Int(py=min_count),
        ),
        "move_covmatrix",
    )


def _apply_exp_unary[
    dtype: DType,
    Op: GUFuncKernel,
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    validate_dtype[dtype](arr, op_name)
    validate_dtype[dtype](alpha, op_name)
    var input = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](arr)
    var input_alpha = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](alpha)
    var input_axes = axes_from_py(axes, op_name)
    if input_axes.count != 1:
        raise Error(op_name + ": expected exactly one axis")

    # Inputs retain their physical layout. The driver gathers the selected
    # input core into a contiguous span, while the public facade moves the
    # completed output core back to the requested axis.
    var output_axes = AxisSpec.empty()
    output_axes.count = 1
    output_axes[0] = input.ndim - 1
    var signature = Tuple(
        input,
        input_alpha,
        GUTensor[dtype, True, CoreSpec[Dim[0]]].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        input_axes,
        output_axes,
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](operation, signature, plan, dispatch_policy_from_py(cfg))
    return outputs[0]


def _apply_exp_binary[
    dtype: DType,
    Op: GUFuncKernel,
](
    a: PythonObject,
    b: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    validate_dtype[dtype](a, op_name)
    validate_dtype[dtype](b, op_name)
    validate_dtype[dtype](alpha, op_name)
    var input_a = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](a)
    var input_b = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](b)
    var input_alpha = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0]],
    ](alpha)
    var input_axes = axes_from_py(axes, op_name)
    if input_axes.count != 1:
        raise Error(op_name + ": expected exactly one axis")

    # The facade broadcasts the read operands and alpha to one common rank;
    # guvectorize still handles their outer broadcast strides and selected
    # axis alignment without moving the source arrays.
    var output_axes = AxisSpec.empty()
    output_axes.count = 1
    output_axes[0] = input_a.ndim - 1
    var signature = Tuple(
        input_a,
        input_b,
        input_alpha,
        GUTensor[dtype, True, CoreSpec[Dim[0]]].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        input_axes,
        output_axes,
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](operation, signature, plan, dispatch_policy_from_py(cfg))
    return outputs[0]


def _apply_exp_matrix[
    dtype: DType,
    Op: GUFuncKernel,
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
    operation: Op,
    op_name: String,
) raises -> PythonObject:
    """Run an exponential moving matrix operation over one matrix core."""

    validate_dtype[dtype](arr, op_name)
    validate_dtype[dtype](alpha, op_name)
    var input = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0], Dim[1]],
    ](arr)
    var input_alpha = borrow_numpy_tensor[
        dtype,
        False,
        CoreSpec[Dim[0], Dim[1]],
    ](alpha)
    var input_axes = axes_from_py(axes, op_name)
    if input_axes.count != 2:
        raise Error(op_name + ": expected exactly two axes")

    var output_axes = AxisSpec.empty()
    output_axes.count = 3
    output_axes[0] = input.ndim - 2
    output_axes[1] = input.ndim - 1
    output_axes[2] = input.ndim
    var signature = Tuple(
        input,
        input_alpha,
        GUTensor[
            dtype,
            True,
            CoreSpec[Dim[0], Dim[1], Dim[1]],
        ].empty(),
    )
    var plan = build_signature_plan[Op](
        signature,
        input_axes,
        output_axes,
    )
    var outputs = materialize_outputs[Op](signature)
    guvectorize[Op](operation, signature, plan, dispatch_policy_from_py(cfg))
    return outputs[0]


def move_exp_nancount_binding[
    dtype: DType
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_exp_unary[dtype, MoveExpNanCountKernel[dtype]](
        arr,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanCountKernel[dtype](Float64(py=min_weight)),
        "move_exp_nancount",
    )


def move_exp_nanmean_binding[
    dtype: DType
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_exp_unary[dtype, MoveExpNanMeanKernel[dtype]](
        arr,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanMeanKernel[dtype](Float64(py=min_weight)),
        "move_exp_nanmean",
    )


def move_exp_nansum_binding[
    dtype: DType
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_exp_unary[dtype, MoveExpNanSumKernel[dtype]](
        arr,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanSumKernel[dtype](Float64(py=min_weight)),
        "move_exp_nansum",
    )


def move_exp_nanvar_binding[
    dtype: DType
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_exp_unary[dtype, MoveExpNanVarKernel[dtype]](
        arr,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanVarKernel[dtype](Float64(py=min_weight)),
        "move_exp_nanvar",
    )


def move_exp_nanstd_binding[
    dtype: DType
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_exp_unary[dtype, MoveExpNanStdKernel[dtype]](
        arr,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanStdKernel[dtype](Float64(py=min_weight)),
        "move_exp_nanstd",
    )


def move_exp_nancov_binding[
    dtype: DType
](
    a: PythonObject,
    b: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_exp_binary[dtype, MoveExpNanCovKernel[dtype]](
        a,
        b,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanCovKernel[dtype](Float64(py=min_weight)),
        "move_exp_nancov",
    )


def move_exp_nancorr_binding[
    dtype: DType
](
    a: PythonObject,
    b: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_exp_binary[dtype, MoveExpNanCorrKernel[dtype]](
        a,
        b,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanCorrKernel[dtype](Float64(py=min_weight)),
        "move_exp_nancorr",
    )


def move_exp_nancorrmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    var input_ndim = Int(py=arr.ndim)
    var n_obs = Int(py=arr.shape[input_ndim - 2])
    var n_vars = Int(py=arr.shape[input_ndim - 1])
    return _apply_exp_matrix[
        dtype,
        MoveExpNanCorrMatrixKernel[dtype],
    ](
        arr,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanCorrMatrixKernel[dtype](
            n_vars,
            n_obs,
            Float64(py=min_weight),
        ),
        "move_exp_nancorrmatrix",
    )


def move_exp_nancovmatrix_binding[
    dtype: DType
](
    arr: PythonObject,
    alpha: PythonObject,
    axes: PythonObject,
    min_weight: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    var input_ndim = Int(py=arr.ndim)
    var n_obs = Int(py=arr.shape[input_ndim - 2])
    var n_vars = Int(py=arr.shape[input_ndim - 1])
    return _apply_exp_matrix[
        dtype,
        MoveExpNanCovMatrixKernel[dtype],
    ](
        arr,
        alpha,
        axes,
        min_weight,
        cfg,
        MoveExpNanCovMatrixKernel[dtype](
            n_vars,
            n_obs,
            Float64(py=min_weight),
        ),
        "move_exp_nancovmatrix",
    )


@export
def PyInit_moving_native() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("moving_native")
        m.def_function[move_corr_binding[DType.float64]]("move_corr_f64")
        m.def_function[move_corr_binding[DType.float32]]("move_corr_f32")
        m.def_function[move_corrmatrix_binding[DType.float64]](
            "move_corrmatrix_f64"
        )
        m.def_function[move_corrmatrix_binding[DType.float32]](
            "move_corrmatrix_f32"
        )
        m.def_function[move_cov_binding[DType.float64]]("move_cov_f64")
        m.def_function[move_cov_binding[DType.float32]]("move_cov_f32")
        m.def_function[move_covmatrix_binding[DType.float64]](
            "move_covmatrix_f64"
        )
        m.def_function[move_covmatrix_binding[DType.float32]](
            "move_covmatrix_f32"
        )
        m.def_function[move_mean_binding[DType.float64]]("move_mean_f64")
        m.def_function[move_mean_binding[DType.float32]]("move_mean_f32")
        m.def_function[move_std_binding[DType.float64]]("move_std_f64")
        m.def_function[move_std_binding[DType.float32]]("move_std_f32")
        m.def_function[move_sum_binding[DType.float64]]("move_sum_f64")
        m.def_function[move_sum_binding[DType.float32]]("move_sum_f32")
        m.def_function[move_var_binding[DType.float64]]("move_var_f64")
        m.def_function[move_var_binding[DType.float32]]("move_var_f32")
        m.def_function[move_exp_nancorr_binding[DType.float64]](
            "move_exp_nancorr_f64"
        )
        m.def_function[move_exp_nancorr_binding[DType.float32]](
            "move_exp_nancorr_f32"
        )
        m.def_function[move_exp_nancorrmatrix_binding[DType.float64]](
            "move_exp_nancorrmatrix_f64"
        )
        m.def_function[move_exp_nancorrmatrix_binding[DType.float32]](
            "move_exp_nancorrmatrix_f32"
        )
        m.def_function[move_exp_nancount_binding[DType.float64]](
            "move_exp_nancount_f64"
        )
        m.def_function[move_exp_nancount_binding[DType.float32]](
            "move_exp_nancount_f32"
        )
        m.def_function[move_exp_nancov_binding[DType.float64]](
            "move_exp_nancov_f64"
        )
        m.def_function[move_exp_nancov_binding[DType.float32]](
            "move_exp_nancov_f32"
        )
        m.def_function[move_exp_nancovmatrix_binding[DType.float64]](
            "move_exp_nancovmatrix_f64"
        )
        m.def_function[move_exp_nancovmatrix_binding[DType.float32]](
            "move_exp_nancovmatrix_f32"
        )
        m.def_function[move_exp_nanmean_binding[DType.float64]](
            "move_exp_nanmean_f64"
        )
        m.def_function[move_exp_nanmean_binding[DType.float32]](
            "move_exp_nanmean_f32"
        )
        m.def_function[move_exp_nanstd_binding[DType.float64]](
            "move_exp_nanstd_f64"
        )
        m.def_function[move_exp_nanstd_binding[DType.float32]](
            "move_exp_nanstd_f32"
        )
        m.def_function[move_exp_nansum_binding[DType.float64]](
            "move_exp_nansum_f64"
        )
        m.def_function[move_exp_nansum_binding[DType.float32]](
            "move_exp_nansum_f32"
        )
        m.def_function[move_exp_nanvar_binding[DType.float64]](
            "move_exp_nanvar_f64"
        )
        m.def_function[move_exp_nanvar_binding[DType.float32]](
            "move_exp_nanvar_f32"
        )
        return m.finalize()
    except e:
        abort(String("failed to create moving_native module: ", e))
