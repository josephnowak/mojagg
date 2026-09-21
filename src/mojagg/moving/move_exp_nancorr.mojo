"""NaN-aware exponentially weighted moving correlation."""

from std.collections import Span
from std.math import isnan, sqrt

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def _move_exp_nancorr[
    dtype: DType, scalar_alpha: Bool
](
    values_a: Span[Scalar[dtype], ImmUntrackedOrigin],
    values_b: Span[Scalar[dtype], ImmUntrackedOrigin],
    alphas: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    min_weight: Float64,
):
    var minimum_weight = Scalar[dtype](min_weight)
    var sum_x1 = Scalar[dtype](0)
    var sum_x2 = Scalar[dtype](0)
    var sum_x1x2 = Scalar[dtype](0)
    var sum_x1_2 = Scalar[dtype](0)
    var sum_x2_2 = Scalar[dtype](0)
    var sum_weight = Scalar[dtype](0)
    var sum_weight_2 = Scalar[dtype](0)
    var weight = Scalar[dtype](0)
    var values_a_ptr = values_a.unsafe_ptr()
    var values_b_ptr = values_b.unsafe_ptr()
    var alphas_ptr = alphas.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    var scalar_alpha_value = Scalar[dtype](0)
    var scalar_decay = Scalar[dtype](0)
    comptime if scalar_alpha:
        scalar_alpha_value = alphas_ptr[unsafe_offset=0]
        scalar_decay = 1.0 - scalar_alpha_value
    var scalar_decay_squared = scalar_decay * scalar_decay

    for i in range(len(values_a)):
        var value_a = values_a_ptr[unsafe_offset=i]
        var value_b = values_b_ptr[unsafe_offset=i]
        var alpha = scalar_alpha_value
        var decay = scalar_decay
        var decay_squared = scalar_decay_squared
        comptime if not scalar_alpha:
            alpha = alphas_ptr[unsafe_offset=i]
            decay = 1.0 - alpha
            decay_squared = decay * decay

        sum_x1 *= decay
        sum_x2 *= decay
        sum_x1x2 *= decay
        sum_weight *= decay
        sum_weight_2 *= decay_squared
        weight *= decay
        sum_x1_2 *= decay
        sum_x2_2 *= decay

        if not (isnan(value_a) or isnan(value_b)):
            sum_x1 += value_a
            sum_x2 += value_b
            sum_x1x2 += value_a * value_b
            sum_weight += 1.0
            sum_weight_2 += 1.0
            weight += alpha
            sum_x1_2 += value_a * value_a
            sum_x2_2 += value_b * value_b

        # Gate on the cheap conditions before the four divisions and the
        # square root: below the weight threshold they would be discarded.
        var output = nan_or_zero[dtype]()
        if weight >= minimum_weight and sum_weight != 0.0:
            var bias = 1.0 - sum_weight_2 / (sum_weight * sum_weight)
            if bias > 0.0:
                var cov = sum_x1x2 - sum_x1 * sum_x2 / sum_weight
                var var_a = sum_x1_2 - sum_x1 * sum_x1 / sum_weight
                var var_b = sum_x2_2 - sum_x2 * sum_x2 / sum_weight
                var denominator = sqrt(var_a * var_b)
                if denominator > 0.0:
                    output = cov / denominator
        destination_ptr[unsafe_offset=i] = output


struct MoveExpNanCorrKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD-driver-compatible ``(n), (n), (n) -> (n)`` moving correlation."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[Dim[0]]],
    ]

    var min_weight: Float64

    def __init__(out self, min_weight: Float64):
        self.min_weight = min_weight

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input_a, input_b, alpha, output = tensors
        _move_exp_nancorr[Self.dtype, False](
            input_a.read_span(),
            input_b.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )


struct MoveExpNanCorrScalarKernel[dtype: DType](
    GUFuncKernel, ImplicitlyCopyable
):
    """``(n), (n), () -> (n)`` moving correlation for scalar alpha."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, False, CoreSpec[]],
        GUTensor[Self.dtype, True, CoreSpec[Dim[0]]],
    ]

    var min_weight: Float64

    def __init__(out self, min_weight: Float64):
        self.min_weight = min_weight

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input_a, input_b, alpha, output = tensors
        _move_exp_nancorr[Self.dtype, True](
            input_a.read_span(),
            input_b.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )
