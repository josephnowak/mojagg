"""NaN-aware exponentially weighted moving variance and standard deviation."""

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
def _move_exp_nanvar[
    dtype: DType, take_sqrt: Bool
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    alphas: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    min_weight: Float64,
):
    var sum_x_2 = Float64(0.0)
    var sum_x = Float64(0.0)
    var sum_weight = Float64(0.0)
    var sum_weight_2 = Float64(0.0)
    var weight = Float64(0.0)
    var values_ptr = values.unsafe_ptr()
    var alphas_ptr = alphas.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()

    for i in range(len(values)):
        var value = values_ptr[unsafe_offset=i]
        var alpha = Float64(alphas_ptr[unsafe_offset=i])
        var decay = 1.0 - alpha

        sum_x_2 *= decay
        sum_x *= decay
        sum_weight *= decay
        sum_weight_2 *= decay * decay
        weight *= decay

        if not isnan(value):
            sum_x_2 += Float64(value * value)
            sum_x += Float64(value)
            sum_weight += 1.0
            sum_weight_2 += 1.0
            weight += alpha

        if sum_weight != 0.0:
            var var_biased = (sum_x_2 / sum_weight) - (
                (sum_x / sum_weight) * (sum_x / sum_weight)
            )
            var bias = 1.0 - sum_weight_2 / (sum_weight * sum_weight)
            if weight >= min_weight and bias > 0.0:
                var result = var_biased / bias
                comptime if take_sqrt:
                    result = sqrt(result)
                destination_ptr[unsafe_offset=i] = result.cast[dtype]()
            else:
                destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()
        else:
            destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()


struct MoveExpNanVarKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD-driver-compatible ``(n), (n) -> (n)`` moving variance."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[Dim[0]]],
    ]

    var min_weight: Float64

    def __init__(out self, min_weight: Float64):
        self.min_weight = min_weight

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, alpha, output = tensors
        _move_exp_nanvar[Self.dtype, False](
            input.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )


struct MoveExpNanStdKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD-driver-compatible ``(n), (n) -> (n)`` moving standard deviation."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[Dim[0]]],
    ]

    var min_weight: Float64

    def __init__(out self, min_weight: Float64):
        self.min_weight = min_weight

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, alpha, output = tensors
        _move_exp_nanvar[Self.dtype, True](
            input.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )
