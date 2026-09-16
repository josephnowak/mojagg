"""NaN-aware exponentially weighted moving covariance."""

from std.collections import Span
from std.math import isnan

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def _move_exp_nancov[
    dtype: DType
](
    values_a: Span[Scalar[dtype], ImmUntrackedOrigin],
    values_b: Span[Scalar[dtype], ImmUntrackedOrigin],
    alphas: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    min_weight: Float64,
):
    var sum_x1 = Float64(0.0)
    var sum_x2 = Float64(0.0)
    var sum_x1x2 = Float64(0.0)
    var sum_weight = Float64(0.0)
    var sum_weight_2 = Float64(0.0)
    var weight = Float64(0.0)
    var values_a_ptr = values_a.unsafe_ptr()
    var values_b_ptr = values_b.unsafe_ptr()
    var alphas_ptr = alphas.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()

    for i in range(len(values_a)):
        var value_a = values_a_ptr[unsafe_offset=i]
        var value_b = values_b_ptr[unsafe_offset=i]
        var alpha = Float64(alphas_ptr[unsafe_offset=i])
        var decay = 1.0 - alpha

        sum_x1 *= decay
        sum_x2 *= decay
        sum_x1x2 *= decay
        sum_weight *= decay
        sum_weight_2 *= decay * decay
        weight *= decay

        if not (isnan(value_a) or isnan(value_b)):
            sum_x1 += Float64(value_a)
            sum_x2 += Float64(value_b)
            sum_x1x2 += Float64(value_a * value_b)
            sum_weight += 1.0
            sum_weight_2 += 1.0
            weight += alpha

        if sum_weight != 0.0:
            var cov_biased = (
                sum_x1x2 - sum_x1 * sum_x2 / sum_weight
            ) / sum_weight
            var bias = 1.0 - sum_weight_2 / (sum_weight * sum_weight)
            if weight >= min_weight and bias > 0.0:
                destination_ptr[unsafe_offset=i] = (cov_biased / bias).cast[
                    dtype
                ]()
            else:
                destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()
        else:
            destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()


struct MoveExpNanCovKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD-driver-compatible ``(n), (n), (n) -> (n)`` moving covariance."""

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
        _move_exp_nancov[Self.dtype](
            input_a.read_span(),
            input_b.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )
