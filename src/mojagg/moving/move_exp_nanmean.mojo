"""NaN-aware exponentially weighted moving mean."""

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
def _move_exp_nanmean[
    dtype: DType, scalar_alpha: Bool
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    alphas: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    min_weight: Float64,
):
    var minimum_weight = Scalar[dtype](min_weight)
    var numerator = Scalar[dtype](0)
    var denominator = Scalar[dtype](0)
    var weight = Scalar[dtype](0)
    var values_ptr = values.unsafe_ptr()
    var alphas_ptr = alphas.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    var scalar_alpha_value = Scalar[dtype](0)
    var scalar_decay = Scalar[dtype](0)
    comptime if scalar_alpha:
        scalar_alpha_value = alphas_ptr[unsafe_offset=0]
        scalar_decay = 1.0 - scalar_alpha_value

    for i in range(len(values)):
        var value = values_ptr[unsafe_offset=i]
        var alpha = scalar_alpha_value
        var decay = scalar_decay
        comptime if not scalar_alpha:
            alpha = alphas_ptr[unsafe_offset=i]
            decay = 1.0 - alpha

        numerator *= decay
        denominator *= decay
        weight *= decay

        if not isnan(value):
            numerator += value
            denominator += 1.0
            weight += alpha

        # Gate before dividing: below the weight threshold the quotient would
        # be discarded anyway.
        var output = nan_or_zero[dtype]()
        if weight >= minimum_weight and denominator != 0.0:
            output = numerator / denominator
        destination_ptr[unsafe_offset=i] = output


struct MoveExpNanMeanKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD-driver-compatible ``(n), (n) -> (n)`` moving mean."""

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
        _move_exp_nanmean[Self.dtype, False](
            input.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )


struct MoveExpNanMeanScalarKernel[dtype: DType](
    GUFuncKernel, ImplicitlyCopyable
):
    """``(n), () -> (n)`` moving mean for scalar alpha."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, False, CoreSpec[]],
        GUTensor[Self.dtype, True, CoreSpec[Dim[0]]],
    ]

    var min_weight: Float64

    def __init__(out self, min_weight: Float64):
        self.min_weight = min_weight

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, alpha, output = tensors
        _move_exp_nanmean[Self.dtype, True](
            input.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )
