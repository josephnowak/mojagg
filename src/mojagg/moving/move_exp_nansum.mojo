"""NaN-aware exponentially weighted moving sum."""

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
def _move_exp_nansum[
    dtype: DType
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    alphas: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    min_weight: Float64,
):
    var numerator = Float64(0.0)
    var weight = Float64(0.0)
    var seen = False
    var values_ptr = values.unsafe_ptr()
    var alphas_ptr = alphas.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()

    for i in range(len(values)):
        var value = values_ptr[unsafe_offset=i]
        var alpha = Float64(alphas_ptr[unsafe_offset=i])
        var decay = 1.0 - alpha

        numerator *= decay
        weight *= decay

        if not isnan(value):
            seen = True
            numerator += Float64(value)
            weight += alpha

        if weight >= min_weight and seen:
            destination_ptr[unsafe_offset=i] = numerator.cast[dtype]()
        else:
            destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()


struct MoveExpNanSumKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD-driver-compatible ``(n), (n) -> (n)`` moving sum."""

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
        _move_exp_nansum[Self.dtype](
            input.read_span(),
            alpha.read_span(),
            output.write_span(),
            self.min_weight,
        )
