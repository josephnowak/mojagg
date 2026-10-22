"""NaN-aware trailing moving variance and standard deviation."""

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
def _move_var_sequential[
    dtype: DType, take_sqrt: Bool
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute sample variance statistics with a scalar rolling recurrence."""

    var n = len(values)
    var total = Scalar[dtype](0)
    var sum_of_squares = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    if threshold < 2.0:
        threshold = 2.0
    var source_ptr = values.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    for i in range(n):
        var entering = source_ptr[unsafe_offset=i]
        var delta_total = Scalar[dtype](0)
        var delta_squares = Scalar[dtype](0)
        var delta_count = Scalar[dtype](0)
        if not isnan(entering):
            delta_total = entering
            delta_squares = entering * entering
            delta_count = 1
        if i >= window:
            var expiring = source_ptr[unsafe_offset=i - window]
            if not isnan(expiring):
                delta_total -= expiring
                delta_squares -= expiring * expiring
                delta_count -= 1
        total += delta_total
        sum_of_squares += delta_squares
        count += delta_count
        if count >= threshold:
            var variance = (sum_of_squares - total * total / count) / (
                count - 1.0
            )
            comptime if take_sqrt:
                destination_ptr[unsafe_offset=i] = sqrt(variance)
            else:
                destination_ptr[unsafe_offset=i] = variance
        else:
            destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()


struct MoveVarKernel[
    dtype: DType,
    take_sqrt: Bool,
](GUFuncKernel, ImplicitlyCopyable):
    """SIMD ``(n) -> (n)`` trailing sample variance or standard deviation."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[Dim[0]]],
    ]

    var window: Int
    var min_count: Int

    def __init__(out self, window: Int, min_count: Int):
        self.window = window
        self.min_count = min_count

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        _move_var_sequential[Self.dtype, Self.take_sqrt](
            input.read_span(),
            output.write_span(),
            self.window,
            self.min_count,
        )


comptime MoveStdKernel[dtype: DType] = MoveVarKernel[dtype, True]
