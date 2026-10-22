"""NaN-aware trailing moving mean."""

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
def _move_mean_sequential[
    dtype: DType
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute one moving-mean core with a scalar rolling recurrence."""

    var n = len(values)
    var total = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    if threshold < 1.0:
        threshold = 1.0
    var source_ptr = values.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    for i in range(n):
        var entering = source_ptr[unsafe_offset=i]
        var delta_total = Scalar[dtype](0)
        var delta_count = Scalar[dtype](0)
        if not isnan(entering):
            delta_total = entering
            delta_count = 1
        if i >= window:
            var expiring = source_ptr[unsafe_offset=i - window]
            if not isnan(expiring):
                delta_total -= expiring
                delta_count -= 1
        total += delta_total
        count += delta_count
        if count >= threshold:
            destination_ptr[unsafe_offset=i] = total / count
        else:
            destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()


struct MoveMeanKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD ``(n) -> (n)`` trailing moving mean."""

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
        _move_mean_sequential[Self.dtype](
            input.read_span(),
            output.write_span(),
            self.window,
            self.min_count,
        )
