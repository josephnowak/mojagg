"""NaN-aware trailing moving sum for the generic ``guvectorize`` driver.

The kernel receives one contiguous logical core and accumulates the rolling
sum in order with a scalar carry.  Independent outer slices remain available
for guvectorize parallelism.
"""

from std.math import isnan

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def _move_sum_sequential[
    dtype: DType
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute one moving-sum core with a scalar rolling recurrence.

    NaN entries contribute neither value nor count. The expiring sample is
    only subtracted once the window is full, so every output observes the
    preceding output's state.
    """

    var n = len(values)
    var sum = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    var source_ptr = values.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()

    for i in range(n):
        var entering = source_ptr[unsafe_offset=i]
        var delta_sum = Scalar[dtype](0)
        var delta_count = Scalar[dtype](0)
        if not isnan(entering):
            delta_sum = entering
            delta_count = 1
        if i >= window:
            var expiring = source_ptr[unsafe_offset=i - window]
            if not isnan(expiring):
                delta_sum -= expiring
                delta_count -= 1
        sum += delta_sum
        count += delta_count
        if count >= threshold:
            destination_ptr[unsafe_offset=i] = sum
        else:
            destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()


struct MoveSumKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD ``(n) -> (n)`` trailing moving sum."""

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
        _move_sum_sequential[Self.dtype](
            input.read_span(),
            output.write_span(),
            self.window,
            self.min_count,
        )
