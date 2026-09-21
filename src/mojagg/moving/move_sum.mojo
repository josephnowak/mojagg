"""NaN-aware trailing moving sum for the generic ``guvectorize`` driver.

The kernel receives one contiguous logical core.  SIMD loads and NaN masks
prepare each block, while the rolling dependency is accumulated in lane order
with a scalar carry.  Independent outer slices remain available for
guvectorize parallelism.
"""

from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)
from mojagg.moving.moving_helpers import (
    load_expiring_block,
    load_masked_block,
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
    """Compute one moving-sum core with a sequential SIMD-block recurrence.

    Loads, NaN masks, and entering-minus-expiring deltas are SIMD operations.
    The rolling dependency is accumulated one lane at a time so every output
    observes the preceding output's state.
    """

    comptime width = 1
    var n = len(values)
    var input_offset = 0
    var sum = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    var destination_ptr = destination.unsafe_ptr()

    while input_offset < n:
        var active = min(width, n - input_offset)
        if input_offset < window:
            active = min(active, window - input_offset)
        var entering = load_masked_block[dtype, width](
            values, input_offset, active
        )
        var expiring = load_expiring_block[dtype, width](
            values, input_offset, window, active
        )
        var delta_sum = entering.values - expiring.values
        var delta_count = entering.counts - expiring.counts
        comptime for lane in range(width):
            if lane < active:
                sum += delta_sum[lane]
                count += delta_count[lane]
                if count >= threshold:
                    destination_ptr[unsafe_offset=input_offset + lane] = sum
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active


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
