"""NaN-aware trailing moving sum for the generic ``guvectorize`` driver.

The kernel receives one contiguous logical core.  SIMD loads and NaN masks
prepare each block, while the rolling dependency is accumulated in lane order
with a scalar carry.  Independent outer slices remain available for
guvectorize parallelism.
"""

from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)
from mojagg.moving.moving_helpers import load_block


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

    comptime width = simd_width_of[dtype]()
    var n = len(values)
    var input_offset = 0
    var sum = Float64(0.0)
    var count = Float64(0.0)
    var threshold = Float64(min_count)
    var zero = SIMD[DType.float64, width](0.0)
    var one = SIMD[DType.float64, width](1.0)
    var destination_ptr = destination.unsafe_ptr()

    # Warm-up uses the same SIMD loads and masks as the scan kernel, then
    # accumulates each lane in order.
    while input_offset < min(window, n):
        var active = min(width, min(window, n) - input_offset)
        var block = load_block[dtype, width](values, input_offset, active)
        var missing = isnan(block)
        var cleaned = missing.select(SIMD[dtype, width](0), block).cast[
            DType.float64
        ]()
        var valid = missing.select(zero, one)
        comptime for lane in range(width):
            if lane < active:
                sum += cleaned[lane]
                count += valid[lane]
                if count >= threshold:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = sum.cast[dtype]()
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active

    # Steady state still computes deltas in SIMD, but performs the dependent
    # prefix accumulation explicitly in lane order.
    while input_offset < n:
        var active = min(width, n - input_offset)
        var entering = load_block[dtype, width](values, input_offset, active)
        var expiring = load_block[dtype, width](
            values, input_offset - window, active
        )
        var entering_missing = isnan(entering)
        var expiring_missing = isnan(expiring)
        var entering_values = entering_missing.select(
            SIMD[dtype, width](0), entering
        ).cast[DType.float64]()
        var expiring_values = expiring_missing.select(
            SIMD[dtype, width](0), expiring
        ).cast[DType.float64]()
        var entering_count = entering_missing.select(zero, one)
        var expiring_count = expiring_missing.select(zero, one)
        var delta_sum = entering_values - expiring_values
        var delta_count = entering_count - expiring_count
        comptime for lane in range(width):
            if lane < active:
                sum += delta_sum[lane]
                count += delta_count[lane]
                if count >= threshold:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = sum.cast[dtype]()
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
