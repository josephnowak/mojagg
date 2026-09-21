"""NaN-aware trailing moving mean."""

from std.collections import Span
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
def _move_mean_sequential[
    dtype: DType
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute one moving-mean core with SIMD block deltas."""

    comptime width = 1
    var n = len(values)
    var input_offset = 0
    var total = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    if threshold < 1.0:
        threshold = 1.0
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
        var delta_total = entering.values - expiring.values
        var delta_count = entering.counts - expiring.counts
        comptime for lane in range(width):
            if lane < active:
                total += delta_total[lane]
                count += delta_count[lane]
                if count >= threshold:
                    destination_ptr[unsafe_offset=input_offset + lane] = (
                        total / count
                    )
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active


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
