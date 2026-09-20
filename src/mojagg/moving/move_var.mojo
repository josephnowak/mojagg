"""NaN-aware trailing moving variance and standard deviation."""

from std.collections import Span
from std.math import sqrt
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
def _move_var_sequential[
    dtype: DType, take_sqrt: Bool
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute sample variance statistics with SIMD block deltas."""

    comptime width = simd_width_of[dtype]()
    var n = len(values)
    var input_offset = 0
    var total = Scalar[dtype](0)
    var sum_of_squares = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    if threshold < 2.0:
        threshold = 2.0
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
        var delta_squares = entering.squares() - expiring.squares()
        var delta_count = entering.counts - expiring.counts
        comptime for lane in range(width):
            if lane < active:
                total += delta_total[lane]
                sum_of_squares += delta_squares[lane]
                count += delta_count[lane]
                if count >= threshold:
                    var variance = (sum_of_squares - total * total / count) / (
                        count - 1.0
                    )
                    comptime if take_sqrt:
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = sqrt(variance)
                    else:
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = variance
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active


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
