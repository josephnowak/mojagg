"""NaN-aware trailing moving variance and standard deviation."""

from std.collections import Span
from std.math import isnan, sqrt
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
    var total = Float64(0.0)
    var sum_of_squares = Float64(0.0)
    var count = Float64(0.0)
    var threshold = Float64(min_count)
    if threshold < 2.0:
        threshold = 2.0
    var zero = SIMD[DType.float64, width](0.0)
    var one = SIMD[DType.float64, width](1.0)
    var destination_ptr = destination.unsafe_ptr()
    var warmup = min(window, n)

    while input_offset < warmup:
        var active = min(width, warmup - input_offset)
        var block = load_block[dtype, width](values, input_offset, active)
        var missing = isnan(block)
        var cleaned_native = missing.select(SIMD[dtype, width](0), block)
        var cleaned = cleaned_native.cast[DType.float64]()
        var valid = missing.select(zero, one)
        var squares = (cleaned_native * cleaned_native).cast[DType.float64]()
        comptime for lane in range(width):
            if lane < active:
                total += cleaned[lane]
                sum_of_squares += squares[lane]
                count += valid[lane]
                if count >= threshold:
                    var variance = (sum_of_squares - total * total / count) / (
                        count - 1.0
                    )
                    comptime if take_sqrt:
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = sqrt(variance).cast[dtype]()
                    else:
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = variance.cast[dtype]()
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active

    while input_offset < n:
        var active = min(width, n - input_offset)
        var entering = load_block[dtype, width](values, input_offset, active)
        var expiring = load_block[dtype, width](
            values, input_offset - window, active
        )
        var entering_missing = isnan(entering)
        var expiring_missing = isnan(expiring)
        var entering_native = entering_missing.select(
            SIMD[dtype, width](0), entering
        )
        var expiring_native = expiring_missing.select(
            SIMD[dtype, width](0), expiring
        )
        var entering_values = entering_native.cast[DType.float64]()
        var expiring_values = expiring_native.cast[DType.float64]()
        var entering_count = entering_missing.select(zero, one)
        var expiring_count = expiring_missing.select(zero, one)
        var delta_total = entering_values - expiring_values
        var delta_squares = (entering_native * entering_native).cast[
            DType.float64
        ]() - (expiring_native * expiring_native).cast[DType.float64]()
        var delta_count = entering_count - expiring_count
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
                        ] = sqrt(variance).cast[dtype]()
                    else:
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = variance.cast[dtype]()
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
