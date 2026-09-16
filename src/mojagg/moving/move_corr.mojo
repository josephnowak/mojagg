"""NaN-aware trailing moving correlation."""

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
def _move_corr_sequential[
    dtype: DType
](
    a_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    b_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute pairwise correlation with SIMD block deltas."""

    comptime width = simd_width_of[dtype]()
    var n = len(a_values)
    var input_offset = 0
    var a_sum = Float64(0.0)
    var b_sum = Float64(0.0)
    var product_sum = Float64(0.0)
    var a_sum_of_squares = Float64(0.0)
    var b_sum_of_squares = Float64(0.0)
    var count = Float64(0.0)
    var threshold = Float64(min_count)
    if threshold < 1.0:
        threshold = 1.0
    var zero = SIMD[DType.float64, width](0.0)
    var one = SIMD[DType.float64, width](1.0)
    var destination_ptr = destination.unsafe_ptr()
    var warmup = min(window, n)

    while input_offset < warmup:
        var active = min(width, warmup - input_offset)
        var a_block = load_block[dtype, width](a_values, input_offset, active)
        var b_block = load_block[dtype, width](b_values, input_offset, active)
        var missing = isnan(a_block) | isnan(b_block)
        var cleaned_a_native = missing.select(SIMD[dtype, width](0), a_block)
        var cleaned_b_native = missing.select(SIMD[dtype, width](0), b_block)
        var cleaned_a = cleaned_a_native.cast[DType.float64]()
        var cleaned_b = cleaned_b_native.cast[DType.float64]()
        var valid = missing.select(zero, one)
        var products = (cleaned_a_native * cleaned_b_native).cast[
            DType.float64
        ]()
        var a_squares = (cleaned_a_native * cleaned_a_native).cast[
            DType.float64
        ]()
        var b_squares = (cleaned_b_native * cleaned_b_native).cast[
            DType.float64
        ]()
        comptime for lane in range(width):
            if lane < active:
                a_sum += cleaned_a[lane]
                b_sum += cleaned_b[lane]
                product_sum += products[lane]
                a_sum_of_squares += a_squares[lane]
                b_sum_of_squares += b_squares[lane]
                count += valid[lane]
                if count >= threshold:
                    var reciprocal = 1.0 / count
                    var mean_a = a_sum * reciprocal
                    var mean_b = b_sum * reciprocal
                    var variance_a = (
                        a_sum_of_squares * reciprocal - mean_a * mean_a
                    )
                    var variance_b = (
                        b_sum_of_squares * reciprocal - mean_b * mean_b
                    )
                    var covariance = product_sum * reciprocal - mean_a * mean_b
                    var variance_product = variance_a * variance_b
                    if variance_product > 0.0:
                        destination_ptr[unsafe_offset=input_offset + lane] = (
                            covariance / sqrt(variance_product)
                        ).cast[dtype]()
                    else:
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = nan_or_zero[dtype]()
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active

    while input_offset < n:
        var active = min(width, n - input_offset)
        var entering_a = load_block[dtype, width](
            a_values, input_offset, active
        )
        var entering_b = load_block[dtype, width](
            b_values, input_offset, active
        )
        var expiring_a = load_block[dtype, width](
            a_values, input_offset - window, active
        )
        var expiring_b = load_block[dtype, width](
            b_values, input_offset - window, active
        )
        var entering_missing = isnan(entering_a) | isnan(entering_b)
        var expiring_missing = isnan(expiring_a) | isnan(expiring_b)
        var entering_a_native = entering_missing.select(
            SIMD[dtype, width](0), entering_a
        )
        var entering_b_native = entering_missing.select(
            SIMD[dtype, width](0), entering_b
        )
        var expiring_a_native = expiring_missing.select(
            SIMD[dtype, width](0), expiring_a
        )
        var expiring_b_native = expiring_missing.select(
            SIMD[dtype, width](0), expiring_b
        )
        var entering_a_values = entering_a_native.cast[DType.float64]()
        var entering_b_values = entering_b_native.cast[DType.float64]()
        var expiring_a_values = expiring_a_native.cast[DType.float64]()
        var expiring_b_values = expiring_b_native.cast[DType.float64]()
        var entering_products = (entering_a_native * entering_b_native).cast[
            DType.float64
        ]()
        var expiring_products = (expiring_a_native * expiring_b_native).cast[
            DType.float64
        ]()
        var entering_a_squares = (entering_a_native * entering_a_native).cast[
            DType.float64
        ]()
        var expiring_a_squares = (expiring_a_native * expiring_a_native).cast[
            DType.float64
        ]()
        var entering_b_squares = (entering_b_native * entering_b_native).cast[
            DType.float64
        ]()
        var expiring_b_squares = (expiring_b_native * expiring_b_native).cast[
            DType.float64
        ]()
        comptime for lane in range(width):
            if lane < active:
                if not expiring_missing[lane]:
                    a_sum -= expiring_a_values[lane]
                    b_sum -= expiring_b_values[lane]
                    product_sum -= expiring_products[lane]
                    a_sum_of_squares -= expiring_a_squares[lane]
                    b_sum_of_squares -= expiring_b_squares[lane]
                    count -= 1.0
                if not entering_missing[lane]:
                    a_sum += entering_a_values[lane]
                    b_sum += entering_b_values[lane]
                    product_sum += entering_products[lane]
                    a_sum_of_squares += entering_a_squares[lane]
                    b_sum_of_squares += entering_b_squares[lane]
                    count += 1.0
                if count >= threshold:
                    var reciprocal = 1.0 / count
                    var mean_a = a_sum * reciprocal
                    var mean_b = b_sum * reciprocal
                    var variance_a = (
                        a_sum_of_squares * reciprocal - mean_a * mean_a
                    )
                    var variance_b = (
                        b_sum_of_squares * reciprocal - mean_b * mean_b
                    )
                    var covariance = product_sum * reciprocal - mean_a * mean_b
                    var variance_product = variance_a * variance_b
                    if variance_product > 0.0:
                        destination_ptr[unsafe_offset=input_offset + lane] = (
                            covariance / sqrt(variance_product)
                        ).cast[dtype]()
                    else:
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = nan_or_zero[dtype]()
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active


struct MoveCorrKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD ``(n), (n) -> (n)`` trailing correlation."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
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
        var input_a, input_b, output = tensors
        _move_corr_sequential[Self.dtype](
            input_a.read_span(),
            input_b.read_span(),
            output.write_span(),
            self.window,
            self.min_count,
        )
