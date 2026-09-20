"""NaN-aware trailing moving correlation."""

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
    load_expiring_pair,
    load_masked_pair,
)


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
    var a_sum = Scalar[dtype](0)
    var b_sum = Scalar[dtype](0)
    var product_sum = Scalar[dtype](0)
    var a_sum_of_squares = Scalar[dtype](0)
    var b_sum_of_squares = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    if threshold < 1.0:
        threshold = 1.0
    var destination_ptr = destination.unsafe_ptr()
    while input_offset < n:
        var active = min(width, n - input_offset)
        if input_offset < window:
            active = min(active, window - input_offset)
        var entering_a, entering_b = load_masked_pair[dtype, width](
            a_values, b_values, input_offset, active
        )
        var expiring_a, expiring_b = load_expiring_pair[dtype, width](
            a_values, b_values, input_offset, window, active
        )
        var delta_a = entering_a.values - expiring_a.values
        var delta_b = entering_b.values - expiring_b.values
        var delta_products = entering_a.products(
            entering_b
        ) - expiring_a.products(expiring_b)
        var delta_a_squares = entering_a.squares() - expiring_a.squares()
        var delta_b_squares = entering_b.squares() - expiring_b.squares()
        var delta_count = entering_a.counts - expiring_a.counts

        comptime for lane in range(width):
            if lane < active:
                a_sum += delta_a[lane]
                b_sum += delta_b[lane]
                product_sum += delta_products[lane]
                a_sum_of_squares += delta_a_squares[lane]
                b_sum_of_squares += delta_b_squares[lane]
                count += delta_count[lane]
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
                        destination_ptr[
                            unsafe_offset=input_offset + lane
                        ] = covariance / sqrt(variance_product)
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
