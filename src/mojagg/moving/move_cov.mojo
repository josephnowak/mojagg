"""NaN-aware trailing moving covariance."""

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
    load_expiring_pair,
    load_masked_pair,
)


@always_inline
def _move_cov_sequential[
    dtype: DType
](
    a_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    b_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute pairwise sample covariance with SIMD block deltas."""

    comptime width = simd_width_of[dtype]()
    var n = len(a_values)
    var input_offset = 0
    var a_sum = Float64(0.0)
    var b_sum = Float64(0.0)
    var product_sum = Float64(0.0)
    var count = Float64(0.0)
    var threshold = Float64(min_count)
    if threshold < 2.0:
        threshold = 2.0
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
        var delta_count = entering_a.counts - expiring_a.counts

        comptime for lane in range(width):
            if lane < active:
                a_sum += delta_a[lane]
                b_sum += delta_b[lane]
                product_sum += delta_products[lane]
                count += delta_count[lane]
                if count >= threshold:
                    var covariance = (product_sum - a_sum * b_sum / count) / (
                        count - 1.0
                    )
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = covariance.cast[dtype]()
                else:
                    destination_ptr[
                        unsafe_offset=input_offset + lane
                    ] = nan_or_zero[dtype]()
        input_offset += active


struct MoveCovKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD ``(n), (n) -> (n)`` trailing sample covariance."""

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
        _move_cov_sequential[Self.dtype](
            input_a.read_span(),
            input_b.read_span(),
            output.write_span(),
            self.window,
            self.min_count,
        )
