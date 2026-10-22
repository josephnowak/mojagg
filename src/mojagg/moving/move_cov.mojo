"""NaN-aware trailing moving covariance."""

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
def _move_cov_sequential[
    dtype: DType
](
    a_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    b_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    window: Int,
    min_count: Int,
):
    """Compute pairwise sample covariance with a scalar rolling recurrence."""

    var n = len(a_values)
    var a_sum = Scalar[dtype](0)
    var b_sum = Scalar[dtype](0)
    var product_sum = Scalar[dtype](0)
    var count = Scalar[dtype](0)
    var threshold = Scalar[dtype](min_count)
    if threshold < 2.0:
        threshold = 2.0
    var a_ptr = a_values.unsafe_ptr()
    var b_ptr = b_values.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    for i in range(n):
        var entering_a = a_ptr[unsafe_offset=i]
        var entering_b = b_ptr[unsafe_offset=i]
        var delta_a = Scalar[dtype](0)
        var delta_b = Scalar[dtype](0)
        var delta_product = Scalar[dtype](0)
        var delta_count = Scalar[dtype](0)
        if not (isnan(entering_a) or isnan(entering_b)):
            delta_a = entering_a
            delta_b = entering_b
            delta_product = entering_a * entering_b
            delta_count = 1
        if i >= window:
            var expiring_a = a_ptr[unsafe_offset=i - window]
            var expiring_b = b_ptr[unsafe_offset=i - window]
            if not (isnan(expiring_a) or isnan(expiring_b)):
                delta_a -= expiring_a
                delta_b -= expiring_b
                delta_product -= expiring_a * expiring_b
                delta_count -= 1
        a_sum += delta_a
        b_sum += delta_b
        product_sum += delta_product
        count += delta_count
        if count >= threshold:
            var covariance = (product_sum - a_sum * b_sum / count) / (
                count - 1.0
            )
            destination_ptr[unsafe_offset=i] = covariance
        else:
            destination_ptr[unsafe_offset=i] = nan_or_zero[dtype]()


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
