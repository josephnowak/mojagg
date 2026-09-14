"""NaN-aware mean operation for the guvectorize driver."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def nan_mean_contiguous[
    dtype: DType
](values: Span[Scalar[dtype], ImmUntrackedOrigin]) -> Tuple[Float64, Int64]:
    """Accumulate a contiguous core with float64 SIMD sum and count."""

    comptime width = simd_width_of[dtype]() * 8
    var pointer = values.unsafe_ptr()
    var total = SIMD[DType.float64, width](0.0)
    var count = SIMD[DType.float64, width](0.0)
    var zero = SIMD[DType.float64, width](0.0)
    var one = SIMD[DType.float64, width](1.0)

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {imm pointer, mut total, mut count, imm zero, imm one}:
        if evl == width:
            var block = pointer.unsafe_load[width=width](i)
            var widened = block.cast[DType.float64]()

            comptime if dtype.is_floating_point():
                var missing = isnan(block)
                total += missing.select(zero, widened)
                count += missing.select(zero, one)
            else:
                total += widened
                count += one
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if not isnan(value):
                            total[lane] += Float64(value)
                            count[lane] += 1.0
                    else:
                        total[lane] += Float64(value)
                        count[lane] += 1.0

    vectorize[width](len(values), step)
    return (Float64(total.reduce_add()), Int64(count.reduce_add()))


@fieldwise_init
struct NanMean[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """Accumulate in float64 and write the result in the requested dtype."""

    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime Signature = Tuple[
        GUTensor[Self.value_dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.out_dtype, True, CoreSpec[]],
    ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        var values = input.read_span()
        var state = nan_mean_contiguous[Self.dtype](values)
        var total = state[0]
        var count = state[1]
        if count == 0:
            output.write_span()[0] = nan_or_zero[Self.dtype]()
        else:
            output.write_span()[0] = (total / Float64(count)).cast[Self.dtype]()
