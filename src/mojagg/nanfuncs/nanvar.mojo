"""NaN-aware variance and standard deviation operations."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan, sqrt
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation
from mojagg.nanfuncs.nanmean import nan_mean_contiguous


@always_inline
def nan_squared_deviation_contiguous[
    dtype: DType
](values: Span[Scalar[dtype], ImmUntrackedOrigin], mean: Float64,) -> Float64:
    """Accumulate squared deviations with a float64 SIMD accumulator."""

    comptime width = simd_width_of[dtype]() * 8
    var squared = SIMD[DType.float64, width](0.0)
    var zero = SIMD[DType.float64, width](0.0)
    var mean_vector = SIMD[DType.float64, width](mean)
    var pointer = values.unsafe_ptr()

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {
        imm pointer, mut squared, imm zero, imm mean_vector, imm mean
    }:
        if evl == width:
            var block = pointer.unsafe_load[width=width](i)
            var widened = block.cast[DType.float64]()
            var delta = widened - mean_vector
            comptime if dtype.is_floating_point():
                squared += isnan(block).select(zero, delta * delta)
            else:
                squared += delta * delta
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if isnan(value):
                            continue
                    var delta = Float64(value) - mean
                    squared[lane] += delta * delta

    vectorize[width](len(values), step)
    return squared.reduce_add()


struct NanVar[
    dtype: DType,
    take_sqrt: Bool = False,
](GUFuncOperation, ImplicitlyCopyable):
    """Compute a two-pass float64 accumulation over one prepared core."""

    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime Tensors = Tuple[
        TensorArg[Self.value_dtype, False],
        TensorArg[Self.out_dtype, True],
    ]

    var ddof: Int

    def __init__(out self, ddof: Int = 1):
        self.ddof = ddof

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        var values = input.read_span()
        var state = nan_mean_contiguous[Self.dtype](values)
        var total = state[0]
        var count = state[1]

        if count <= Int64(self.ddof):
            output.write_span()[0] = nan_or_zero[Self.dtype]()
            return

        var mean = total / Float64(count)
        var squared = nan_squared_deviation_contiguous[Self.dtype](values, mean)

        var variance = squared / Float64(count - Int64(self.ddof))
        if variance < 0.0:
            variance = 0.0
        comptime if Self.take_sqrt:
            output.write_span()[0] = sqrt(variance).cast[Self.dtype]()
        else:
            output.write_span()[0] = variance.cast[Self.dtype]()
