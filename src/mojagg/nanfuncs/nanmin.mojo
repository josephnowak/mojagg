"""NaN-aware minimum and maximum operations for the native-tuple GUFunc."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan, max, min
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero, neg_inf_or_min, pos_inf_or_max
from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


@always_inline
def nan_extreme_contiguous[
    dtype: DType,
    is_min: Bool,
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
) -> Tuple[
    Scalar[dtype], Float64
]:
    """Reduce one contiguous core with SIMD min/max and a valid count."""

    comptime width = simd_width_of[dtype]() * 8
    var identity = pos_inf_or_max[dtype]() if is_min else neg_inf_or_min[
        dtype
    ]()
    var identity_vector = SIMD[dtype, width](identity)
    var accumulator = identity_vector
    var valid_count = SIMD[DType.float64, width](0.0)
    var zero = SIMD[DType.float64, width](0.0)
    var one = SIMD[DType.float64, width](1.0)
    var pointer = values.unsafe_ptr()

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {
        imm pointer,
        mut accumulator,
        mut valid_count,
        imm identity_vector,
        imm zero,
        imm one,
    }:
        if evl == width:
            var block = pointer.unsafe_load[width=width](i)
            var block_values = block
            comptime if dtype.is_floating_point():
                var missing = isnan(block)
                block_values = missing.select(identity_vector, block)
                valid_count += missing.select(zero, one)
            else:
                valid_count += one

            comptime if is_min:
                accumulator = min(accumulator, block_values)
            else:
                accumulator = max(accumulator, block_values)
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if isnan(value):
                            continue
                    valid_count[lane] += 1.0
                    comptime if is_min:
                        accumulator[lane] = min(accumulator[lane], value)
                    else:
                        accumulator[lane] = max(accumulator[lane], value)

    vectorize[width](len(values), step)
    var result = (
        accumulator.reduce_min() if is_min else accumulator.reduce_max()
    )
    return (result, valid_count.reduce_add())


@fieldwise_init
struct NanExtrema[
    dtype: DType,
    is_min: Bool,
    result_dtype: DType = dtype,
](GUFuncOperation, ImplicitlyCopyable):
    """Select the first finite minimum or maximum in one prepared core."""

    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.result_dtype
    comptime Tensors = Tuple[
        TensorArg[Self.value_dtype, False],
        TensorArg[Self.out_dtype, True],
    ]

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        var values = input.read_span()
        var state = nan_extreme_contiguous[Self.dtype, Self.is_min](values)
        if state[1] == 0.0:
            output.write_span()[0] = nan_or_zero[Self.out_dtype]()
        else:
            output.write_span()[0] = Scalar[Self.out_dtype](state[0])


comptime NanMin[
    dtype: DType,
    result_dtype: DType = dtype,
] = NanExtrema[dtype, True, result_dtype]
