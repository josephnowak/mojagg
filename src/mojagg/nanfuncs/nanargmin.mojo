"""NaN-aware first-occurrence argmin and argmax operations."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import iota, isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import neg_inf_or_min, pos_inf_or_max
from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


@always_inline
def nan_arg_extreme_contiguous[
    dtype: DType,
    is_min: Bool,
](values: Span[Scalar[dtype], ImmUntrackedOrigin]) -> Int64:
    """Find the first SIMD-reduced minimum or maximum index."""

    comptime width = simd_width_of[dtype]() * 8
    var identity = pos_inf_or_max[dtype]() if is_min else neg_inf_or_min[
        dtype
    ]()
    var identity_vector = SIMD[dtype, width](identity)
    var lane_indices = iota[DType.int64, width]()
    var sentinel = SIMD[DType.int64, width](Int64(width))
    var best = identity
    var best_index = Int64(-1)
    var pointer = values.unsafe_ptr()

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {
        imm pointer,
        mut best,
        mut best_index,
        imm identity_vector,
        imm lane_indices,
        imm sentinel,
    }:
        if evl == width:
            var block = pointer.unsafe_load[width=width](i)
            var block_values = block
            comptime if dtype.is_floating_point():
                var missing = isnan(block)
                block_values = missing.select(identity_vector, block)

            var block_best: Scalar[dtype]
            comptime if is_min:
                block_best = block_values.reduce_min()
            else:
                block_best = block_values.reduce_max()

            var matches = block_values.eq(SIMD[dtype, width](block_best))
            comptime if dtype.is_floating_point():
                matches = isnan(block).select(
                    SIMD[DType.bool, width](fill=False), matches
                )
            var candidates = matches.select(lane_indices, sentinel)
            var local_index = candidates.reduce_min()
            if local_index < Int64(width):
                var absolute_index = Int64(i) + local_index
                if best_index < 0:
                    best = block_best
                    best_index = absolute_index
                elif is_min and block_best < best:
                    best = block_best
                    best_index = absolute_index
                elif not is_min and block_best > best:
                    best = block_best
                    best_index = absolute_index
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if isnan(value):
                            continue
                    var absolute_index = Int64(i + lane)
                    if best_index < 0:
                        best = value
                        best_index = absolute_index
                    elif is_min and value < best:
                        best = value
                        best_index = absolute_index
                    elif not is_min and value > best:
                        best = value
                        best_index = absolute_index

    vectorize[width](len(values), step)
    return best_index


@fieldwise_init
struct NanArgExtrema[
    dtype: DType,
    is_min: Bool,
](GUFuncOperation, ImplicitlyCopyable):
    comptime value_dtype = Self.dtype
    comptime out_dtype = DType.int64
    comptime Tensors = Tuple[
        TensorArg[Self.value_dtype, False],
        TensorArg[Self.out_dtype, True],
    ]

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        var index = nan_arg_extreme_contiguous[Self.dtype, Self.is_min](
            input.read_span()
        )
        output.write_span()[0] = index


comptime NanArgMin[dtype: DType] = NanArgExtrema[dtype, True]
