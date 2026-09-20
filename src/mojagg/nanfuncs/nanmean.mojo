"""NaN-aware mean operation for the guvectorize driver."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import load_block_or_identity, nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def nan_mean_contiguous[
    dtype: DType
](values: Span[Scalar[dtype], ImmUntrackedOrigin]) -> Tuple[
    Scalar[dtype], Scalar[DType.int64]
]:
    """Accumulate a contiguous core with dtype-native sum and integer count."""

    comptime width = simd_width_of[dtype]()
    var pointer = values.unsafe_ptr()
    var total = SIMD[dtype, width](0.0)
    var count = SIMD[DType.int64, width](0)
    var value_zero = SIMD[dtype, width](0.0)
    var count_zero = SIMD[DType.int64, width](0)
    var one = SIMD[DType.int64, width](1)

    def step[
        vector_width: Int
    ](
        i: Int,
        evl: Int,
    ) {
        imm pointer,
        mut total,
        mut count,
        imm value_zero,
        imm count_zero,
        imm one,
    }:
        var block = load_block_or_identity[dtype, width](
            pointer, i, evl, nan_or_zero[dtype]()
        )
        var missing = isnan(block)
        total += missing.select(value_zero, block)
        count += missing.select(count_zero, one)

    vectorize[width, unroll_factor=1](len(values), step)
    return (total.reduce_add(), count.reduce_add())


@fieldwise_init
struct NanMean[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """Accumulate and write the result in the requested dtype."""

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
            output.write_span()[0] = total / Scalar[Self.dtype](count)
