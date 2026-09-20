"""NaN-aware count operation for the guvectorize driver."""

from std.algorithm import vectorize
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import load_block_or_identity, nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@fieldwise_init
struct NanCount[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """Count non-NaN values in the first tensor and write one int64 result."""

    comptime value_dtype = Self.dtype
    comptime out_dtype = DType.int64
    comptime Signature = Tuple[
        GUTensor[Self.value_dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.out_dtype, True, CoreSpec[]],
    ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        var values = input.read_span()
        comptime if not Self.dtype.is_floating_point():
            output.write_span()[0] = Int64(len(values))
            return

        comptime width = simd_width_of[Self.dtype]()
        var pointer = values.unsafe_ptr()
        var count = SIMD[DType.int64, width](0)
        var zero = SIMD[DType.int64, width](0)
        var one = SIMD[DType.int64, width](1)

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {imm pointer, mut count, imm zero, imm one}:
            var block = load_block_or_identity[Self.dtype, width](
                pointer, i, evl, nan_or_zero[Self.dtype]()
            )
            count += isnan(block).select(zero, one)

        vectorize[width, unroll_factor=1](len(values), step)
        output.write_span()[0] = Int64(count.reduce_add())
