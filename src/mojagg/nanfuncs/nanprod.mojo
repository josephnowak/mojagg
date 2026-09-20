"""NaN-aware product operation for the guvectorize driver."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import load_block_or_identity
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def nan_product_contiguous[
    dtype: DType
](values: Span[Scalar[dtype], ImmUntrackedOrigin]) -> Scalar[dtype]:
    """Reduce one contiguous core with a SIMD product accumulator."""

    comptime width = simd_width_of[dtype]()
    var product = SIMD[dtype, width](1)
    var one = SIMD[dtype, width](1)
    var pointer = values.unsafe_ptr()

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {imm pointer, mut product, imm one}:
        var block = load_block_or_identity[dtype, width](
            pointer, i, evl, Scalar[dtype](1)
        )
        comptime if dtype.is_floating_point():
            product *= isnan(block).select(one, block)
        else:
            product *= block

    vectorize[width, unroll_factor=1](len(values), step)
    return product.reduce_mul()


@fieldwise_init
struct NanProd[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """Multiply finite values and use one as the empty/all-NaN identity."""

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
        output.write_span()[0] = nan_product_contiguous[Self.dtype](values)
