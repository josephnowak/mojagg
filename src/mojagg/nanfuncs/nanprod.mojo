"""NaN-aware product operation for the native-tuple GUFunc."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


@always_inline
def nan_product_contiguous[
    dtype: DType
](values: Span[Scalar[dtype], ImmUntrackedOrigin]) -> Scalar[dtype]:
    """Reduce one contiguous core with a SIMD product accumulator."""

    comptime width = simd_width_of[dtype]() * 8
    var product = SIMD[dtype, width](1)
    var one = SIMD[dtype, width](1)
    var pointer = values.unsafe_ptr()

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {imm pointer, mut product, imm one}:
        if evl == width:
            var block = pointer.unsafe_load[width=width](i)
            comptime if dtype.is_floating_point():
                product *= isnan(block).select(one, block)
            else:
                product *= block
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if not isnan(value):
                            product[lane] *= value
                    else:
                        product[lane] *= value

    vectorize[width](len(values), step)
    return product.reduce_mul()


@fieldwise_init
struct NanProd[dtype: DType](GUFuncOperation, ImplicitlyCopyable):
    """Multiply finite values and use one as the empty/all-NaN identity."""

    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime Tensors = Tuple[
        TensorArg[Self.value_dtype, False],
        TensorArg[Self.out_dtype, True],
    ]

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        var values = input.read_span()
        output.write_span()[0] = nan_product_contiguous[Self.dtype](values)
