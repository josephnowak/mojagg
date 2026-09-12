"""NaN-aware count operation for the native-tuple GUFunc."""

from std.algorithm import vectorize
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


@fieldwise_init
struct NanCount[dtype: DType](GUFuncOperation, ImplicitlyCopyable):
    """Count non-NaN values in the first tensor and write one int64 result."""

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
        var values = input.read_span()
        comptime if not Self.dtype.is_floating_point():
            output.write_span()[0] = Int64(len(values))
            return

        comptime width = simd_width_of[Self.dtype]()
        var pointer = values.unsafe_ptr()
        var count = SIMD[DType.float64, width](0.0)
        var zero = SIMD[DType.float64, width](0.0)
        var one = SIMD[DType.float64, width](1.0)

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {imm pointer, mut count, imm zero, imm one}:
            if evl == width:
                var block = pointer.unsafe_load[width=width](i)
                count += isnan(block).select(zero, one)
            else:
                comptime for lane in range(width):
                    if lane < evl and not isnan(
                        pointer[unsafe_offset=i + lane]
                    ):
                        count[lane] += 1.0

        vectorize[width](len(values), step)
        output.write_span()[0] = Int64(count.reduce_add())
