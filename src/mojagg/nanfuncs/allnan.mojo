"""NaN-aware all predicate for one prepared contiguous core."""

from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


@fieldwise_init
struct AllNan[dtype: DType](GUFuncOperation, ImplicitlyCopyable):
    """Write whether every value in the active core is NaN."""

    comptime Tensors = Tuple[
        TensorArg[Self.dtype, False],
        TensorArg[DType.bool, True],
    ]

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        var values = input.read_span()
        var result = True
        var n = len(values)

        comptime if Self.dtype.is_floating_point():
            comptime width = simd_width_of[Self.dtype]() * 8
            var pointer = values.unsafe_ptr()
            var i = 0
            while i + width <= n and result:
                var block = pointer.unsafe_load[width=width](i)
                if not isnan(block).reduce_and():
                    result = False
                    break
                i += width
            while i < n and result:
                if not isnan(pointer[unsafe_offset=i]):
                    result = False
                    break
                i += 1
        else:
            # Integer values cannot be NaN.  Preserve the empty identity.
            result = n == 0

        output.write_span()[0] = Scalar[DType.bool](result)
