"""NaN-aware any predicate for one prepared contiguous core."""

from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@fieldwise_init
struct AnyNan[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """Write whether at least one value in the active core is NaN."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[DType.bool, True, CoreSpec[]],
    ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        var values = input.read_span()
        var result = False
        var n = len(values)

        comptime if Self.dtype.is_floating_point():
            comptime width = simd_width_of[Self.dtype]() * 8
            var pointer = values.unsafe_ptr()
            var i = 0
            while i + width <= n and not result:
                var block = pointer.unsafe_load[width=width](i)
                if isnan(block).reduce_or():
                    result = True
                    break
                i += width
            while i < n and not result:
                if isnan(pointer[unsafe_offset=i]):
                    result = True
                    break
                i += 1
        else:
            # Integer values cannot be NaN.
            result = False

        output.write_span()[0] = Scalar[DType.bool](result)
