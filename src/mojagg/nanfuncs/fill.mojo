"""Forward and backward fill operations for the generic GUFunc driver.

The driver presents one contiguous input and output span for every logical
core. A fill is stateful along that span, so it deliberately stays scalar for
floating point values; integer specializations are a bulk copy because they
cannot contain NaNs.
"""

from std.math import isnan
from std.memory import unsafe_memcpy

from mojagg.core.numeric import nan_or_zero
from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


struct FillKernel[
    dtype: DType,
    backward: Bool = False,
](GUFuncOperation, ImplicitlyCopyable):
    """Fill NaN runs in one core.

    ``limit < 0`` permits an unlimited run after a valid value. A nonnegative
    limit permits at most that many missing values to inherit the last valid
    value. Leading missing values remain NaN for floating point input.
    """

    comptime Tensors = Tuple[
        TensorArg[Self.dtype, False],
        TensorArg[Self.dtype, True],
    ]

    var limit: Int

    def __init__(out self, limit: Int = -1):
        self.limit = limit

    @always_inline
    @staticmethod
    def _start(n: Int) -> Int:
        comptime if Self.backward:
            return n - 1
        else:
            return 0

    @always_inline
    @staticmethod
    def _step() -> Int:
        comptime if Self.backward:
            return -1
        else:
            return 1

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var source_view = tensors[0].copy()
        var destination_view = tensors[1].copy()
        var source = source_view.read_span()
        var destination = destination_view.write_span()
        var n = len(source)

        comptime if Self.dtype.is_floating_point():
            var current = nan_or_zero[Self.dtype]()
            var allowed = self.limit if self.limit >= 0 else n
            var remaining = allowed
            var i = Self._start(n)
            var step = Self._step()
            while i >= 0 and i < n:
                var value = source.unsafe_ptr()[unsafe_offset=i]
                if isnan(value):
                    if remaining <= 0:
                        current = nan_or_zero[Self.dtype]()
                    remaining -= 1
                else:
                    current = value
                    remaining = allowed
                destination.unsafe_ptr()[unsafe_offset=i] = current
                i += step
        else:
            unsafe_memcpy(
                dest=destination.unsafe_ptr(),
                src=source.unsafe_ptr(),
                count=n,
            )


comptime FFill[dtype: DType] = FillKernel[dtype, False]
comptime BFill[dtype: DType] = FillKernel[dtype, True]
