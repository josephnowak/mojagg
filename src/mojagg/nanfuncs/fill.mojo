"""Forward and backward fill operations for the generic GUFunc driver.

The driver presents one contiguous input and output span for every logical
core. A fill is stateful along that span, so it uses SIMD chunk loading with
an internal recurrence step; integer specializations are a bulk copy because
they cannot contain NaNs.
"""

from std.math import isnan
from std.memory import unsafe_memcpy
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


struct FillKernel[
    dtype: DType,
    backward: Bool = False,
](GUFuncKernel, ImplicitlyCopyable):
    """Fill NaN runs in one contiguous core span using unified chunked SIMD loads.

    ``limit < 0`` permits an unlimited run after a valid value. A nonnegative
    limit permits at most that many missing values to inherit the last valid
    value. Leading missing values remain NaN for floating point input.
    """

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[Dim[0]]],
    ]

    var limit: Int

    def __init__(out self, limit: Int = -1):
        self.limit = limit

    @always_inline
    def _apply_step(
        self,
        value: Scalar[Self.dtype],
        mut current: Scalar[Self.dtype],
        mut remaining: Int,
        allowed: Int,
    ) -> Scalar[Self.dtype]:
        """Internal recurrence step updating state and returning output."""
        if isnan(value):
            if remaining <= 0:
                current = nan_or_zero[Self.dtype]()
            remaining -= 1
        else:
            current = value
            remaining = allowed
        return current

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var source_view, destination_view = tensors
        var source = source_view.read_span()
        var destination = destination_view.write_span()
        var n = len(source)

        if n <= 0:
            return

        # Integers cannot contain NaNs; delegate to bulk copy
        comptime if not Self.dtype.is_floating_point():
            unsafe_memcpy(
                dest=destination.unsafe_ptr(),
                src=source.unsafe_ptr(),
                count=n,
            )
            return

        var current = nan_or_zero[Self.dtype]()
        var allowed = self.limit if self.limit >= 0 else n
        var remaining = allowed
        var src_ptr = source.unsafe_ptr()
        var dest_ptr = destination.unsafe_ptr()

        # The recurrence is serial, so a plain scalar loop lets the compiler
        # keep the state in registers without SIMD lane shuffles.
        comptime if Self.backward:
            var i = n - 1
            while i >= 0:
                dest_ptr[unsafe_offset=i] = self._apply_step(
                    src_ptr[unsafe_offset=i],
                    current,
                    remaining,
                    allowed,
                )
                i -= 1
        else:
            for i in range(n):
                dest_ptr[unsafe_offset=i] = self._apply_step(
                    src_ptr[unsafe_offset=i],
                    current,
                    remaining,
                    allowed,
                )


comptime FFill[dtype: DType] = FillKernel[dtype, False]
comptime BFill[dtype: DType] = FillKernel[dtype, True]
