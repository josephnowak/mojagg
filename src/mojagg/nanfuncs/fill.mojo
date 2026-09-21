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
    @staticmethod
    def _chunk_start(n: Int, width: Int) -> Int:
        comptime if Self.backward:
            return n - width
        else:
            return 0

    @always_inline
    @staticmethod
    def _chunk_step(width: Int) -> Int:
        comptime if Self.backward:
            return -width
        else:
            return width

    @always_inline
    @staticmethod
    def _chunk_valid(offset: Int, n: Int, width: Int) -> Bool:
        comptime if Self.backward:
            return offset >= 0
        else:
            return offset + width <= n

    @always_inline
    @staticmethod
    def _tail_start(offset: Int, width: Int) -> Int:
        comptime if Self.backward:
            return offset + width - 1
        else:
            return offset

    @always_inline
    @staticmethod
    def _tail_step() -> Int:
        comptime if Self.backward:
            return -1
        else:
            return 1

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

        comptime width = 4
        var current = nan_or_zero[Self.dtype]()
        var allowed = self.limit if self.limit >= 0 else n
        var remaining = allowed
        var src_ptr = source.unsafe_ptr()
        var dest_ptr = destination.unsafe_ptr()

        # -----------------------------------------------------------------
        # 1. Unified SIMD Chunk Loop
        # -----------------------------------------------------------------
        var offset = Self._chunk_start(n, width)
        var chunk_step = Self._chunk_step(width)

        while Self._chunk_valid(offset, n, width):
            var block = src_ptr.unsafe_load[width=width](offset)

            comptime if Self.backward:
                comptime for i in range(width):
                    comptime lane = width - 1 - i
                    block[lane] = self._apply_step(
                        block[lane],
                        current,
                        remaining,
                        allowed,
                    )
            else:
                comptime for i in range(width):
                    block[i] = self._apply_step(
                        block[i],
                        current,
                        remaining,
                        allowed,
                    )
            dest_ptr.unsafe_store[width=width](offset, block)

            offset += chunk_step

        # -----------------------------------------------------------------
        # 2. Unified Scalar Tail Loop
        # -----------------------------------------------------------------
        var tail_i = Self._tail_start(offset, width)
        var tail_step = Self._tail_step()

        while tail_i >= 0 and tail_i < n:
            dest_ptr[unsafe_offset=tail_i] = self._apply_step(
                src_ptr[unsafe_offset=tail_i],
                current,
                remaining,
                allowed,
            )
            tail_i += tail_step


comptime FFill[dtype: DType] = FillKernel[dtype, False]
comptime BFill[dtype: DType] = FillKernel[dtype, True]
