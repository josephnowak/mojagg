"""NaN-aware sum operation for the ``guvectorize`` driver.

The driver owns all rank, axis, stride, scratch, and scheduling decisions.
``NanSum`` only receives a prepared one-dimensional read span and a writable
one-element output span.  A contiguous source core is borrowed directly; a
strided source core has already been copied by ``guvectorize`` into worker-local
contiguous storage.  That makes the SIMD loop below identical for both cases
and keeps layout branches out of the numerical kernel.

The operation is specialized for each supported dtype.  Floating-point
specializations ignore NaNs and integer specializations erase the NaN branch
at compile time.  Empty and all-NaN cores naturally produce zero, matching
``numpy.nansum`` and numbagg's nansum contract.
"""

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
def nan_sum_powered_block[
    dtype: DType,
    power: Int,
    width: Int,
](values: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Raise a full SIMD block to a non-negative compile-time power."""

    comptime if power == 0:
        return SIMD[dtype, width](1)
    elif power == 1:
        return values
    else:
        var result = values
        comptime for _ in range(power - 1):
            result *= values
        return result


@always_inline
def nan_sum_contiguous[
    dtype: DType,
    power: Int = 1,
](values: Span[Scalar[dtype], ImmUntrackedOrigin]) -> Scalar[dtype]:
    """Reduce one already-contiguous core with a wide SIMD accumulator."""

    # Eight independent native-width groups retain the measured reduction
    # throughput of the previous nansum kernel while leaving the driver free
    # to materialize arbitrary input layouts once per slice.
    comptime width = simd_width_of[dtype]()
    var acc = SIMD[dtype, width](0)
    comptime zero = SIMD[dtype, width](0)
    var pointer = values.unsafe_ptr()

    def step[vector_width: Int](i: Int, evl: Int) {imm pointer, mut acc}:
        var block = load_block_or_identity[dtype, width](
            pointer, i, evl, Scalar[dtype](0)
        )
        block = nan_sum_powered_block[dtype, power, width](block)
        comptime if dtype.is_floating_point():
            acc += isnan(block).select(zero, block)
        else:
            acc += block

    vectorize[width, unroll_factor=1](len(values), step)
    return acc.reduce_add()


@fieldwise_init
struct NanSum[dtype: DType, power: Int = 1](GUFuncKernel, ImplicitlyCopyable):
    """SIMD ``(n) -> ()`` nansum operation."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[]],
    ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        var values = input.read_span()
        var result = nan_sum_contiguous[Self.dtype, Self.power](values)
        output.write_span()[0] = result
