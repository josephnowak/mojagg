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
from std.math import isnan, pow
from std.sys.info import simd_width_of

from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def nan_sum_powered_value[
    dtype: DType,
    power: Int,
](value: Scalar[dtype]) -> Scalar[dtype]:
    """Return one valid input raised to the compile-time power."""

    comptime assert power >= 0, "nansum power must be non-negative"
    comptime if power == 0:
        return Scalar[dtype](1)
    elif power == 1:
        return value
    else:
        return pow(value, power)


@always_inline
def nan_sum_powered_block[
    dtype: DType,
    power: Int,
    width: Int,
](values: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Raise a full SIMD block to a non-negative compile-time power."""

    comptime assert power >= 0, "nansum power must be non-negative"
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
    comptime width = simd_width_of[dtype]() * 8
    var acc = SIMD[dtype, width](0)
    var pointer = values.unsafe_ptr()

    def step[vector_width: Int](i: Int, evl: Int) {imm pointer, mut acc}:
        if evl == width:
            var block = pointer.unsafe_load[width=width](i)
            comptime if power == 1:
                comptime if dtype.is_floating_point():
                    acc += isnan(block).select(SIMD[dtype, width](0), block)
                else:
                    acc += block
            else:
                var powered = nan_sum_powered_block[dtype, power, width](block)
                comptime if dtype.is_floating_point():
                    powered = isnan(block).select(
                        SIMD[dtype, width](0), powered
                    )
                acc += powered
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if not isnan(value):
                            comptime if power == 1:
                                acc[lane] += value
                            else:
                                acc[lane] += nan_sum_powered_value[
                                    dtype, power
                                ](value)
                    else:
                        comptime if power == 1:
                            acc[lane] += value
                        else:
                            acc[lane] += nan_sum_powered_value[dtype, power](
                                value
                            )

    vectorize[width](len(values), step)
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
