"""NaN-aware sum operation for the generic ``gufunc`` driver.

The driver owns all rank, axis, stride, scratch, and scheduling decisions.
``NanSum`` only receives a prepared one-dimensional read span and a writable
one-element output span.  A contiguous source core is borrowed directly; a
strided source core has already been copied by ``GUFunc`` into worker-local
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

from mojagg.core.tensor_view import (
    TensorArg,
)
from mojagg.drivers.gufunc import GUFuncOperation


@always_inline
def nan_sum_contiguous[
    dtype: DType
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
            comptime if dtype.is_floating_point():
                acc += isnan(block).select(SIMD[dtype, width](0), block)
            else:
                acc += block
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if not isnan(value):
                            acc[lane] += value
                    else:
                        acc[lane] += value

    vectorize[width](len(values), step)
    return acc.reduce_add()


@fieldwise_init
struct NanSum[dtype: DType](GUFuncOperation, ImplicitlyCopyable):
    """SIMD ``(n) -> ()`` nansum operation.

    ``Tensors`` is one fixed compile-time tuple containing the read input and
    writable output.  Keeping both views in one tuple gives every GUFunc
    operation one uniform ``apply(tensors)`` entry point while retaining
    heterogeneous dtypes and zero-copy output writes.
    """

    comptime Tensors = Tuple[
        TensorArg[Self.dtype, False],
        TensorArg[Self.dtype, True],
    ]

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        var values = input.read_span()
        var result = nan_sum_contiguous[Self.dtype](values)
        var destination = output.write_span()
        destination[0] = result
