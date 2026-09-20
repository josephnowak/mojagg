"""Grouped NaN-aware variance and standard deviation kernels."""

from std.algorithm import vectorize
from std.math import isnan, sqrt
from std.sys.info import simd_width_of

from mojagg.core.numeric import load_block_or_identity, nan_or_zero
from mojagg.core.preallocated import Preallocated
from mojagg.groupby.group_kernel import GroupKernel
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
)


@fieldwise_init
struct GroupNanVarStd[
    value_t: DType,
    label_t: DType,
    is_std: Bool,
](GroupKernel, ImplicitlyCopyable):
    """Accumulate sum, sum of squares, and count for each group."""

    var ddof: Int
    var counts: Preallocated[Self.label_t]
    var sums_of_squares: Preallocated[Self.value_t]

    def __init__(out self, ddof: Int):
        self.ddof = ddof
        self.counts = Preallocated[Self.label_t]()
        self.sums_of_squares = Preallocated[Self.value_t]()

    def __init__(out self, *, copy: Self):
        self.ddof = copy.ddof
        self.counts = Preallocated[Self.label_t]()
        self.sums_of_squares = Preallocated[Self.value_t]()

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    @always_inline
    @staticmethod
    def _add_lane(
        sums: Pointer[mut=True, Scalar[Self.value_t], _],
        sums_of_squares: Pointer[mut=True, Scalar[Self.value_t], _],
        counts: Pointer[mut=True, Scalar[Self.label_t], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        var label = Int(label_value)
        if label < 0:
            return
        if isnan(value):
            return
        sums[unsafe_offset=label] += value
        sums_of_squares[unsafe_offset=label] += value * value
        counts[unsafe_offset=label] += 1

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()
        var count_ptr = self.counts.get_ptr(
            len(destination), Scalar[Self.label_t](0)
        )
        var squares_ptr = self.sums_of_squares.get_ptr(
            len(destination), Scalar[Self.value_t](0)
        )

        comptime width = 2
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()
        var ddof = Scalar[Self.label_t](self.ddof)

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
            imm squares_ptr,
            imm count_ptr,
        }:
            var value_block = load_block_or_identity[Self.value_t, width](
                value_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var label_block = load_block_or_identity[Self.label_t, width](
                label_ptr, i, evl, Scalar[Self.label_t](-1)
            )
            comptime for lane in range(width):
                Self._add_lane(
                    destination_ptr,
                    squares_ptr,
                    count_ptr,
                    label_block[lane],
                    value_block[lane],
                )

        vectorize[width, unroll_factor=8](len(values), step)

        def finalize[
            vector_width: Int
        ](
            i: Int,
            evl: Int,
        ) {
            imm destination_ptr, imm squares_ptr, imm count_ptr, imm ddof
        }:
            var sum_block = load_block_or_identity[Self.value_t, width](
                destination_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var squares_block = load_block_or_identity[Self.value_t, width](
                squares_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var count_block = load_block_or_identity[Self.label_t, width](
                count_ptr, i, evl, Scalar[Self.label_t](0)
            )
            var denominator_block = count_block - SIMD[Self.label_t, width](
                ddof
            )
            var count_values = count_block.cast[Self.value_t]()
            var denominator_values = denominator_block.cast[Self.value_t]()
            var variance_block = (
                squares_block - sum_block * sum_block / count_values
            ) / denominator_values
            var invalid = isnan(sqrt(denominator_values) / denominator_values)
            var nan_block = SIMD[Self.value_t, width](
                nan_or_zero[Self.value_t]()
            )
            comptime if Self.is_std:
                variance_block = sqrt(variance_block)
            variance_block = invalid.select(nan_block, variance_block)
            if evl == width:
                destination_ptr.unsafe_store[width=width](i, variance_block)
            else:
                comptime for lane in range(width):
                    if lane < evl:
                        destination_ptr[
                            unsafe_offset=i + lane
                        ] = variance_block[lane]

        comptime width_finalize = simd_width_of[Self.value_t]()
        vectorize[width_finalize, unroll_factor=8](len(destination), finalize)
