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

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    var ddof: Int
    var counts: Preallocated[Self.label_t]
    var sums: Preallocated[Self.value_t]
    var sums_of_squares: Preallocated[Self.value_t]

    def __init__(out self, ddof: Int):
        self.ddof = ddof
        self.counts = Preallocated[Self.label_t]()
        self.sums = Preallocated[Self.value_t]()
        self.sums_of_squares = Preallocated[Self.value_t]()

    def __init__(out self, *, copy: Self):
        self.ddof = copy.ddof
        self.counts = Preallocated[Self.label_t]()
        self.sums = Preallocated[Self.value_t]()
        self.sums_of_squares = Preallocated[Self.value_t]()

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
        var clean_value: Scalar[Self.value_t]
        var count: Scalar[Self.label_t]
        comptime if Self.value_t.is_floating_point():
            var value_block = SIMD[Self.value_t, 1](value)
            var mask = isnan(value_block)
            clean_value = mask.select(SIMD[Self.value_t, 1](0), value_block)[0]
            count = mask.select(
                SIMD[Self.label_t, 1](0), SIMD[Self.label_t, 1](1)
            )[0]
        else:
            clean_value = value
            count = Scalar[Self.label_t](1)
        sums[unsafe_offset=label] += clean_value
        sums_of_squares[unsafe_offset=label] += clean_value * clean_value
        counts[unsafe_offset=label] += count

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()
        var count_ptr = self.counts.get_ptr(
            len(destination), Scalar[Self.label_t](0)
        )
        var sums_ptr = self.sums.get_ptr(
            len(destination), Scalar[Self.value_t](0)
        )
        var squares_ptr = self.sums_of_squares.get_ptr(
            len(destination), Scalar[Self.value_t](0)
        )

        comptime width = 2
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()
        var ddof = self.ddof
        comptime width_finalize = simd_width_of[Self.value_t]()

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm sums_ptr,
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
                var label = label_block[lane]
                if label < 0:
                    continue
                Self._add_lane(
                    sums_ptr,
                    squares_ptr,
                    count_ptr,
                    label,
                    value_block[lane],
                )

        vectorize[width, unroll_factor=8](len(values), step)

        def finalize[
            vector_width: Int
        ](
            i: Int,
            evl: Int,
        ) {
            imm destination_ptr,
            imm sums_ptr,
            imm squares_ptr,
            imm count_ptr,
            imm ddof,
        }:
            var sum_block = load_block_or_identity[
                Self.value_t, width_finalize
            ](sums_ptr, i, evl, Scalar[Self.value_t](0))
            var squares_block = load_block_or_identity[
                Self.value_t, width_finalize
            ](squares_ptr, i, evl, Scalar[Self.value_t](0))
            var count_block = load_block_or_identity[
                Self.label_t, width_finalize
            ](count_ptr, i, evl, Scalar[Self.label_t](0))
            comptime for lane in range(width_finalize):
                if lane < evl:
                    var count = Int(count_block[lane])
                    if count <= ddof:
                        destination_ptr[unsafe_offset=i + lane] = nan_or_zero[
                            Self.value_t
                        ]()
                    else:
                        var count_value = Scalar[Self.value_t](count)
                        var denominator = Scalar[Self.value_t](count - ddof)
                        var variance = (
                            squares_block[lane]
                            - sum_block[lane] * sum_block[lane] / count_value
                        ) / denominator
                        comptime if Self.is_std:
                            variance = sqrt(variance)
                        destination_ptr[unsafe_offset=i + lane] = variance

        vectorize[width_finalize, unroll_factor=1](len(destination), finalize)
