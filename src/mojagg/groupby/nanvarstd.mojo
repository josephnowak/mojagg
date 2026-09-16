"""Grouped NaN-aware variance and standard deviation kernels."""

from std.algorithm import vectorize
from std.math import isnan, sqrt
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
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

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
        GUTensor[DType.int64, True, CoreSpec[Dim[1]]],
    ]

    @always_inline
    @staticmethod
    def _add_lane(
        sums: Pointer[mut=True, Scalar[Self.value_t], _],
        sums_of_squares: Pointer[mut=True, Scalar[Self.value_t], _],
        counts: Pointer[mut=True, Scalar[DType.int64], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        comptime assert (
            Self.value_t == DType.float32 or Self.value_t == DType.float64
        ), "group_nanvar and group_nanstd require float32 or float64"

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
        var value_input, label_input, output, squares_output, counts_output = (
            tensors
        )
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()
        var sums_of_squares = squares_output.write_span()
        var counts = counts_output.write_span()

        comptime width = simd_width_of[Self.value_t]() * 8
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()
        var squares_ptr = sums_of_squares.unsafe_ptr()
        var count_ptr = counts.unsafe_ptr()
        var ddof = Int64(self.ddof)

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
            imm squares_ptr,
            imm count_ptr,
        }:
            if evl == width:
                var value_block = value_ptr.unsafe_load[width=width](i)
                var label_block = label_ptr.unsafe_load[width=width](i)
                comptime for lane in range(width):
                    Self._add_lane(
                        destination_ptr,
                        squares_ptr,
                        count_ptr,
                        label_block[lane],
                        value_block[lane],
                    )
            else:
                comptime for lane in range(width):
                    if lane < evl:
                        Self._add_lane(
                            destination_ptr,
                            squares_ptr,
                            count_ptr,
                            label_ptr[unsafe_offset=i + lane],
                            value_ptr[unsafe_offset=i + lane],
                        )

        vectorize[width](len(values), step)

        def finalize[
            vector_width: Int
        ](
            i: Int,
            evl: Int,
        ) {
            imm destination_ptr, imm squares_ptr, imm count_ptr, imm ddof
        }:
            comptime for lane in range(width):
                if lane < evl:
                    var offset = i + lane
                    var count = count_ptr[unsafe_offset=offset]
                    var denom = count - ddof
                    if denom <= 0:
                        destination_ptr[unsafe_offset=offset] = nan_or_zero[
                            Self.value_t
                        ]()
                    else:
                        var count_value = Scalar[Self.value_t](count)
                        var denominator = Scalar[Self.value_t](denom)
                        var variance = (
                            squares_ptr[unsafe_offset=offset]
                            - destination_ptr[unsafe_offset=offset]
                            * destination_ptr[unsafe_offset=offset]
                            / count_value
                        ) / denominator
                        comptime if Self.is_std:
                            destination_ptr[unsafe_offset=offset] = sqrt(
                                variance
                            )
                        else:
                            destination_ptr[unsafe_offset=offset] = variance

        vectorize[width](len(destination), finalize)
