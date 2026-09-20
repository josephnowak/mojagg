"""Grouped NaN-aware mean over aligned value and label cores."""

from std.algorithm import vectorize
from std.math import isnan
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
struct GroupNanMean[
    value_t: DType,
    label_t: DType,
](GroupKernel, ImplicitlyCopyable):
    """Accumulate sums and counts, then finalize each group mean."""

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    var counts: Preallocated[Self.label_t]

    def __init__(out self):
        self.counts = Preallocated[Self.label_t]()

    def __init__(out self, *, copy: Self):
        self.counts = Preallocated[Self.label_t]()

    @always_inline
    @staticmethod
    def _add_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        counts: Pointer[mut=True, Scalar[Self.label_t], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        var label = Int(label_value)
        if label < 0:
            return
        if isnan(value):
            return
        destination[unsafe_offset=label] += value
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

        comptime width = 2
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
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
                    count_ptr,
                    label_block[lane],
                    value_block[lane],
                )

        vectorize[width, unroll_factor=8](len(values), step)

        def finalize[
            vector_width: Int
        ](i: Int, evl: Int,) {imm destination_ptr, imm count_ptr}:
            var value_block = load_block_or_identity[Self.value_t, width](
                destination_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var count_block = load_block_or_identity[Self.label_t, width](
                count_ptr, i, evl, Scalar[Self.label_t](0)
            )
            var count_values = count_block.cast[Self.value_t]()
            var mean_block = value_block / count_values
            var empty = isnan(SIMD[Self.value_t, width](0) / count_values)
            var nan_block = SIMD[Self.value_t, width](
                nan_or_zero[Self.value_t]()
            )
            mean_block = empty.select(nan_block, mean_block)
            if evl == width:
                destination_ptr.unsafe_store[width=width](i, mean_block)
            else:
                comptime for lane in range(width):
                    if lane < evl:
                        destination_ptr[unsafe_offset=i + lane] = mean_block[
                            lane
                        ]

        comptime width_finalize = simd_width_of[Self.value_t]
        vectorize[width_finalize, unroll_factor=8](len(destination), finalize)
