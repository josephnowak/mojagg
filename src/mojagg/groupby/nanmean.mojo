"""Grouped NaN-aware mean over aligned value and label cores."""

from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.core.vectorize import vectorize_no_evl
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
        comptime if Self.value_t.is_floating_point():
            var value_block = SIMD[Self.value_t, 1](value)
            comptime zero_block = SIMD[Self.value_t, 1](0)
            var mask = isnan(value_block)
            var clean_value = mask.select(zero_block, value_block)[0]
            var valid_count = mask.select(
                SIMD[Self.label_t, 1](0), SIMD[Self.label_t, 1](1)
            )[0]
            destination[unsafe_offset=label] += clean_value
            counts[unsafe_offset=label] += valid_count
        else:
            destination[unsafe_offset=label] += value
            counts[unsafe_offset=label] += Scalar[Self.label_t](1)

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
        ](i: Int, _evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
            imm count_ptr,
        }:
            var value_block = value_ptr.unsafe_load[width=vector_width](i)
            var label_block = label_ptr.unsafe_load[width=vector_width](i)
            comptime for lane in range(vector_width):
                var label = label_block[lane]
                if label < 0:
                    continue
                Self._add_lane(
                    destination_ptr,
                    count_ptr,
                    label,
                    value_block[lane],
                )

        vectorize_no_evl[width, unroll_factor=8](len(values), step)

        def finalize[
            vector_width: Int
        ](i: Int, _evl: Int) {imm destination_ptr, imm count_ptr}:
            var value_block = destination_ptr.unsafe_load[width=vector_width](i)
            var count_block = count_ptr.unsafe_load[width=vector_width](i)
            var zero_block = SIMD[Self.label_t, vector_width](0)
            var empty = count_block.eq(zero_block)
            var count_values = count_block.cast[Self.value_t]()
            var mean_block = value_block / count_values
            mean_block = empty.select(
                SIMD[Self.value_t, vector_width](nan_or_zero[Self.value_t]()),
                mean_block,
            )
            destination_ptr.unsafe_store[width=vector_width](i, mean_block)

        vectorize_no_evl[simd_width_of[Self.value_t](), unroll_factor=8](
            len(destination), finalize
        )
