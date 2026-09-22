"""Grouped NaN-aware sum over aligned value and label cores.

``guvectorize`` prepares one values core, one labels core, and one output core
for each outer slice.  The operation only performs the grouped scatter for
that slice; labels and values are expected to be aligned and equally long.
"""

from std.math import isnan, pow
from std.sys.info import simd_width_of

from mojagg.core.vectorize import vectorize_no_evl
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
)
from mojagg.groupby.group_kernel import GroupKernel


@fieldwise_init
struct GroupNanSum[
    value_t: DType,
    label_t: DType,
    power: Int,
](GroupKernel, ImplicitlyCopyable):
    """Accumulate a compile-time power of valid values into dense groups."""

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    @always_inline
    @staticmethod
    def _add_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        var label = Int(label_value)
        var powered_value: Scalar[Self.value_t]
        comptime if Self.value_t.is_floating_point():
            var value_block = SIMD[Self.value_t, 1](value)
            var clean_value = isnan(value_block).select(
                SIMD[Self.value_t, 1](0), value_block
            )[0]
            powered_value = clean_value
        else:
            powered_value = value
        comptime if Self.power == 1:
            destination[unsafe_offset=label] += powered_value
        else:
            destination[unsafe_offset=label] += pow(powered_value, Self.power)

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()

        # Group scatter stores are scalar because each label selects an
        # arbitrary destination.  Load value/label pairs in SIMD blocks and
        # keep only the unavoidable per-lane scatter scalar.
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
        }:
            var value_block = value_ptr.unsafe_load[width=vector_width](i)
            var label_block = label_ptr.unsafe_load[width=vector_width](i)
            comptime for lane in range(vector_width):
                var label = label_block[lane]
                if label < 0:
                    continue
                Self._add_lane(
                    destination_ptr,
                    label,
                    value_block[lane],
                )

        vectorize_no_evl[width, unroll_factor=8](len(values), step)
