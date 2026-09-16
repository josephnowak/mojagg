"""Grouped NaN-aware sum over aligned value and label cores.

``guvectorize`` prepares one values core, one labels core, and one output core
for each outer slice.  The operation only performs the grouped scatter for
that slice; labels and values are expected to be aligned and equally long.
"""

from std.algorithm import vectorize
from std.math import isnan, pow
from std.sys.info import simd_width_of

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
        comptime assert Self.power >= 0, "group power must be non-negative"
        var label = Int(label_value)
        if label < 0:
            return
        comptime if Self.value_t.is_floating_point():
            if isnan(value):
                return
        comptime if Self.power == 0:
            destination[unsafe_offset=label] += Scalar[Self.value_t](1)
        elif Self.power == 1:
            destination[unsafe_offset=label] += value
        else:
            destination[unsafe_offset=label] += pow(value, Self.power)

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()

        # Group scatter stores are scalar because each label selects an
        # arbitrary destination.  Load value/label pairs in SIMD blocks and
        # keep only the unavoidable per-lane scatter scalar.
        comptime width = simd_width_of[Self.value_t]() * 8
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
        }:
            if evl == width:
                var value_block = value_ptr.unsafe_load[width=width](i)
                var label_block = label_ptr.unsafe_load[width=width](i)
                comptime for lane in range(width):
                    Self._add_lane(
                        destination_ptr,
                        label_block[lane],
                        value_block[lane],
                    )
            else:
                comptime for lane in range(width):
                    if lane < evl:
                        Self._add_lane(
                            destination_ptr,
                            label_ptr[unsafe_offset=i + lane],
                            value_ptr[unsafe_offset=i + lane],
                        )

        vectorize[width](len(values), step)
