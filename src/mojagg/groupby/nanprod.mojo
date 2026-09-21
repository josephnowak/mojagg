"""Grouped NaN-aware product over aligned values and labels."""

from std.algorithm import vectorize
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import load_block_or_identity
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
)
from mojagg.groupby.group_kernel import GroupKernel


@fieldwise_init
struct GroupNanProd[
    value_t: DType,
    label_t: DType,
](GroupKernel, ImplicitlyCopyable):
    """Multiply valid values into dense groups, using one as the identity."""

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    @always_inline
    @staticmethod
    def _multiply_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        var label = Int(label_value)
        comptime identity = Scalar[Self.value_t](1)

        comptime if Self.value_t.is_floating_point():
            destination[unsafe_offset=label] *= isnan(value).select(
                identity, value
            )
        else:
            destination[unsafe_offset=label] *= value

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()
        var destination_ptr = destination.unsafe_ptr()

        comptime width = 2
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
        }:
            var value_block = load_block_or_identity[Self.value_t, width](
                value_ptr, i, evl, Scalar[Self.value_t](1)
            )
            var label_block = load_block_or_identity[Self.label_t, width](
                label_ptr, i, evl, Scalar[Self.label_t](-1)
            )
            comptime for lane in range(width):
                var label = label_block[lane]
                if label < 0:
                    continue
                Self._multiply_lane(
                    destination_ptr,
                    label,
                    value_block[lane],
                )

        vectorize[width, unroll_factor=8](len(values), step)
