"""Grouped NaN-aware argmin and argmax kernels."""

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
struct GroupNanArgMinMax[
    value_t: DType,
    label_t: DType,
    is_max: Bool,
](GroupKernel, ImplicitlyCopyable):
    """Track the best value and its local flattened index per group."""

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    @always_inline
    @staticmethod
    def _is_valid(value: Scalar[Self.value_t]) -> Bool:
        """Integer lanes are always valid; float lanes skip NaN."""
        comptime if Self.value_t.is_floating_point():
            return not isnan(value)
        else:
            return True

    @always_inline
    @staticmethod
    def _is_unset(index_slot: Scalar[Self.value_t]) -> Bool:
        """An untouched index slot holds NaN for floats and -1 for ints."""
        comptime if Self.value_t.is_floating_point():
            return isnan(index_slot)
        else:
            return Bool(index_slot < Scalar[Self.value_t](0))

    @always_inline
    @staticmethod
    def _improves(
        value: Scalar[Self.value_t], best_value: Scalar[Self.value_t]
    ) -> Bool:
        """Whether `value` beats the incumbent for this kernel's direction."""
        comptime if Self.is_max:
            return Bool(value > best_value)
        else:
            return Bool(value < best_value)

    @always_inline
    @staticmethod
    def _update_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        best_values: Pointer[mut=True, Scalar[Self.value_t], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
        flat_index: Int,
    ):
        var label = Int(label_value)
        if label < 0 or not Self._is_valid(value):
            return

        var should_update = Self._is_unset(
            destination[unsafe_offset=label]
        ) or Self._improves(value, best_values[unsafe_offset=label])
        if not should_update:
            return

        destination[unsafe_offset=label] = Scalar[Self.value_t](flat_index)
        best_values[unsafe_offset=label] = value

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output, best_output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()
        var best_values = best_output.write_span()

        comptime width = simd_width_of[Self.value_t]() * 2
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()
        var best_ptr = best_values.unsafe_ptr()

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
            imm best_ptr,
        }:
            var value_block = load_block_or_identity[Self.value_t, width](
                value_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var label_block = load_block_or_identity[Self.label_t, width](
                label_ptr, i, evl, Scalar[Self.label_t](-1)
            )
            comptime for lane in range(width):
                Self._update_lane(
                    destination_ptr,
                    best_ptr,
                    label_block[lane],
                    value_block[lane],
                    i + lane,
                )

        vectorize[width](len(values), step)
