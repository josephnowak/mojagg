"""Grouped NaN-aware argmin and argmax kernels."""

from std.algorithm import vectorize
from std.math import isnan
from std.sys.info import simd_width_of

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
        GUTensor[DType.int64, True, CoreSpec[Dim[1]]],
    ]

    @always_inline
    @staticmethod
    def _update_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        best_values: Pointer[mut=True, Scalar[Self.value_t], _],
        seen: Pointer[mut=True, Scalar[DType.int64], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
        flat_index: Int,
    ):
        var label = Int(label_value)
        if label < 0:
            return
        comptime if Self.value_t.is_floating_point():
            if isnan(value):
                return
        var has_best = seen[unsafe_offset=label] != 0

        var better = not has_best
        if has_best:
            comptime if Self.is_max:
                better = value > best_values[unsafe_offset=label]
            else:
                better = value < best_values[unsafe_offset=label]
        if better:
            destination[unsafe_offset=label] = Scalar[Self.value_t](flat_index)
            best_values[unsafe_offset=label] = value
            seen[unsafe_offset=label] = 1

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output, best_output, seen_output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()
        var best_values = best_output.write_span()
        var seen = seen_output.write_span()

        comptime width = simd_width_of[Self.value_t]() * 8
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()
        var best_ptr = best_values.unsafe_ptr()
        var seen_ptr = seen.unsafe_ptr()

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
            imm best_ptr,
            imm seen_ptr,
        }:
            if evl == width:
                var value_block = value_ptr.unsafe_load[width=width](i)
                var label_block = label_ptr.unsafe_load[width=width](i)
                comptime for lane in range(width):
                    Self._update_lane(
                        destination_ptr,
                        best_ptr,
                        seen_ptr,
                        label_block[lane],
                        value_block[lane],
                        i + lane,
                    )
            else:
                comptime for lane in range(width):
                    if lane < evl:
                        Self._update_lane(
                            destination_ptr,
                            best_ptr,
                            seen_ptr,
                            label_ptr[unsafe_offset=i + lane],
                            value_ptr[unsafe_offset=i + lane],
                            i + lane,
                        )

        vectorize[width](len(values), step)
