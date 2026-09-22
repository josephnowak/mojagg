"""Grouped NaN-aware boolean reductions."""

from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.vectorize import vectorize_no_evl
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
)
from mojagg.groupby.group_kernel import GroupKernel


@fieldwise_init
struct GroupNanAnyAll[
    value_t: DType,
    label_t: DType,
    is_all: Bool,
](GroupKernel, ImplicitlyCopyable):
    """Update group truth values while ignoring NaNs."""

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    @always_inline
    @staticmethod
    def _update_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        var label = Int(label_value)
        var truth: Bool
        comptime if Self.value_t.is_floating_point():
            var value_block = SIMD[Self.value_t, 1](value)
            var zero_block = SIMD[Self.value_t, 1](0)
            var is_nan = isnan(value_block)[0]
            var is_nonzero = value_block != zero_block
            truth = (
                is_nan
                or is_nonzero if Self.is_all else not is_nan
                and is_nonzero
            )
        else:
            truth = value != Scalar[Self.value_t](0)
        # `all` clears the group on a falsy lane, `any` sets it on a truthy one.
        var truth_block = SIMD[DType.bool, 1](truth)
        var destination_block = SIMD[Self.value_t, 1](
            destination[unsafe_offset=label]
        )
        comptime if Self.is_all:
            destination[unsafe_offset=label] = truth_block.select(
                destination_block, SIMD[Self.value_t, 1](0)
            )[0]
        else:
            destination[unsafe_offset=label] = truth_block.select(
                SIMD[Self.value_t, 1](1), destination_block
            )[0]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()

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
                Self._update_lane(
                    destination_ptr,
                    label,
                    value_block[lane],
                )

        vectorize_no_evl[width, unroll_factor=8](len(values), step)
