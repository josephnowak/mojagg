"""Grouped NaN-aware minimum and maximum kernels."""

from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import (
    neg_inf_or_min,
    pos_inf_or_max,
    nan_or_zero,
)
from mojagg.core.vectorize import vectorize_no_evl
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
)
from mojagg.groupby.group_kernel import GroupKernel


@fieldwise_init
struct GroupNanMinMax[
    value_t: DType,
    label_t: DType,
    is_max: Bool,
](GroupKernel, ImplicitlyCopyable):
    """Update preinitialized group extrema with valid values."""

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
        var current = destination[unsafe_offset=label]
        var should_update: SIMD[DType.bool, 1]
        comptime if Self.is_max:
            should_update = value >= current
        else:
            should_update = value <= current

        should_update |= isnan(current)
        destination[unsafe_offset=label] = should_update.select(value, current)

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
        comptime identity = (
            neg_inf_or_min[Self.value_t]() if Self.is_max else pos_inf_or_max[
                Self.value_t
            ]()
        )
        comptime identity_block = SIMD[Self.value_t, width](identity)
        comptime missing_label = Scalar[Self.label_t](-1)
        comptime empty_value = nan_or_zero[Self.value_t]()

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
