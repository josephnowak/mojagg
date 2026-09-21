"""Grouped NaN-aware first and last kernels."""

from std.algorithm import vectorize
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import load_block_or_identity
from mojagg.core.preallocated import Preallocated
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
)
from mojagg.groupby.group_kernel import GroupKernel


struct GroupNanFirst[
    value_t: DType,
    label_t: DType,
](GroupKernel, ImplicitlyCopyable):
    """Write the first valid value observed by each group."""

    comptime Signature = Tuple[
        GUTensor[Self.value_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.label_t, False, CoreSpec[Dim[0]]],
        GUTensor[Self.value_t, True, CoreSpec[Dim[1]]],
    ]

    var seen: Preallocated[DType.bool]

    def __init__(out self):
        self.seen = Preallocated[DType.bool]()

    def __init__(out self, *, copy: Self):
        self.seen = Preallocated[DType.bool](copy=copy.seen)

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input = tensors[0]
        var label_input = tensors[1]
        var output = tensors[2]
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()

        comptime width = 2
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()
        var seen_ptr = self.seen.get_ptr(len(destination), False)

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
            imm seen_ptr,
        }:
            var value_block = load_block_or_identity[Self.value_t, width](
                value_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var label_block = load_block_or_identity[Self.label_t, width](
                label_ptr, i, evl, Scalar[Self.label_t](-1)
            )
            comptime for lane in range(width):
                if lane < evl:
                    var label = Int(label_block[lane])
                    if label < 0:
                        continue
                    comptime if Self.value_t.is_floating_point():
                        if isnan(value_block[lane]):
                            continue
                    if seen_ptr[unsafe_offset=label]:
                        continue
                    destination_ptr[unsafe_offset=label] = value_block[lane]
                    seen_ptr[unsafe_offset=label] = True

        vectorize[width, unroll_factor=8](len(values), step)


@fieldwise_init
struct GroupNanLast[
    value_t: DType,
    label_t: DType,
](GroupKernel, ImplicitlyCopyable):
    """Write the last valid value observed by each group."""

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
        comptime if Self.value_t.is_floating_point():
            if isnan(value):
                return
        # Cores are traversed in order, so the final store wins; empty groups
        # keep the initialized identity.
        destination[unsafe_offset=label] = value

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
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
        }:
            var value_block = load_block_or_identity[Self.value_t, width](
                value_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var label_block = load_block_or_identity[Self.label_t, width](
                label_ptr, i, evl, Scalar[Self.label_t](-1)
            )
            comptime for lane in range(width):
                var label = label_block[lane]
                if label < 0:
                    continue
                Self._update_lane(
                    destination_ptr,
                    label,
                    value_block[lane],
                )

        vectorize[width, unroll_factor=8](len(values), step)
