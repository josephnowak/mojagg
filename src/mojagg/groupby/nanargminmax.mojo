"""Grouped NaN-aware argmin and argmax kernels."""

from std.collections import Span
from std.math import isnan

from mojagg.core.numeric import (
    load_block_or_identity,
    neg_inf_or_min,
    pos_inf_or_max,
)
from mojagg.core.preallocated import Preallocated
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
)
from mojagg.groupby.group_kernel import GroupKernel


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
    ]

    var preallocated: Preallocated[Self.value_t]

    def __init__(out self):
        self.preallocated = Preallocated[Self.value_t]()

    def __init__(out self, *, copy: Self):
        self.preallocated = Preallocated[Self.value_t]()

    @always_inline
    @staticmethod
    def _identity() -> Scalar[Self.value_t]:
        comptime if Self.value_t == DType.bool:
            return Scalar[Self.value_t](False if Self.is_max else True)
        else:
            return neg_inf_or_min[
                Self.value_t
            ]() if Self.is_max else pos_inf_or_max[Self.value_t]()

    @always_inline
    def get_best_ptr(
        mut self,
        destination: Span[mut=True, Scalar[Self.value_t], _],
    ) -> Pointer[mut=True, Scalar[Self.value_t], MutUntrackedOrigin]:
        return self.preallocated.get_ptr(len(destination), Self._identity())

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
        var current = best_values[unsafe_offset=label]
        var pos = destination[unsafe_offset=label]

        var should_update: SIMD[DType.bool, 1]
        comptime if Self.is_max:
            should_update = value >= current
        else:
            should_update = value <= current

        destination[unsafe_offset=label] = should_update.select(
            Scalar[Self.value_t](flat_index), pos
        )[0]
        best_values[unsafe_offset=label] = should_update.select(value, current)[
            0
        ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var value_input, label_input, output = tensors
        var values = value_input.read_span()
        var labels = label_input.read_span()
        var destination = output.write_span()
        var best_ptr = self.get_best_ptr(destination)

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
            imm best_ptr,
        }:
            var value_block = load_block_or_identity[Self.value_t, width](
                value_ptr, i, evl, Scalar[Self.value_t](0)
            )
            var label_block = load_block_or_identity[Self.label_t, width](
                label_ptr, i, evl, Scalar[Self.label_t](-1)
            )
            comptime for lane in range(width - 1, -1, -1):
                var label = label_block[lane]
                if label < 0:
                    continue
                Self._update_lane(
                    destination_ptr,
                    best_ptr,
                    label,
                    value_block[lane],
                    i + lane,
                )

        var n = len(values)
        var tail = n % width
        var tail_start = n - tail

        if tail > 0:
            step[width](tail_start, tail)

        for i in range(tail_start - width, -1, -width):
            step[width](i, width)
