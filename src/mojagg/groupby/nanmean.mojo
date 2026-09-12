"""Grouped NaN-aware mean over aligned value and label cores."""

from std.algorithm import vectorize
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


@fieldwise_init
struct GroupNanMean[
    value_t: DType,
    label_t: DType,
](GUFuncOperation, ImplicitlyCopyable):
    """Accumulate sums and counts, then finalize each group mean."""

    comptime Tensors = Tuple[
        TensorArg[Self.value_t, False],
        TensorArg[Self.label_t, False],
        TensorArg[Self.value_t, True],
        TensorArg[DType.int64, True],
    ]

    @always_inline
    @staticmethod
    def _add_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        counts: Pointer[mut=True, Scalar[DType.int64], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        var label = Int(label_value)
        if label < 0:
            return
        comptime if Self.value_t.is_floating_point():
            if isnan(value):
                return
        destination[unsafe_offset=label] += value
        counts[unsafe_offset=label] += 1

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var values_arg = tensors[0].copy()
        var labels_arg = tensors[1].copy()
        var output_arg = tensors[2].copy()
        var counts_arg = tensors[3].copy()
        var values = values_arg.read_span()
        var labels = labels_arg.read_span()
        var destination = output_arg.write_span()
        var counts = counts_arg.write_span()

        comptime width = simd_width_of[Self.value_t]() * 8
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()
        var destination_ptr = destination.unsafe_ptr()
        var count_ptr = counts.unsafe_ptr()

        def step[
            vector_width: Int
        ](i: Int, evl: Int) {
            imm value_ptr,
            imm label_ptr,
            imm destination_ptr,
            imm count_ptr,
        }:
            if evl == width:
                var value_block = value_ptr.unsafe_load[width=width](i)
                var label_block = label_ptr.unsafe_load[width=width](i)
                comptime for lane in range(width):
                    Self._add_lane(
                        destination_ptr,
                        count_ptr,
                        label_block[lane],
                        value_block[lane],
                    )
            else:
                comptime for lane in range(width):
                    if lane < evl:
                        Self._add_lane(
                            destination_ptr,
                            count_ptr,
                            label_ptr[unsafe_offset=i + lane],
                            value_ptr[unsafe_offset=i + lane],
                        )

        vectorize[width](len(values), step)

        def finalize[
            vector_width: Int
        ](i: Int, evl: Int,) {imm destination_ptr, imm count_ptr}:
            comptime for lane in range(width):
                if lane < evl:
                    var count = count_ptr[unsafe_offset=i + lane]
                    if count == 0:
                        destination_ptr[unsafe_offset=i + lane] = nan_or_zero[
                            Self.value_t
                        ]()
                    else:
                        destination_ptr[unsafe_offset=i + lane] /= Scalar[
                            Self.value_t
                        ](count)

        vectorize[width](len(destination), finalize)
