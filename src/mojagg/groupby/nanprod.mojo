"""Grouped NaN-aware product over aligned values and labels."""

from std.algorithm import vectorize
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


@fieldwise_init
struct GroupNanProd[
    value_t: DType,
    label_t: DType,
](GUFuncOperation, ImplicitlyCopyable):
    """Multiply valid values into dense groups, using one as the identity."""

    comptime Tensors = Tuple[
        TensorArg[Self.value_t, False],
        TensorArg[Self.label_t, False],
        TensorArg[Self.value_t, True],
    ]

    @always_inline
    @staticmethod
    def _multiply_lane(
        destination: Pointer[mut=True, Scalar[Self.value_t], _],
        label_value: Scalar[Self.label_t],
        value: Scalar[Self.value_t],
    ):
        var label = Int(label_value)
        if label < 0:
            return
        comptime if Self.value_t.is_floating_point():
            if not isnan(value):
                destination[unsafe_offset=label] *= value
        else:
            destination[unsafe_offset=label] *= value

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var values_arg = tensors[0].copy()
        var labels_arg = tensors[1].copy()
        var output_arg = tensors[2].copy()
        var values = values_arg.read_span()
        var labels = labels_arg.read_span()
        var destination = output_arg.write_span()
        var destination_ptr = destination.unsafe_ptr()

        comptime width = simd_width_of[Self.value_t]() * 8
        var value_ptr = values.unsafe_ptr()
        var label_ptr = labels.unsafe_ptr()

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
                    Self._multiply_lane(
                        destination_ptr,
                        label_block[lane],
                        value_block[lane],
                    )
            else:
                comptime for lane in range(width):
                    if lane < evl:
                        Self._multiply_lane(
                            destination_ptr,
                            label_ptr[unsafe_offset=i + lane],
                            value_ptr[unsafe_offset=i + lane],
                        )

        vectorize[width](len(values), step)
