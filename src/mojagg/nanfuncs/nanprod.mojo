"""NaN-skipping product using the shared reduction scanners.

NaNs contribute the multiplicative identity. Integer specializations erase all
NaN handling at compile time; empty and all-NaN slices therefore return one.
"""

from std.math import isnan

from mojagg.core.reduce1d import NaNReduction1D, ReductionWidth


struct NanProd[dtype: DType](NaNReduction1D):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime State = Scalar[Self.dtype]
    comptime block_lanes = ReductionWidth[Self.dtype].block_lanes
    comptime Acc = SIMD[Self.dtype, Self.block_lanes]

    @staticmethod
    def identity() -> Self.State:
        return Self.State(1)

    @staticmethod
    def acc_init() -> Self.Acc:
        return Self.Acc(1)

    @staticmethod
    def step_scalar(state: Self.State, value: Scalar[Self.dtype]) -> Self.State:
        comptime if Self.dtype.is_floating_point():
            if isnan(value):
                return state
        return state * value

    @staticmethod
    def step_simd(
        mut acc: Self.Acc,
        values: SIMD[Self.dtype, Self.block_lanes],
    ):
        comptime if Self.dtype.is_floating_point():
            acc *= isnan(values).select(Self.Acc(1), values)
        else:
            acc *= values

    @staticmethod
    def step_tail[lane: Int](mut acc: Self.Acc, value: Scalar[Self.dtype]):
        acc[lane] = Self.step_scalar(acc[lane], value)

    @staticmethod
    def collapse(acc: Self.Acc) -> Self.State:
        return acc.reduce_mul()

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        return acc * partial

    def __init__(out self):
        pass

    def finalize(self, state: Self.State) -> Scalar[Self.out_dtype]:
        return state
