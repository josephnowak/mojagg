"""NaN-skipping count: element counting over shared contiguous/strided scanners.

Parity semantics (numbagg.nancount / count): count non-NaN elements over the
reduced slice. Output is ALWAYS Int64. Integer dtypes have no NaN, so every
element is non-NaN (returns slice length in O(1) via LENGTH_ONLY).

Floating point uses SIMD vector accumulators with masked NaN handling.
"""

from std.math import isnan

from mojagg.core.reduce1d import NaNReduction1D, ReductionWidth


struct NanCount[
    dtype: DType,
    chains: Int = 4 if dtype == DType.float64 else 1,
](NaNReduction1D):
    comptime value_dtype = Self.dtype
    comptime out_dtype = DType.int64
    comptime State = Scalar[DType.int64]
    comptime Out = Self.State
    comptime block_lanes = ReductionWidth[Self.dtype, Self.chains].block_lanes
    comptime Acc = SIMD[DType.int64, Self.block_lanes]
    comptime SHORT_CIRCUIT = False

    def __init__(out self):
        pass

    @staticmethod
    def identity() -> Self.State:
        return Self.State(0)

    @staticmethod
    def acc_init() -> Self.Acc:
        return Self.Acc(0)

    @staticmethod
    def step_scalar(state: Self.State, value: Scalar[Self.dtype]) -> Self.State:
        comptime if Self.dtype.is_floating_point():
            if isnan(value):
                return state
        return state + 1

    @staticmethod
    def step_simd(
        mut acc: Self.Acc, values: SIMD[Self.dtype, Self.block_lanes]
    ):
        comptime if Self.dtype.is_floating_point():
            var missing = isnan(values)
            acc += missing.select(
                SIMD[DType.int64, Self.block_lanes](0),
                SIMD[DType.int64, Self.block_lanes](1),
            )
        else:
            acc += SIMD[DType.int64, Self.block_lanes](1)

    @staticmethod
    def step_tail[lane: Int](mut acc: Self.Acc, value: Scalar[Self.dtype]):
        acc[lane] = Self.step_scalar(acc[lane], value)

    @staticmethod
    def collapse(acc: Self.Acc) -> Self.State:
        return acc.reduce_add()

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        return acc + partial

    def finalize(self, state: Self.State) -> Self.Out:
        return state
