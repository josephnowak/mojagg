"""Any-NaN predicate: SIMD block checks with slice-wide early termination.

The shared scanner stops before loading the next vector after a NaN.
The axis driver stops the MULTI odometer too, but still computes every
independent output slice. Integer results depend only on slice length:
empty is False, nonempty is False (always False, integers cannot be NaN).
"""

from std.math import isnan

from mojagg.core.reduce1d import NaNReduction1D, ReductionWidth


struct AnyNan[dtype: DType](NaNReduction1D):
    comptime value_dtype = Self.dtype
    comptime out_dtype = DType.bool
    comptime State = Scalar[DType.bool]
    comptime Out = Self.State
    comptime Acc = Self.State
    comptime block_lanes = ReductionWidth[Self.dtype].block_lanes
    comptime SHORT_CIRCUIT = Self.dtype.is_floating_point()

    def __init__(out self):
        pass

    @staticmethod
    def identity() -> Self.State:
        return Self.State(False)

    @staticmethod
    def acc_init() -> Self.Acc:
        return Self.identity()

    @staticmethod
    def step_scalar(state: Self.State, value: Scalar[Self.dtype]) -> Self.State:
        comptime if Self.dtype.is_floating_point():
            return state | isnan(value)
        else:
            return Self.State(False)

    @staticmethod
    def step_simd(
        mut acc: Self.Acc, values: SIMD[Self.dtype, Self.block_lanes]
    ):
        comptime if Self.dtype.is_floating_point():
            acc |= isnan(values).reduce_or()
        else:
            acc = Self.State(False)

    @staticmethod
    def step_tail[lane: Int](mut acc: Self.Acc, value: Scalar[Self.dtype]):
        acc = Self.step_scalar(acc, value)

    @staticmethod
    def collapse(acc: Self.Acc) -> Self.State:
        return acc

    @staticmethod
    def is_terminal(state: Self.State) -> Bool:
        return Bool(state)

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        return acc | partial

    def finalize(self, state: Self.State) -> Self.Out:
        return state
