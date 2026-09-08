"""NaN-skipping mean with mergeable sum/count state.

Match numbagg's float64 running sum even for float32 inputs: accumulating
in float32 can overflow on finite inputs whose mean is representable.
Only accumulator values widen; inputs stay zero-copy, results keep input dtype.
Integer/float16 promotion belongs to the Python facade.

The shared scanners reduce into partial states. Only finalize divides,
after the axis driver merges all runs, including empty/all-NaN runs.
"""

from std.collections import Span
from std.math import isnan

from mojagg.core.numeric import nan_or_zero
from mojagg.core.reduce1d import (
    NaNReduction1D,
    ReductionWidth,
    scan_contig,
    scan_strided,
)


@fieldwise_init
struct MeanState[width: Int](Copyable):
    var total: SIMD[DType.float64, Self.width]
    var count: SIMD[DType.int64, Self.width]


struct NanMean[
    dtype: DType,
](NaNReduction1D):
    """Chain defaults measured with benchmarks/mean_widths.mojo.

    Float32 widening plus int64 counts makes extra chains expensive; one
    wins the longer scans. Float64's four chains improve 1K+ scans, at a
    small absolute cost on tiny runs. No size-dependent dispatch threshold.
    """

    comptime chains = 1 if Self.dtype == DType.float32 else 4
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime State = MeanState[1]
    comptime block_lanes = ReductionWidth[Self.dtype, Self.chains].block_lanes
    comptime Acc = MeanState[Self.block_lanes]

    # Mojo 1.0 compiler limitation: default methods in sub-traits (NaNReduction1D)
    # do not fulfill parent trait (Reduction1D) requirements on generic structs.
    @staticmethod
    def contig(data: Span[Scalar[Self.value_dtype], _]) -> Self.State:
        return scan_contig[Self](data)

    @staticmethod
    def strided(
        ptr: Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin],
        n: Int,
        stride: Int,
    ) -> Self.State:
        return scan_strided[Self](ptr, n, stride)

    @staticmethod
    def identity() -> Self.State:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "nanmean requires float32 or float64"
        return Self.State(Float64(0), Int64(0))

    @staticmethod
    def acc_init() -> Self.Acc:
        return Self.Acc(
            SIMD[DType.float64, Self.block_lanes](0),
            SIMD[DType.int64, Self.block_lanes](0),
        )

    @staticmethod
    def step_scalar(state: Self.State, value: Scalar[Self.dtype]) -> Self.State:
        if not isnan(value):
            return Self.State(state.total + Float64(value), state.count + 1)
        return state.copy()

    @staticmethod
    def step_simd(
        mut acc: Self.Acc, values: SIMD[Self.dtype, Self.block_lanes]
    ):
        var missing = isnan(values)
        acc.total += missing.select(
            SIMD[Self.dtype, Self.block_lanes](0), values
        ).cast[DType.float64]()
        acc.count += missing.select(
            SIMD[DType.int64, Self.block_lanes](0),
            SIMD[DType.int64, Self.block_lanes](1),
        )

    @staticmethod
    def step_tail[lane: Int](mut acc: Self.Acc, value: Scalar[Self.dtype]):
        var state = Self.step_scalar(
            Self.State(acc.total[lane], acc.count[lane]), value
        )
        acc.total[lane] = state.total
        acc.count[lane] = state.count

    @staticmethod
    def collapse(acc: Self.Acc) -> Self.State:
        return Self.State(acc.total.reduce_add(), acc.count.reduce_add())

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        return Self.State(acc.total + partial.total, acc.count + partial.count)

    def __init__(out self):
        pass

    def finalize(self, state: Self.State) -> Scalar[Self.dtype]:
        if state.count == 0:
            return nan_or_zero[Self.dtype]()
        return (state.total / Float64(state.count)).cast[Self.dtype]()
