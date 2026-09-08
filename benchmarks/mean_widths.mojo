"""Compare mean's 1/2/4/8 native-vector accumulator chains.

Run: mojo run -O3 -I src benchmarks/mean_widths.mojo
Each row reports dtype, chains, length, trial, ns/call and a consumed result.
The input is mutated before every no-inline call to prevent hoisting.
"""

from std.collections import Span
from std.math import abs, isnan
from std.python import Python
from std.testing import assert_equal, assert_true
from std.time import perf_counter_ns

from mojagg.core.numeric import nan_or_zero
from mojagg.core.reduce1d import (
    NaNReduction1D,
    ReductionWidth,
    scan_contig,
    scan_strided,
)
from mojagg.nanfuncs.nanmean import MeanState


struct BenchNanMean[
    dtype: DType,
    chains: Int,
](NaNReduction1D):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime State = MeanState[1]
    comptime block_lanes = ReductionWidth[Self.dtype, Self.chains].block_lanes
    comptime Acc = MeanState[Self.block_lanes]

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


@no_inline
def mean[
    dtype: DType, chains: Int
](ptr: Pointer[mut=True, Scalar[dtype], MutAnyOrigin], n: Int) -> Float64:
    var state = scan_contig[BenchNanMean[dtype, chains]](
        Span[Scalar[dtype], MutAnyOrigin](unsafe_ptr=ptr, length=n)
    )
    return Float64(BenchNanMean[dtype, chains]().finalize(state))


def bench[
    dtype: DType, chains: Int
](ptr: Pointer[mut=True, Scalar[dtype], MutAnyOrigin], n: Int, trial: Int,):
    var reps = max(16, 100_000_000 // n)
    var sink = Float64(0)
    var start = perf_counter_ns()
    for r in range(reps):
        ptr[unsafe_offset=0] = Scalar[dtype](r & 7)
        sink += mean[dtype, chains](ptr, n)
    var elapsed = Float64(perf_counter_ns() - start) / Float64(reps)
    print(dtype, chains, n, trial, elapsed, sink)


def sweep[dtype: DType](n: Int) raises:
    var np = Python.import_module("numpy")
    var a = (np.arange(n) % 13).astype(
        "float32" if dtype == DType.float32 else "float64"
    )
    a[np.arange(n) % 7 == 0] = np.nan
    var ptr = Pointer[mut=True, Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(py=a.ctypes.data)
    )
    var expected = Float64(py=np.nanmean(a, dtype=np.float64))
    comptime for i in range(4):
        comptime chains = 1 << i
        assert_true(abs(mean[dtype, chains](ptr, n) - expected) < 1e-6)
        var state = scan_contig[BenchNanMean[dtype, chains]](
            Span[Scalar[dtype], MutAnyOrigin](unsafe_ptr=ptr, length=n)
        )
        assert_equal(
            state.count, Int64(Int(py=np.count_nonzero(np.isfinite(a))))
        )
    for trial in range(7):
        if trial % 2 == 0:
            bench[dtype, 1](ptr, n, trial)
            bench[dtype, 2](ptr, n, trial)
            bench[dtype, 4](ptr, n, trial)
            bench[dtype, 8](ptr, n, trial)
        else:
            bench[dtype, 8](ptr, n, trial)
            bench[dtype, 4](ptr, n, trial)
            bench[dtype, 2](ptr, n, trial)
            bench[dtype, 1](ptr, n, trial)


def main() raises:
    for n in [17, 1024, 100_003, 1_000_003]:
        sweep[DType.float32](n)
        sweep[DType.float64](n)
