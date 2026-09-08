"""Mergeable NaN-aware variance state.

Each contiguous or strided partial uses the numerically stable two-pass
algorithm used by numbagg: compute the mean first, then the sum of squared
deviations. The driver merges partials with Chan's formula so tuple-axis
reductions remain stable without a second Python-visible pass.
"""

from std.collections import Span
from std.math import isnan, sqrt

from mojagg.core.numeric import nan_or_zero
from mojagg.core.reduce1d import Reduction1D


@fieldwise_init
struct VarianceState(Copyable):
    var mean: Float64
    var m2: Float64
    var count: Int64


struct NanVar[
    dtype: DType,
    take_sqrt: Bool = False,
](Reduction1D):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime State = VarianceState

    var ddof: Int

    def __init__(out self, ddof: Int = 1):
        self.ddof = ddof

    @staticmethod
    def identity() -> Self.State:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        ), "nanvar/nanstd require floating-point input"
        return Self.State(0.0, 0.0, 0)

    @staticmethod
    def _from_contig(data: Span[Scalar[Self.dtype], _]) -> Self.State:
        var total = Float64(0)
        var count = Int64(0)
        for i in range(len(data)):
            var value = data[i]
            if not isnan(value):
                total += Float64(value)
                count += 1
        if count == 0:
            return Self.identity()

        var mean = total / Float64(count)
        var m2 = Float64(0)
        for i in range(len(data)):
            var value = data[i]
            if not isnan(value):
                var delta = Float64(value) - mean
                m2 += delta * delta
        return Self.State(mean, m2, count)

    @staticmethod
    def _from_strided(
        ptr: Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin],
        n: Int,
        stride: Int,
    ) -> Self.State:
        var total = Float64(0)
        var count = Int64(0)
        var off = 0
        for _ in range(n):
            var value = ptr[unsafe_offset=off]
            if not isnan(value):
                total += Float64(value)
                count += 1
            off += stride
        if count == 0:
            return Self.identity()

        var mean = total / Float64(count)
        var m2 = Float64(0)
        off = 0
        for _ in range(n):
            var value = ptr[unsafe_offset=off]
            if not isnan(value):
                var delta = Float64(value) - mean
                m2 += delta * delta
            off += stride
        return Self.State(mean, m2, count)

    @staticmethod
    def contig(data: Span[Scalar[Self.dtype], _]) -> Self.State:
        return Self._from_contig(data)

    @staticmethod
    def strided(
        ptr: Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin],
        n: Int,
        stride: Int,
    ) -> Self.State:
        return Self._from_strided(ptr, n, stride)

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        if acc.count == 0:
            return partial.copy()
        if partial.count == 0:
            return acc.copy()

        var total_count = acc.count + partial.count
        var delta = partial.mean - acc.mean
        var weight = Float64(partial.count) / Float64(total_count)
        var correction = (
            delta
            * delta
            * Float64(acc.count)
            * Float64(partial.count)
            / Float64(total_count)
        )
        return Self.State(
            acc.mean + delta * weight,
            acc.m2 + partial.m2 + correction,
            total_count,
        )

    def finalize(self, state: Self.State) -> Scalar[Self.dtype]:
        if state.count <= Int64(self.ddof):
            return nan_or_zero[Self.dtype]()
        var variance = state.m2 / Float64(state.count - Int64(self.ddof))
        if variance < 0:
            variance = 0
        comptime if Self.take_sqrt:
            return sqrt(variance).cast[Self.dtype]()
        else:
            return variance.cast[Self.dtype]()
