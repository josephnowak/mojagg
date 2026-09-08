"""NaN-aware minimum reduction."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan, max, min
from std.sys.info import simd_width_of

from mojagg.core.numeric import (
    nan_or_zero,
    neg_inf_or_min,
    pos_inf_or_max,
)
from mojagg.core.reduce1d import Reduction1D, scan_contig, scan_strided
from mojagg.nanfuncs.allnan import AllNan


struct NanExtrema[
    dtype: DType,
    is_min: Bool,
    result_dtype: DType = dtype,
](Copyable, Reduction1D):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.result_dtype
    comptime State = Scalar[Self.dtype]

    comptime inf_val = (
        pos_inf_or_max[Self.dtype]() if Self.is_min else neg_inf_or_min[
            Self.dtype
        ]()
    )

    def __init__(out self):
        pass

    @always_inline
    @staticmethod
    def _op[
        W: Int = 1
    ](a: SIMD[Self.dtype, W], b: SIMD[Self.dtype, W]) -> SIMD[Self.dtype, W]:
        comptime if Self.is_min:
            return min(a, b)
        else:
            return max(a, b)

    @always_inline
    @staticmethod
    def _reduce(
        v: SIMD[Self.dtype, simd_width_of[Self.dtype]()]
    ) -> Scalar[Self.dtype]:
        comptime if Self.is_min:
            return v.reduce_min()
        else:
            return v.reduce_max()

    @always_inline
    @staticmethod
    def _is_valid(res: Scalar[Self.dtype]) -> Bool:
        comptime if Self.is_min:
            return res < Self.inf_val
        else:
            return res > Self.inf_val

    @staticmethod
    def identity() -> Self.State:
        return nan_or_zero[Self.dtype]()

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        comptime if Self.dtype.is_floating_point():
            if isnan(acc):
                return partial
            if isnan(partial):
                return acc
        return Self._op(acc, partial)

    def finalize(self, state: Self.State) -> Scalar[Self.out_dtype]:
        return Scalar[Self.out_dtype](state)

    @staticmethod
    def contig(data: Span[Scalar[Self.value_dtype], _]) -> Self.State:
        var n = len(data)
        if n == 0:
            return nan_or_zero[Self.value_dtype]()

        var ptr = data.unsafe_ptr()
        comptime W = simd_width_of[Self.value_dtype]()
        var acc = SIMD[Self.value_dtype, W](Self.inf_val)
        var inf_vec = SIMD[Self.value_dtype, W](Self.inf_val)

        def step[width: Int](i: Int, evl: Int) {imm ptr, mut acc, imm inf_vec}:
            if evl == W:
                var v = ptr.unsafe_load[width=W](i)
                comptime if Self.value_dtype.is_floating_point():
                    acc = Self._op(acc, isnan(v).select(inf_vec, v))
                else:
                    acc = Self._op(acc, v)
            else:
                comptime for k in range(W):
                    if k < evl:
                        var val = ptr[unsafe_offset=i + k]
                        comptime if Self.value_dtype.is_floating_point():
                            if not isnan(val):
                                acc[k] = Self._op(acc[k], val)
                        else:
                            acc[k] = Self._op(acc[k], val)

        vectorize[W](n, step)
        var res = Self._reduce(acc)

        comptime if Self.value_dtype.is_floating_point():
            if Self._is_valid(res):
                return res
            if Bool(scan_contig[AllNan[Self.value_dtype]](data)):
                return nan_or_zero[Self.value_dtype]()
            return Self.inf_val
        else:
            return res

    @staticmethod
    def strided(
        ptr: Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin],
        n: Int,
        stride: Int,
    ) -> Self.State:
        if n == 0:
            return nan_or_zero[Self.value_dtype]()

        var res = Self.inf_val
        var off = 0
        for _ in range(n):
            var val = ptr[unsafe_offset=off]
            comptime if Self.value_dtype.is_floating_point():
                if not isnan(val):
                    res = Self._op(res, val)
            else:
                res = Self._op(res, val)
            off += stride

        comptime if Self.value_dtype.is_floating_point():
            if Self._is_valid(res):
                return res
            if Bool(scan_strided[AllNan[Self.value_dtype]](ptr, n, stride)):
                return nan_or_zero[Self.value_dtype]()
            return Self.inf_val
        else:
            return res


comptime NanMin[
    dtype: DType,
    result_dtype: DType = dtype,
] = NanExtrema[dtype, True, result_dtype]
