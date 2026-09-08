"""Forward and backward fill kernels."""

from std.algorithm import vectorize
from std.collections import InlineArray
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.ndview import DimArray
from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.gufunc import GUFuncKernel, GUFuncPlan


struct FillKernel[dtype: DType, backward: Bool = False](Copyable, GUFuncKernel):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime NUM_INPUTS = 1

    var limit: Int

    def __init__(out self, limit: Int = -1):
        self.limit = limit

    @always_inline
    @staticmethod
    def _update_state(
        val: Scalar[Self.value_dtype],
        actual_limit: Int,
        mut current: Scalar[Self.value_dtype],
        mut lives_remaining: Int,
    ) -> Scalar[Self.value_dtype]:
        if isnan(val):
            if lives_remaining <= 0:
                current = nan_or_zero[Self.value_dtype]()
            lives_remaining -= 1
        else:
            lives_remaining = actual_limit
            current = val
        return current

    @always_inline
    @staticmethod
    def _chunk_offset[width: Int](chunk_idx: Int, n: Int) -> Int:
        comptime if Self.backward:
            return n - chunk_idx - width
        else:
            return chunk_idx

    @always_inline
    @staticmethod
    def _lane_idx[width: Int](s: Int) -> Int:
        comptime if Self.backward:
            return width - 1 - s
        else:
            return s

    @always_inline
    @staticmethod
    def _stride_step(stride: Int) -> Int:
        comptime if Self.backward:
            return -stride
        else:
            return stride

    @always_inline
    @staticmethod
    def _start_offset(stride: Int, n: Int) -> Int:
        comptime if Self.backward:
            return (n - 1) * stride
        else:
            return 0

    @always_inline
    def apply_contig(
        self,
        in_ptrs: InlineArray[
            Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin],
            Self.NUM_INPUTS,
        ],
        out_ptr: Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin],
        n: Int,
        worker_id: Int = 0,
    ):
        var src = in_ptrs[0]
        var dst = out_ptr
        comptime if Self.value_dtype.is_floating_point():
            comptime W = simd_width_of[Self.value_dtype]()
            var current = nan_or_zero[Self.value_dtype]()
            var actual_limit = self.limit if self.limit >= 0 else n
            var lives_remaining = actual_limit

            def step[
                width: Int
            ](chunk_idx: Int) {
                imm src,
                imm dst,
                imm actual_limit,
                imm n,
                mut current,
                mut lives_remaining,
            }:
                var i = Self._chunk_offset[width](chunk_idx, n)
                var v = src.unsafe_load[width=width](i)
                var out_vec = SIMD[Self.value_dtype, width]()
                comptime for s in range(width):
                    comptime k = Self._lane_idx[width](s)
                    out_vec[k] = Self._update_state(
                        v[k], actual_limit, current, lives_remaining
                    )
                dst.unsafe_store[width=width](i, out_vec)

            vectorize[W](n, step)
        else:
            comptime W = simd_width_of[Self.value_dtype]()

            def step_copy[width: Int](i: Int) {imm src, imm dst}:
                dst.unsafe_store[width=width](
                    i, src.unsafe_load[width=width](i)
                )

            vectorize[W](n, step_copy)

    @always_inline
    def apply_strided(
        self,
        in_ptrs: InlineArray[
            Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin],
            Self.NUM_INPUTS,
        ],
        in_strides: InlineArray[Int, Self.NUM_INPUTS],
        out_ptr: Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin],
        out_stride: Int,
        n: Int,
        worker_id: Int = 0,
    ):
        var src = in_ptrs[0]
        var dst = out_ptr
        var s_step = Self._stride_step(in_strides[0])
        var d_step = Self._stride_step(out_stride)
        var src_off = Self._start_offset(in_strides[0], n)
        var dst_off = Self._start_offset(out_stride, n)
        var actual_limit = self.limit if self.limit >= 0 else n
        var lives_remaining = actual_limit
        var current = nan_or_zero[Self.value_dtype]()

        comptime if Self.value_dtype.is_floating_point():
            for _ in range(n):
                dst[unsafe_offset=dst_off] = Self._update_state(
                    src[unsafe_offset=src_off],
                    actual_limit,
                    current,
                    lives_remaining,
                )
                src_off += s_step
                dst_off += d_step
        else:
            for _ in range(n):
                dst[unsafe_offset=dst_off] = src[unsafe_offset=src_off]
                src_off += s_step
                dst_off += d_step

    def apply_multi(
        self,
        plan: GUFuncPlan[Self.value_dtype, Self.out_dtype, Self.NUM_INPUTS],
        flat_o: Int,
        worker_id: Int = 0,
    ):
        var in_base = plan.in_base(0, flat_o)
        var out_base = plan.out_base(flat_o)
        var ctr = DimArray(fill=0)
        var in_rbase = 0
        var out_rbase = 0
        var last = plan.k - 1
        var in_ptrs = InlineArray[
            Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin],
            Self.NUM_INPUTS,
        ](
            fill=Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin](
                unsafe_from_address=plan.in_addrs[0]
            )
        )
        for block in range(plan.rcount):
            var src = Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin](
                unsafe_from_address=plan.in_addrs[0]
            ).unsafe_offset(in_base + in_rbase)
            var dst = Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin](
                unsafe_from_address=plan.out_addr
            ).unsafe_offset(out_base + out_rbase)
            in_ptrs[0] = src
            if plan.inner_in_strides[0] == 1 and plan.inner_out_stride == 1:
                self.apply_contig(in_ptrs, dst, plan.rs[last], worker_id)
            else:
                self.apply_strided(
                    in_ptrs,
                    plan.inner_in_strides,
                    dst,
                    plan.inner_out_stride,
                    plan.rs[last],
                    worker_id,
                )
            var d = plan.k - 2
            while d >= 0:
                ctr[d] += 1
                in_rbase += plan.in_rt[0][d]
                out_rbase += plan.out_rt[d]
                if ctr[d] < plan.rs[d]:
                    break
                ctr[d] = 0
                in_rbase -= plan.rs[d] * plan.in_rt[0][d]
                out_rbase -= plan.rs[d] * plan.out_rt[d]
                d -= 1

    def empty_result(
        self,
        out_ptr: Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin],
    ):
        pass


comptime FFill[dtype: DType] = FillKernel[dtype, False]
comptime BFill[dtype: DType] = FillKernel[dtype, True]
