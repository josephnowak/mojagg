"""Quantile and median reductions conforming to GUFuncKernel."""

from std.builtin.sort import partition, sort
from std.collections import InlineArray, Span
from std.math import isnan
from std.memory import alloc, dealloc, Layout

from mojagg.core.ndview import DimArray, NDView
from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.gufunc import (
    apply_gufunc,
    GUFuncKernel,
    GUFuncPlan,
)


@always_inline
def partition_kth[
    dtype: DType, origin: MutOrigin
](span: Span[Scalar[dtype], origin], k: Int):
    """Mojo builtin partition with capturing comparator."""

    def cmp(x: Scalar[dtype], y: Scalar[dtype]) capturing -> Bool:
        return x < y

    partition[cmp_fn=cmp](span, k)


def _select_kth[
    dtype: DType
](
    ptr: Pointer[mut=True, Scalar[dtype], _],
    left: Int,
    right: Int,
    k_targets: Pointer[mut=False, Int, _],
    left_t: Int,
    right_t: Int,
):
    if left >= right or left_t > right_t:
        return

    var mid_t = left_t + (right_t - left_t) // 2
    var target = k_targets[unsafe_offset=mid_t]

    var l = left
    var r = right
    while l < r:
        var pivot_idx = l + (r - l) // 2
        var pivot = ptr[unsafe_offset=pivot_idx]
        ptr[unsafe_offset=pivot_idx] = ptr[unsafe_offset=r]
        ptr[unsafe_offset=r] = pivot

        var i = l
        for j in range(l, r):
            if ptr[unsafe_offset=j] < pivot:
                var tmp = ptr[unsafe_offset=i]
                ptr[unsafe_offset=i] = ptr[unsafe_offset=j]
                ptr[unsafe_offset=j] = tmp
                i += 1
        var tmp = ptr[unsafe_offset=i]
        ptr[unsafe_offset=i] = ptr[unsafe_offset=r]
        ptr[unsafe_offset=r] = tmp

        if i == target:
            break
        elif i < target:
            l = i + 1
        else:
            r = i - 1

    if left_t < mid_t:
        _select_kth[dtype](ptr, left, target - 1, k_targets, left_t, mid_t - 1)
    if mid_t < right_t:
        _select_kth[dtype](
            ptr, target + 1, right, k_targets, mid_t + 1, right_t
        )


struct NanQuantileKernel[dtype: DType](Copyable, GUFuncKernel):
    """Unified GUFuncKernel for nanquantile and nanmedian operations."""

    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime NUM_INPUTS = 1

    var q_addr: Int
    var num_q: Int
    var scratch_addr: Int
    var slice_n: Int

    def __init__(
        out self,
        q_addr: Int,
        num_q: Int,
        scratch_addr: Int,
        slice_n: Int,
    ):
        self.q_addr = q_addr
        self.num_q = num_q
        self.scratch_addr = scratch_addr
        self.slice_n = slice_n

    def q_ptr(self) -> Pointer[mut=False, Float64, MutAnyOrigin]:
        return Pointer[mut=False, Float64, MutAnyOrigin](
            unsafe_from_address=self.q_addr
        )

    def scratch_buf(
        self, worker_id: Int
    ) -> Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin]:
        return Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=self.scratch_addr
        ).unsafe_offset(worker_id * self.slice_n)

    def empty_result(
        self,
        out_ptr: Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin],
    ):
        for m in range(self.num_q):
            out_ptr[unsafe_offset=m] = nan_or_zero[Self.dtype]()

    def _compute_quantiles(
        self,
        buf: Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin],
        valid_count: Int,
        out_ptr: Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin],
    ):
        if valid_count == 0:
            for m in range(self.num_q):
                out_ptr[unsafe_offset=m] = nan_or_zero[Self.dtype]()
            return

        var q_p = self.q_ptr()
        if self.num_q == 1:
            var q = q_p[unsafe_offset=0]
            if isnan(q):
                out_ptr[unsafe_offset=0] = nan_or_zero[Self.dtype]()
                return
            var rank = Float64(valid_count - 1) * q
            var low = Int(rank)
            var high = min(low + 1, valid_count - 1)
            partition_kth[Self.dtype](
                Span(unsafe_ptr=buf, length=valid_count), low
            )
            var v_low = Float64(buf[unsafe_offset=low])
            if high == low:
                out_ptr[unsafe_offset=0] = Scalar[Self.dtype](v_low)
            else:
                var v_high_val = buf[unsafe_offset=low + 1]
                for j in range(low + 2, valid_count):
                    if buf[unsafe_offset=j] < v_high_val:
                        v_high_val = buf[unsafe_offset=j]
                var v_high = Float64(v_high_val)
                var frac = rank - Float64(low)
                var res = v_low + frac * (v_high - v_low)
                out_ptr[unsafe_offset=0] = Scalar[Self.dtype](res)
            return

        if self.num_q <= 4 and valid_count > 64:
            var k_targets = InlineArray[Int, 8](fill=0)
            var k_cnt = 0
            for m in range(self.num_q):
                var q = q_p[unsafe_offset=m]
                if not isnan(q):
                    var rank = Float64(valid_count - 1) * q
                    var low = Int(rank)
                    var high = min(low + 1, valid_count - 1)
                    k_targets[k_cnt] = low
                    k_cnt += 1
                    k_targets[k_cnt] = high
                    k_cnt += 1

            for i in range(1, k_cnt):
                var key = k_targets[i]
                var j = i - 1
                while j >= 0 and k_targets[j] > key:
                    k_targets[j + 1] = k_targets[j]
                    j -= 1
                k_targets[j + 1] = key

            var uniq_cnt = 0
            if k_cnt > 0:
                uniq_cnt = 1
                for i in range(1, k_cnt):
                    if k_targets[i] != k_targets[uniq_cnt - 1]:
                        k_targets[uniq_cnt] = k_targets[i]
                        uniq_cnt += 1

            if uniq_cnt > 0:
                _select_kth[Self.dtype](
                    buf,
                    0,
                    valid_count - 1,
                    k_targets.unsafe_ptr(),
                    0,
                    uniq_cnt - 1,
                )
        else:
            sort(Span(unsafe_ptr=buf, length=valid_count))

        for m in range(self.num_q):
            var q = q_p[unsafe_offset=m]
            if isnan(q):
                out_ptr[unsafe_offset=m] = nan_or_zero[Self.dtype]()
            else:
                var rank = Float64(valid_count - 1) * q
                var low = Int(rank)
                var high = min(low + 1, valid_count - 1)
                var frac = rank - Float64(low)
                var v_low = Float64(buf[unsafe_offset=low])
                var v_high = Float64(buf[unsafe_offset=high])
                var res = v_low + frac * (v_high - v_low)
                out_ptr[unsafe_offset=m] = Scalar[Self.dtype](res)

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
        var buf = self.scratch_buf(worker_id)
        var p = in_ptrs[0]
        var valid_count = 0
        for k in range(n):
            var v = p[unsafe_offset=k]
            if not isnan(v):
                buf[unsafe_offset=valid_count] = v
                valid_count += 1
        self._compute_quantiles(buf, valid_count, out_ptr)

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
        var buf = self.scratch_buf(worker_id)
        var p = in_ptrs[0]
        var stride = in_strides[0]
        var valid_count = 0
        for k in range(n):
            var v = p[unsafe_offset=k * stride]
            if not isnan(v):
                buf[unsafe_offset=valid_count] = v
                valid_count += 1
        self._compute_quantiles(buf, valid_count, out_ptr)

    def apply_multi(
        self,
        plan: GUFuncPlan[Self.value_dtype, Self.out_dtype, Self.NUM_INPUTS],
        flat_o: Int,
        worker_id: Int = 0,
    ):
        var buf = self.scratch_buf(worker_id)
        var base = plan.in_base(0, flat_o)
        var ctr = DimArray(fill=0)
        var rbase = 0
        var last = plan.k - 1
        var stride = plan.inner_in_strides[0]
        var valid_count = 0
        for _ in range(plan.rcount):
            var p = Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin](
                unsafe_from_address=plan.in_addrs[0]
            ).unsafe_offset(base + rbase)
            for k in range(plan.rs[last]):
                var v = p[unsafe_offset=k * stride]
                if not isnan(v):
                    buf[unsafe_offset=valid_count] = v
                    valid_count += 1
            var d = plan.k - 2
            while d >= 0:
                ctr[d] += 1
                rbase += plan.in_rt[0][d]
                if ctr[d] < plan.rs[d]:
                    break
                ctr[d] = 0
                rbase -= plan.rs[d] * plan.in_rt[0][d]
                d -= 1
        self._compute_quantiles(buf, valid_count, plan.out_ptr(flat_o))


def nanquantile_driver[
    dtype: DType
](
    view: NDView[dtype],
    axes: DimArray,
    k: Int,
    q_addr: Int,
    num_q: Int,
    out_addr: Int,
    threshold: Int,
    workers: Int,
):
    var plan = GUFuncPlan[dtype, dtype, 1].build(
        view, axes, k, out_addr, out_core_size=num_q
    )
    var effective_workers = workers if workers > 0 else 16
    var num_workers = 1
    if (
        plan.outer_count > 1
        and plan.outer_count * plan.n >= threshold
        and effective_workers > 1
    ):
        num_workers = min(effective_workers, plan.outer_count)

    var scratch_count = max(num_workers * plan.n, 1)
    var scratch_alloc = alloc(Layout[Scalar[dtype]](count=scratch_count))
    var kernel = NanQuantileKernel[dtype](
        q_addr, num_q, Int(scratch_alloc.unsafe_ptr()), plan.n
    )

    apply_gufunc(kernel, plan, threshold, num_workers)

    dealloc(scratch_alloc^)
