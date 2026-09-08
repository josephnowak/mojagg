"""apply_gufunc — Unified Generalized Universal Function (GUFunc) driver.

Modeled after Numba/NumPy guvectorize conventions:
- Reductions: `(n) -> ()` (scalar output per outer slice)
- Transforms: `(n) -> (n)` (array output per outer slice)
- Groupby / Multi-input: `(n),(n) -> (m)` (supports arbitrary NUM_INPUTS)

Outer dimensions loop across independent slices (serially or parallelized via
`parallelize(worker, NUM_WORKERS)`). For each slice, mixed-radix decomposition
resolves base pointers for each input and the output, then dispatches to
a single unified `apply_slice` function.
"""

from std.collections import InlineArray, Span

from max.algorithm import parallelize

from mojagg.core.ndview import DimArray, NDView
from mojagg.core.reduce1d import Reduction1D

# Scenario identifiers (runtime Int; classified once in `GUFuncPlan.build`).
comptime SCENARIO_MERGED = 0
comptime SCENARIO_STRIDED = 1
comptime SCENARIO_MULTI = 2

# Coarse parallel chunking: one task per worker.
comptime NUM_WORKERS = 16


struct GUFuncPlan[
    dtype: DType,
    out_dtype: DType,
    NUM_INPUTS: Int = 1,
](Copyable):
    """Everything per-slice dispatch needs, computed once per call.

    Supports arbitrary NUM_INPUTS via stack-allocated InlineArray.
    """

    var in_addrs: InlineArray[Int, Self.NUM_INPUTS]
    var out_addr: Int
    var osizes: DimArray
    var in_ostrides: InlineArray[DimArray, Self.NUM_INPUTS]
    var out_ostrides: DimArray
    var ko: Int
    var outer_count: Int
    var rs: DimArray
    var in_rt: InlineArray[DimArray, Self.NUM_INPUTS]
    var out_rt: DimArray
    var k: Int
    var n: Int
    var inner_in_strides: InlineArray[Int, Self.NUM_INPUTS]
    var inner_out_stride: Int
    var rcount: Int
    var scenario: Int

    def __init__(
        out self,
        in_addrs: InlineArray[Int, Self.NUM_INPUTS],
        out_addr: Int,
        osizes: DimArray,
        in_ostrides: InlineArray[DimArray, Self.NUM_INPUTS],
        out_ostrides: DimArray,
        ko: Int,
        outer_count: Int,
        rs: DimArray,
        in_rt: InlineArray[DimArray, Self.NUM_INPUTS],
        out_rt: DimArray,
        k: Int,
        n: Int,
        inner_in_strides: InlineArray[Int, Self.NUM_INPUTS],
        inner_out_stride: Int,
        rcount: Int,
        scenario: Int,
    ):
        self.in_addrs = in_addrs.copy()
        self.out_addr = out_addr
        self.osizes = osizes.copy()
        self.in_ostrides = in_ostrides.copy()
        self.out_ostrides = out_ostrides.copy()
        self.ko = ko
        self.outer_count = outer_count
        self.rs = rs.copy()
        self.in_rt = in_rt.copy()
        self.out_rt = out_rt.copy()
        self.k = k
        self.n = n
        self.inner_in_strides = inner_in_strides.copy()
        self.inner_out_stride = inner_out_stride
        self.rcount = rcount
        self.scenario = scenario

    def in_base(self, inp: Int, flat_o: Int) -> Int:
        if self.ko == 0:
            return 0
        if self.ko == 1:
            return flat_o * self.in_ostrides[inp][0]
        var base = 0
        var rem = flat_o
        for d in range(self.ko - 1, -1, -1):
            var q = rem % self.osizes[d]
            base += q * self.in_ostrides[inp][d]
            rem //= self.osizes[d]
        return base

    def out_base(self, flat_o: Int) -> Int:
        if self.ko == 0:
            return 0
        if self.ko == 1:
            return flat_o * self.out_ostrides[0]
        var base = 0
        var rem = flat_o
        for d in range(self.ko - 1, -1, -1):
            var q = rem % self.osizes[d]
            base += q * self.out_ostrides[d]
            rem //= self.osizes[d]
        return base

    def in_ptr(
        self, inp: Int, flat_o: Int
    ) -> Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin]:
        return Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=self.in_addrs[inp]
        ).unsafe_offset(self.in_base(inp, flat_o))

    def in_ptrs(
        self, flat_o: Int
    ) -> InlineArray[
        Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin],
        Self.NUM_INPUTS,
    ]:
        var ptrs = InlineArray[
            Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin],
            Self.NUM_INPUTS,
        ](
            fill=Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin](
                unsafe_from_address=self.in_addrs[0]
            )
        )
        comptime for i in range(Self.NUM_INPUTS):
            ptrs[i] = self.in_ptr(i, flat_o)
        return ptrs^

    def out_ptr(
        self, flat_o: Int
    ) -> Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin]:
        return Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin](
            unsafe_from_address=self.out_addr
        ).unsafe_offset(self.out_base(flat_o))

    @staticmethod
    def build(
        view: NDView[Self.dtype],
        axes: DimArray,
        k: Int,
        out_addr: Int,
        out_core_size: Int = 1,
    ) -> Self:
        """Build plan for a single-input reduction (n)->() or multi-output (n)->(m).
        """
        var in_reduced = DimArray(fill=0)
        for i in range(k):
            in_reduced[axes[i]] = 1

        var osizes = DimArray(fill=0)
        var ostrides = DimArray(fill=0)
        var ko = 0
        var outer_count = 1
        for d in range(view.ndim):
            if in_reduced[d] == 0:
                osizes[ko] = view.sizes[d]
                ostrides[ko] = view.strides[d]
                outer_count *= view.sizes[d]
                ko += 1

        var rs = DimArray(fill=0)
        var rt = DimArray(fill=0)
        for i in range(k):
            rs[i] = view.sizes[axes[i]]
            rt[i] = view.strides[axes[i]]

        var n = 1
        for i in range(k):
            n *= rs[i]
        var inner_stride = rt[k - 1]

        var merged = rt[k - 1] == 1
        var i = k - 2
        while merged and i >= 0:
            if rt[i] != rt[i + 1] * rs[i + 1]:
                merged = False
            i -= 1

        var scenario = SCENARIO_MULTI
        if merged:
            scenario = SCENARIO_MERGED
        elif k == 1:
            scenario = SCENARIO_STRIDED

        var rcount = 1
        for j in range(k - 1):
            rcount *= rs[j]

        var out_ostrides = DimArray(fill=0)
        var s = out_core_size
        for d in range(ko - 1, -1, -1):
            out_ostrides[d] = s
            s *= osizes[d]

        var in_addrs = InlineArray[Int, Self.NUM_INPUTS](fill=0)
        in_addrs[0] = view.addr
        var in_ostrides = InlineArray[DimArray, Self.NUM_INPUTS](
            fill=DimArray(fill=0)
        )
        in_ostrides[0] = ostrides.copy()
        var in_rt = InlineArray[DimArray, Self.NUM_INPUTS](
            fill=DimArray(fill=0)
        )
        in_rt[0] = rt.copy()
        var inner_in_strides = InlineArray[Int, Self.NUM_INPUTS](fill=0)
        inner_in_strides[0] = inner_stride
        var out_rt = DimArray(fill=0)
        var inner_out_stride = 0

        return Self(
            in_addrs,
            out_addr,
            osizes,
            in_ostrides,
            out_ostrides,
            ko,
            outer_count,
            rs,
            in_rt,
            out_rt,
            k,
            n,
            inner_in_strides,
            inner_out_stride,
            rcount,
            scenario,
        )

    @staticmethod
    def build_transform(
        view: NDView[Self.dtype],
        out_view: NDView[Self.out_dtype],
        axes: DimArray,
        k: Int,
    ) -> Self:
        """Build plan for a single-input transform (n)->(n)."""
        var in_reduced = DimArray(fill=0)
        for i in range(k):
            in_reduced[axes[i]] = 1

        var osizes = DimArray(fill=0)
        var ostrides = DimArray(fill=0)
        var out_ostrides = DimArray(fill=0)
        var ko = 0
        var outer_count = 1
        for d in range(view.ndim):
            if in_reduced[d] == 0:
                osizes[ko] = view.sizes[d]
                ostrides[ko] = view.strides[d]
                out_ostrides[ko] = out_view.strides[d]
                outer_count *= view.sizes[d]
                ko += 1

        var rs = DimArray(fill=0)
        var rt = DimArray(fill=0)
        var out_rt = DimArray(fill=0)
        for i in range(k):
            rs[i] = view.sizes[axes[i]]
            rt[i] = view.strides[axes[i]]
            out_rt[i] = out_view.strides[axes[i]]

        var n = 1
        for i in range(k):
            n *= rs[i]
        var inner_stride = rt[k - 1]
        var out_inner_stride = out_view.strides[axes[k - 1]]

        var merged = rt[k - 1] == 1 and out_inner_stride == 1
        var i = k - 2
        while merged and i >= 0:
            if rt[i] != rt[i + 1] * rs[i + 1]:
                merged = False
            if out_rt[i] != out_rt[i + 1] * rs[i + 1]:
                merged = False
            i -= 1

        var scenario = SCENARIO_MULTI
        if merged:
            scenario = SCENARIO_MERGED
        elif k == 1:
            scenario = SCENARIO_STRIDED

        var rcount = 1
        for j in range(k - 1):
            rcount *= rs[j]

        var in_addrs = InlineArray[Int, Self.NUM_INPUTS](fill=0)
        in_addrs[0] = view.addr
        var in_ostrides = InlineArray[DimArray, Self.NUM_INPUTS](
            fill=DimArray(fill=0)
        )
        in_ostrides[0] = ostrides.copy()
        var in_rt = InlineArray[DimArray, Self.NUM_INPUTS](
            fill=DimArray(fill=0)
        )
        in_rt[0] = rt.copy()
        var inner_in_strides = InlineArray[Int, Self.NUM_INPUTS](fill=0)
        inner_in_strides[0] = inner_stride

        return Self(
            in_addrs,
            out_view.addr,
            osizes,
            in_ostrides,
            out_ostrides,
            ko,
            outer_count,
            rs,
            in_rt,
            out_rt,
            k,
            n,
            inner_in_strides,
            out_inner_stride,
            rcount,
            scenario,
        )


trait GUFuncKernel(Copyable & Deinitable):
    comptime value_dtype: DType
    comptime out_dtype: DType
    comptime NUM_INPUTS: Int = 1

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
        ...

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
        ...

    def apply_multi(
        self,
        plan: GUFuncPlan[Self.value_dtype, Self.out_dtype, Self.NUM_INPUTS],
        flat_o: Int,
        worker_id: Int = 0,
    ):
        ...

    def empty_result(
        self,
        out_ptr: Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin],
    ):
        ...


struct ReduceKernel[Op: Reduction1D](Copyable, GUFuncKernel):
    """Adapts a Reduction1D operation into the unified GUFuncKernel interface.
    """

    comptime value_dtype = Self.Op.value_dtype
    comptime out_dtype = Self.Op.out_dtype
    comptime NUM_INPUTS = 1

    var op: Self.Op

    def __init__(out self, op: Self.Op):
        self.op = op.copy()

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
        var state = Self.Op.contig(
            Span[Scalar[Self.value_dtype], MutAnyOrigin](
                unsafe_ptr=in_ptrs[0], length=n
            )
        )
        out_ptr[unsafe_offset=0] = self.op.finalize(state)

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
        var state = Self.Op.strided(in_ptrs[0], n, in_strides[0])
        out_ptr[unsafe_offset=0] = self.op.finalize(state)

    def apply_multi(
        self,
        plan: GUFuncPlan[Self.value_dtype, Self.out_dtype, Self.NUM_INPUTS],
        flat_o: Int,
        worker_id: Int = 0,
    ):
        var base = plan.in_base(0, flat_o)
        var acc = Self.Op.identity()
        var ctr = DimArray(fill=0)
        var rbase = 0
        var last = plan.k - 1
        for block in range(plan.rcount):
            var p = Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin](
                unsafe_from_address=plan.in_addrs[0]
            ).unsafe_offset(base + rbase)
            var partial: Self.Op.State
            if plan.inner_in_strides[0] == 1:
                partial = Self.Op.contig(
                    Span[Scalar[Self.value_dtype], MutAnyOrigin](
                        unsafe_ptr=p, length=plan.rs[last]
                    )
                )
            else:
                partial = Self.Op.strided(
                    p, plan.rs[last], plan.inner_in_strides[0]
                )
            acc = Self.Op.combine_at(acc, partial, block * plan.rs[last])
            comptime if Self.Op.SHORT_CIRCUIT:
                if Self.Op.is_terminal(acc):
                    break
            var d = plan.k - 2
            while d >= 0:
                ctr[d] += 1
                rbase += plan.in_rt[0][d]
                if ctr[d] < plan.rs[d]:
                    break
                ctr[d] = 0
                rbase -= plan.rs[d] * plan.in_rt[0][d]
                d -= 1
        plan.out_ptr(flat_o)[unsafe_offset=0] = self.op.finalize(acc)

    def empty_result(
        self,
        out_ptr: Pointer[mut=True, Scalar[Self.out_dtype], MutAnyOrigin],
    ):
        out_ptr[unsafe_offset=0] = self.op.finalize(Self.Op.identity())


def apply_slice[
    Op: GUFuncKernel
](
    op: Op,
    plan: GUFuncPlan[Op.value_dtype, Op.out_dtype, Op.NUM_INPUTS],
    flat_o: Int,
    worker_id: Int = 0,
):
    """Unified per-slice execution for generalized universal functions."""
    var out_p = plan.out_ptr(flat_o)
    if plan.n == 0:
        op.empty_result(out_p)
        return

    var in_ptrs = plan.in_ptrs(flat_o)

    if plan.scenario == SCENARIO_MERGED:
        op.apply_contig(in_ptrs, out_p, plan.n, worker_id)
    elif plan.scenario == SCENARIO_STRIDED:
        op.apply_strided(
            in_ptrs,
            plan.inner_in_strides,
            out_p,
            plan.inner_out_stride,
            plan.n,
            worker_id,
        )
    else:
        op.apply_multi(plan, flat_o, worker_id)


def apply_gufunc[
    Op: GUFuncKernel
](
    op: Op,
    plan: GUFuncPlan[Op.value_dtype, Op.out_dtype, Op.NUM_INPUTS],
    parallel_threshold: Int,
    workers: Int = 0,
):
    """Unified GUFunc driver: parallelized coarse-chunked execution."""
    var effective_workers = workers if workers > 0 else 16
    if (
        plan.outer_count > 1
        and plan.outer_count * plan.n >= parallel_threshold
        and effective_workers > 1
    ):
        var num_workers = min(effective_workers, plan.outer_count)
        var chunk = (plan.outer_count + num_workers - 1) // num_workers

        def worker(w: Int) {imm op, imm plan, imm chunk}:
            var start = w * chunk
            var end = min(start + chunk, plan.outer_count)
            for o in range(start, end):
                apply_slice(op, plan, o, w)

        parallelize(worker, num_workers)
    else:
        for o in range(plan.outer_count):
            apply_slice(op, plan, o, 0)


def apply_gufunc[
    Op: Reduction1D
](
    op: Op,
    view: NDView[Op.value_dtype],
    axes: DimArray,
    k: Int,
    out_addr: Int,
    parallel_threshold: Int,
    workers: Int = 0,
):
    """Reduce-axis driver: wraps Reduction1D into ReduceKernel and runs."""
    var plan = GUFuncPlan[Op.value_dtype, Op.out_dtype, 1].build(
        view, axes, k, out_addr
    )
    var kernel = ReduceKernel[Op](op)
    apply_gufunc(kernel, plan, parallel_threshold, workers)


def apply_gufunc[
    Op: GUFuncKernel
](
    op: Op,
    view: NDView[Op.value_dtype],
    out_view: NDView[Op.out_dtype],
    axes: DimArray,
    k: Int,
    parallel_threshold: Int,
    workers: Int = 0,
):
    """Transform / multi-input driver: builds plan and runs via GUFuncKernel."""
    var plan = GUFuncPlan[
        Op.value_dtype, Op.out_dtype, Op.NUM_INPUTS
    ].build_transform(view, out_view, axes, k)
    apply_gufunc(op, plan, parallel_threshold, workers)
