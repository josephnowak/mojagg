"""NaN quantile/median operation for the generic GUFunc driver.

The driver gives this operation a worker-local contiguous input scratch area.
Sorting is allowed to mutate that private area while all outputs are still
written directly to the caller's contiguous output tensor.
"""

from std.builtin.sort import sort
from std.collections import Span
from std.math import isnan, min
from std.memory import alloc, dealloc
from std.memory.alloc import Allocation, Layout

from mojagg.core.numeric import nan_or_zero
from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


struct NanQuantileKernel[dtype: DType](GUFuncOperation, ImplicitlyCopyable):
    """Sort-based quantiles over one prepared core span."""

    comptime Tensors = Tuple[
        TensorArg[Self.dtype, False],
        TensorArg[Self.dtype, True],
    ]

    var q_addr: Int
    var num_q: Int
    var workspace: Allocation[Scalar[Self.dtype]]
    var workspace_capacity: Int

    def __init__(out self, q_addr: Int, num_q: Int):
        self.q_addr = q_addr
        self.num_q = num_q
        # A zero-size allocation gives the copyable operation a valid owned
        # allocation while reserving element storage lazily on first apply.
        self.workspace = alloc(Layout[Scalar[Self.dtype]](count=0))
        self.workspace_capacity = 0

    def __init__(out self, *, copy: Self):
        self.q_addr = copy.q_addr
        self.num_q = copy.num_q
        # Worker copies start with independent operation-private workspace.
        self.workspace = alloc(Layout[Scalar[Self.dtype]](count=0))
        self.workspace_capacity = 0

    def __deinit__(deinit self):
        dealloc(self.workspace^)

    @always_inline
    def ensure_workspace(mut self, count: Int):
        if count > self.workspace_capacity:
            dealloc(self.workspace^)
            self.workspace = alloc(Layout[Scalar[Self.dtype]](count=count))
            self.workspace_capacity = count

    @always_inline
    def workspace_ptr(
        mut self,
    ) -> Pointer[mut=True, Scalar[Self.dtype], MutUntrackedOrigin]:
        return Pointer[
            mut=True,
            Scalar[Self.dtype],
            MutUntrackedOrigin,
        ](unsafe_from_address=Int(self.workspace.unsafe_ptr()))

    @always_inline
    def q_ptr(self) -> Pointer[mut=False, Float64, MutAnyOrigin]:
        return Pointer[mut=False, Float64, MutAnyOrigin](
            unsafe_from_address=self.q_addr
        )

    def _compute_quantiles(
        self,
        buf: Pointer[mut=True, Scalar[Self.dtype], _],
        valid_count: Int,
        out_ptr: Pointer[mut=True, Scalar[Self.dtype], _],
    ):
        if valid_count == 0:
            for m in range(self.num_q):
                out_ptr[unsafe_offset=m] = nan_or_zero[Self.dtype]()
            return

        if self.num_q == 0:
            return

        # NaNs were compacted out by apply, so the standard-library sort can
        # order the complete valid span once and every quantile can reuse it.
        sort(Span(unsafe_ptr=buf, length=valid_count))

        var q_p = self.q_ptr()
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
                if high == low:
                    # Avoid 0 * (inf - inf) for exact endpoint ranks.
                    out_ptr[unsafe_offset=m] = Scalar[Self.dtype](v_low)
                else:
                    var v_high = Float64(buf[unsafe_offset=high])
                    out_ptr[unsafe_offset=m] = Scalar[Self.dtype](
                        v_low + frac * (v_high - v_low)
                    )

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        var source = input.read_span()
        self.ensure_workspace(len(source))
        var scratch = self.workspace_ptr()
        var valid_count = 0
        for i in range(len(source)):
            var value = source.unsafe_ptr()[unsafe_offset=i]
            if not isnan(value):
                scratch[unsafe_offset=valid_count] = value
                valid_count += 1
        self._compute_quantiles(
            scratch,
            valid_count,
            output.write_span().unsafe_ptr(),
        )
