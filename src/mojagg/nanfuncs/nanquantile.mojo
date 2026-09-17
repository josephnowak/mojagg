"""NaN quantile/median operation for the guvectorize driver.

The driver gives this operation a worker-local contiguous input scratch area.
Sorting is allowed to mutate that private area while all outputs are still
written directly to the caller's contiguous output tensor.
"""

from std.builtin.sort import partition, sort
from std.collections import InlineArray, Span
from std.math import isinf, isnan, min
from std.memory import alloc, dealloc
from std.memory.alloc import Allocation, Layout
from std.sys import size_of

from mojagg.core.numeric import nan_or_zero
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


comptime QUANTILE_DIM = 1

# Selection beats a full ordering only while few order statistics are needed.
# Measured with benchmarks/quantile_select.mojo (-O3, --mcpu x86-64-v3, f64,
# sort_ns / select_ns, restore cost removed):
#
#   unique indices | 1024   2048   16384  262144  1048576
#   ---------------+---------------------------------------
#   2  (1 q)       | 0.88   1.57   1.76   2.88    2.60
#   4  (2 q)       | 0.89   1.35   1.13   1.69    1.88
#   6  (3 q)       | 0.91   1.20   1.03   1.75    1.63
#   10 (5 q)       | 0.61   0.71   1.07   1.24    1.33
#   18 (9 q)       | 0.56   0.56   0.76   1.08    1.20
#   38 (19 q)      | 0.41   0.49   0.60   0.89    0.92
#
# The tiers below keep selection only where it measured faster; every other
# shape still takes the single sort, which stays optimal for many quantiles.
comptime SELECT_MAX_INDICES = 20
comptime SELECT_MIN_LENGTH_SMALL = 2048
comptime SELECT_MIN_LENGTH_MEDIUM = 16384
comptime SELECT_MIN_LENGTH_LARGE = 262144
# Bisecting recursion over at most SELECT_MAX_INDICES positions never nests
# deeper than ceil(log2(20)) + 1 = 6 frames, so the explicit stack cannot spill.
comptime SELECT_STACK_SLOTS = 16


@always_inline
def partition_kth[
    dtype: DType, origin: MutOrigin
](span: Span[Scalar[dtype], origin], k: Int):
    """Partition a span around one requested order statistic."""

    def cmp(x: Scalar[dtype], y: Scalar[dtype]) capturing -> Bool:
        return x < y

    partition[cmp_fn=cmp](span, k)


@always_inline
def value_pointer[
    dtype: DType
](base: Int, offset: Int) -> Pointer[
    mut=True, Scalar[dtype], MutUntrackedOrigin
]:
    return Pointer[mut=True, Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=base + offset * size_of[Scalar[dtype]]()
    )


@always_inline
def index_pointer(
    base: Int, offset: Int
) -> Pointer[mut=True, Int64, MutUntrackedOrigin]:
    return Pointer[mut=True, Int64, MutUntrackedOrigin](
        unsafe_from_address=base + offset * size_of[Int64]()
    )


@always_inline
def insert_index(index_base: Int, count: Int, candidate: Int) -> Int:
    """Insert one rank into an ascending, deduplicated index list."""
    var position = count
    while position > 0 and Int(index_pointer(index_base, position - 1)[]) > (
        candidate
    ):
        position -= 1
    if (
        position > 0
        and Int(index_pointer(index_base, position - 1)[]) == candidate
    ):
        return count
    for shift in range(count, position, -1):
        index_pointer(index_base, shift)[] = index_pointer(
            index_base, shift - 1
        )[]
    index_pointer(index_base, position)[] = Int64(candidate)
    return count + 1


@always_inline
def endpoint_value[dtype: DType](value: Float64) -> Scalar[dtype]:
    """Result for a rank whose floor and ceil indices coincide.

    numbagg evaluates ``floor + 0 * (ceil - floor)`` unconditionally, which is
    ``0 * (inf - inf) = NaN`` when the endpoint is infinite.  The finite case
    reduces to the endpoint itself, so only the infinite case is special.
    """
    if isinf(value):
        return nan_or_zero[dtype]()
    return Scalar[dtype](value)


@always_inline
def prefers_selection(valid_count: Int, index_count: Int) -> Bool:
    """Choose multi-kth selection over a full sort from benchmark tiers."""
    if index_count == 0 or index_count > SELECT_MAX_INDICES:
        return False
    if index_count <= 8:
        return valid_count >= SELECT_MIN_LENGTH_SMALL
    if index_count <= 12:
        return valid_count >= SELECT_MIN_LENGTH_MEDIUM
    return valid_count >= SELECT_MIN_LENGTH_LARGE


def multi_select[
    dtype: DType
](buf_base: Int, valid_count: Int, index_base: Int, index_count: Int):
    """Place every requested order statistic without ordering the rest.

    This is the Mojo spelling of ``np.partition(arr, kth=unique_indices)``:
    partition around the middle requested index, then recurse into the two
    sub-ranges that still hold requested indices.  Bisecting the index list
    keeps the cost at O(n log index_count) instead of O(n * index_count).
    """
    if index_count == 0 or valid_count <= 1:
        return

    var stack = InlineArray[Int, 4 * SELECT_STACK_SLOTS](fill=0)
    var depth = 0

    @always_inline
    def push(k_begin: Int, k_end: Int, lo: Int, hi: Int) {mut stack, mut depth}:
        if k_begin >= k_end or hi - lo <= 1:
            return
        var slot = depth * 4
        stack[slot] = k_begin
        stack[slot + 1] = k_end
        stack[slot + 2] = lo
        stack[slot + 3] = hi
        depth += 1

    push(0, index_count, 0, valid_count)
    while depth > 0:
        depth -= 1
        var slot = depth * 4
        var k_begin = stack[slot]
        var k_end = stack[slot + 1]
        var lo = stack[slot + 2]
        var hi = stack[slot + 3]
        var mid = k_begin + (k_end - k_begin) // 2
        var k = Int(index_pointer(index_base, mid)[])
        partition_kth[dtype](
            Span(unsafe_ptr=value_pointer[dtype](buf_base, lo), length=hi - lo),
            k - lo,
        )
        # buf[k] is final, everything left of it is <=, everything right >=.
        push(k_begin, mid, lo, k)
        push(mid + 1, k_end, k + 1, hi)


struct NanQuantileKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """Selection- or sort-based quantiles over one prepared core span."""

    comptime Signature = Tuple[
        GUTensor[
            Self.dtype,
            False,
            CoreSpec[Dim[0]],
        ],
        GUTensor[
            Self.dtype,
            True,
            CoreSpec[Dim[QUANTILE_DIM]],
        ],
    ]

    var q_addr: Int
    var num_q: Int
    var workspace: Allocation[Scalar[Self.dtype]]
    var workspace_capacity: Int
    var index_workspace: Allocation[Int64]

    def __init__(out self, q_addr: Int, num_q: Int):
        self.q_addr = q_addr
        self.num_q = num_q
        # A zero-size allocation gives the copyable operation a valid owned
        # allocation while reserving element storage lazily on first apply.
        self.workspace = alloc(Layout[Scalar[Self.dtype]](count=0))
        self.workspace_capacity = 0
        # Every quantile contributes a floor and a ceil rank; the list is
        # sized once here so no core allocates while selecting.
        self.index_workspace = alloc(Layout[Int64](count=2 * num_q))

    def __init__(out self, *, copy: Self):
        self.q_addr = copy.q_addr
        self.num_q = copy.num_q
        # Worker copies start with independent operation-private workspace.
        self.workspace = alloc(Layout[Scalar[Self.dtype]](count=0))
        self.workspace_capacity = 0
        self.index_workspace = alloc(Layout[Int64](count=2 * copy.num_q))

    def __deinit__(deinit self):
        dealloc(self.workspace^)
        dealloc(self.index_workspace^)

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
    def workspace_base(self) -> Int:
        return Int(self.workspace.unsafe_ptr())

    @always_inline
    def index_base(self) -> Int:
        return Int(self.index_workspace.unsafe_ptr())

    @always_inline
    def q_ptr(self) -> Pointer[mut=False, Float64, MutAnyOrigin]:
        return Pointer[mut=False, Float64, MutAnyOrigin](
            unsafe_from_address=self.q_addr
        )

    def _compute_quantiles(
        self,
        buf_base: Int,
        valid_count: Int,
        out_ptr: Pointer[mut=True, Scalar[Self.dtype], _],
    ):
        var buf = value_pointer[Self.dtype](buf_base, 0)
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
                out_ptr[unsafe_offset=0] = endpoint_value[Self.dtype](v_low)
            else:
                var v_high_value = buf[unsafe_offset=low + 1]
                for j in range(low + 2, valid_count):
                    if buf[unsafe_offset=j] < v_high_value:
                        v_high_value = buf[unsafe_offset=j]
                var v_high = Float64(v_high_value)
                var frac = rank - Float64(low)
                out_ptr[unsafe_offset=0] = Scalar[Self.dtype](
                    v_low + frac * (v_high - v_low)
                )
            return

        if self.num_q == 0:
            return

        # NaNs were compacted out by apply, so the order statistics can be
        # taken from the whole valid span.  Collect the floor and ceil ranks
        # first: their count decides whether selecting those few positions
        # beats ordering everything.
        var index_base = self.index_base()
        var index_count = 0
        for m in range(self.num_q):
            var q = q_p[unsafe_offset=m]
            if isnan(q):
                continue
            var rank = Float64(valid_count - 1) * q
            var low = Int(rank)
            index_count = insert_index(index_base, index_count, low)
            index_count = insert_index(
                index_base, index_count, min(low + 1, valid_count - 1)
            )

        if prefers_selection(valid_count, index_count):
            multi_select[Self.dtype](
                buf_base, valid_count, index_base, index_count
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
                if high == low:
                    out_ptr[unsafe_offset=m] = endpoint_value[Self.dtype](v_low)
                else:
                    var v_high = Float64(buf[unsafe_offset=high])
                    out_ptr[unsafe_offset=m] = Scalar[Self.dtype](
                        v_low + frac * (v_high - v_low)
                    )

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
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
            self.workspace_base(),
            valid_count,
            output.write_span().unsafe_ptr(),
        )
