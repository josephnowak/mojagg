"""Compare the sort-based multi-quantile path against a multi-kth selection.

The current ``NanQuantileKernel`` orders a whole core span with ``sort`` when
more than one quantile is requested.  NumPy instead calls
``np.partition(arr, kth=unique_indices)``, which is a *multi-select*: partition
around one requested order statistic, then recurse only into the sub-range that
still contains the remaining requested indices.

Both paths are implemented here over the same prepared buffer so the crossover
can be measured instead of guessed.  Every timed repetition restores the
unsorted input and the restore cost is measured separately so it can be
subtracted.  All scratch state is addressed through raw pointers, exactly like
the kernel does, so neither variant is charged for copying scratch arrays.
"""

from std.builtin.sort import partition, sort
from std.collections import Span
from std.collections import Array
from std.math import min
from std.memory import alloc, dealloc
from std.memory.alloc import Layout
from std.sys import size_of
from std.time import perf_counter_ns


comptime WARMUPS = 2
comptime REPEATS = 20
comptime STACK_SLOTS = 64


@always_inline
def value_pointer(
    base: Int, offset: Int
) -> Pointer[mut=True, Float64, MutAnyOrigin]:
    return Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=base + offset * size_of[Float64]()
    )


@always_inline
def index_pointer(
    base: Int, offset: Int
) -> Pointer[mut=True, Int64, MutAnyOrigin]:
    return Pointer[mut=True, Int64, MutAnyOrigin](
        unsafe_from_address=base + offset * size_of[Int64]()
    )


@always_inline
def partition_range(base: Int, lo: Int, hi: Int, k: Int):
    """Partition ``[lo, hi)`` around the global order statistic ``k``."""

    var span = Span(unsafe_ptr=value_pointer(base, lo), length=hi - lo)
    def cmp(x: Float64, y: Float64) -> Bool:
        return x < y

    partition(span, k - lo, cmp)


@always_inline
def insert_index(index_base: Int, count: Int, candidate: Int) -> Int:
    """Insert one index into an ascending, deduplicated list."""
    var position = count
    while (
        position > 0 and Int(index_pointer(index_base, position - 1)[]) > candidate
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


def build_indices(
    q_base: Int, num_q: Int, valid_count: Int, index_base: Int
) -> Int:
    """Collect the floor/ceil ranks as a sorted, deduplicated index list."""
    var count = 0
    for m in range(num_q):
        var rank = Float64(valid_count - 1) * value_pointer(q_base, m)[]
        var low = Int(rank)
        var high = min(low + 1, valid_count - 1)
        count = insert_index(index_base, count, low)
        count = insert_index(index_base, count, high)
    return count


def multi_select(
    base: Int, valid_count: Int, index_base: Int, index_count: Int
):
    """Place every requested order statistic without a full ordering."""
    if index_count == 0 or valid_count <= 1:
        return

    var stack = Array[Int, 4 * STACK_SLOTS](fill=0)
    var depth = 0

    @always_inline
    def push(k_begin: Int, k_end: Int, lo: Int, hi: Int) {
        mut stack, mut depth
    }:
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
        partition_range(base, lo, hi, k)
        push(k_begin, mid, lo, k)
        push(mid + 1, k_end, k + 1, hi)


@always_inline
def interpolate(
    base: Int, valid_count: Int, q_base: Int, num_q: Int, out_base: Int
):
    for m in range(num_q):
        var rank = Float64(valid_count - 1) * value_pointer(q_base, m)[]
        var low = Int(rank)
        var high = min(low + 1, valid_count - 1)
        var v_low = value_pointer(base, low)[]
        var result = v_low
        if high != low:
            var v_high = value_pointer(base, high)[]
            result = v_low + (rank - Float64(low)) * (v_high - v_low)
        value_pointer(out_base, m)[] = result


def quantiles_by_sort(
    base: Int, valid_count: Int, q_base: Int, num_q: Int, out_base: Int
):
    sort(Span(unsafe_ptr=value_pointer(base, 0), length=valid_count))
    interpolate(base, valid_count, q_base, num_q, out_base)


def quantiles_by_select(
    base: Int,
    valid_count: Int,
    q_base: Int,
    num_q: Int,
    out_base: Int,
    index_base: Int,
):
    var index_count = build_indices(q_base, num_q, valid_count, index_base)
    multi_select(base, valid_count, index_base, index_count)
    interpolate(base, valid_count, q_base, num_q, out_base)


def fill_source(base: Int, count: Int):
    """Deterministic pseudo-random doubles; no ordering is assumed anywhere."""
    var state = UInt64(0x2545F4914F6CDD1D)
    for i in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var bits = (state >> 11) & UInt64(0x1FFFFFFFFFFFFF)
        value_pointer(base, i)[] = Float64(bits) * 1.1102230246251565e-16


@always_inline
def restore(source_base: Int, work_base: Int, count: Int):
    for i in range(count):
        value_pointer(work_base, i)[] = value_pointer(source_base, i)[]


def run_case(valid_count: Int, num_q: Int):
    var source_storage = alloc(Layout[Float64](count=valid_count))
    var work_storage = alloc(Layout[Float64](count=valid_count))
    var q_storage = alloc(Layout[Float64](count=num_q))
    var sort_out_storage = alloc(Layout[Float64](count=num_q))
    var select_out_storage = alloc(Layout[Float64](count=num_q))
    # The kernel keeps the index scratch alive per worker, so the benchmark
    # must not charge the selection path for creating it on every call.
    var index_storage = alloc(Layout[Int64](count=2 * num_q))

    var source_base = Int(source_storage.unsafe_ptr())
    var work_base = Int(work_storage.unsafe_ptr())
    var q_base = Int(q_storage.unsafe_ptr())
    var sort_out_base = Int(sort_out_storage.unsafe_ptr())
    var select_out_base = Int(select_out_storage.unsafe_ptr())
    var index_base = Int(index_storage.unsafe_ptr())

    fill_source(source_base, valid_count)
    for m in range(num_q):
        value_pointer(q_base, m)[] = Float64(m + 1) / Float64(num_q + 1)

    for _ in range(WARMUPS):
        restore(source_base, work_base, valid_count)
        quantiles_by_sort(
            work_base, valid_count, q_base, num_q, sort_out_base
        )
        restore(source_base, work_base, valid_count)
        quantiles_by_select(
            work_base, valid_count, q_base, num_q, select_out_base, index_base
        )

    var matches = True
    for m in range(num_q):
        if (
            value_pointer(sort_out_base, m)[]
            != value_pointer(select_out_base, m)[]
        ):
            matches = False

    var restore_start = perf_counter_ns()
    for _ in range(REPEATS):
        restore(source_base, work_base, valid_count)
    var restore_elapsed = perf_counter_ns() - restore_start

    var sort_start = perf_counter_ns()
    for _ in range(REPEATS):
        restore(source_base, work_base, valid_count)
        quantiles_by_sort(
            work_base, valid_count, q_base, num_q, sort_out_base
        )
    var sort_elapsed = perf_counter_ns() - sort_start

    var select_start = perf_counter_ns()
    for _ in range(REPEATS):
        restore(source_base, work_base, valid_count)
        quantiles_by_select(
            work_base, valid_count, q_base, num_q, select_out_base, index_base
        )
    var select_elapsed = perf_counter_ns() - select_start

    var restore_ns = Float64(restore_elapsed) / Float64(REPEATS)
    var sort_ns = Float64(sort_elapsed) / Float64(REPEATS) - restore_ns
    var select_ns = Float64(select_elapsed) / Float64(REPEATS) - restore_ns

    print(
        "n=",
        valid_count,
        " num_q=",
        num_q,
        " sort_ns=",
        sort_ns,
        " select_ns=",
        select_ns,
        " speedup=",
        sort_ns / select_ns,
        " match=",
        matches,
    )

    dealloc(source_storage^)
    dealloc(work_storage^)
    dealloc(q_storage^)
    dealloc(sort_out_storage^)
    dealloc(select_out_storage^)
    dealloc(index_storage^)


def main() raises:
    print("sort-based quantiles vs multi-kth selection (restore cost removed)")
    print("warmups=", WARMUPS, " repeats=", REPEATS)

    var quantile_counts: Array[Int, 8] = [1, 2, 3, 5, 9, 19, 49, 99]
    var lengths: Array[Int, 9] = [
        64,
        256,
        1_024,
        2_048,
        4_096,
        16_384,
        65_536,
        262_144,
        1_048_576,
    ]
    for q_slot in range(len(quantile_counts)):
        for n_slot in range(len(lengths)):
            run_case(lengths[n_slot], quantile_counts[q_slot])
