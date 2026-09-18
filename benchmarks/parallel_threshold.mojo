"""Measure serial row sums against Mojo's current parallel loop primitive.

The benchmark is intentionally small and standalone.  It compares the work
shape that mojagg's dispatch policy sees: ``rows`` independent outer tasks,
each with ``columns`` values to reduce.  The input and output buffers are
allocated once per case and are not included in the measured region.

Older Mojo examples express the parallel loop as::

    for i in parallel_range(rows):
        ...

Current Mojo 1.0 exposes the same style of work distribution through
``max.algorithm.parallelize``.  The ``row_sums_parallel`` function below is
the equivalent comparison for this checkout.
"""

from max.algorithm import parallelize
from std.memory import alloc, dealloc
from std.memory.alloc import Layout
from std.sys import size_of
from std.time import perf_counter_ns


comptime WARMUPS = 1
comptime REPEATS = 5


@fieldwise_init
struct Row(ImplicitlyCopyable):
    """A borrowed row view with the ``matrix[i].sum()`` operation shape."""

    var address: Int
    var length: Int

    @always_inline
    def sum(self) -> Float64:
        var values = Pointer[
            mut=False,
            Float64,
            ImmUntrackedOrigin,
        ](unsafe_from_address=self.address)
        var total = Float64(0)
        for column in range(self.length):
            total += values[unsafe_offset=column]
        return total


@fieldwise_init
struct Matrix(ImplicitlyCopyable):
    """A non-owning row-major matrix used by both loop variants."""

    var address: Int
    var rows: Int
    var columns: Int

    @always_inline
    def __getitem__(self, row: Int) -> Row:
        var row_address = self.address + row * self.columns * size_of[Float64]()
        return Row(row_address, self.columns)


def fill_matrix(address: Int, rows: Int, columns: Int):
    var values = Pointer[
        mut=True,
        Float64,
        MutAnyOrigin,
    ](unsafe_from_address=address)
    for row in range(rows):
        for column in range(columns):
            # Keep the data deterministic and make every row non-identical.
            values[unsafe_offset=row * columns + column] = Float64(
                1 + (column % 31)
            ) + Float64(row) * 0.001


def row_sums_serial(matrix: Matrix, output_address: Int):
    var output = Pointer[
        mut=True,
        Float64,
        MutAnyOrigin,
    ](unsafe_from_address=output_address)
    for i in range(matrix.rows):
        output[unsafe_offset=i] = matrix[i].sum()


def row_sums_parallel(matrix: Matrix, output_address: Int):
    def worker(i: Int) {
        imm matrix, imm output_address
    }:
        var output = Pointer[
            mut=True,
            Float64,
            MutAnyOrigin,
        ](unsafe_from_address=output_address)
        output[unsafe_offset=i] = matrix[i].sum()

    parallelize(worker, matrix.rows)


def output_checksum(address: Int, rows: Int) -> Float64:
    var output = Pointer[
        mut=False,
        Float64,
        ImmUntrackedOrigin,
    ](unsafe_from_address=address)
    var checksum = Float64(0)
    for i in range(rows):
        checksum += output[unsafe_offset=i]
    return checksum


def outputs_match(serial_address: Int, parallel_address: Int, rows: Int) -> Bool:
    var serial = Pointer[
        mut=False,
        Float64,
        ImmUntrackedOrigin,
    ](unsafe_from_address=serial_address)
    var parallel = Pointer[
        mut=False,
        Float64,
        ImmUntrackedOrigin,
    ](unsafe_from_address=parallel_address)
    for i in range(rows):
        if serial[unsafe_offset=i] != parallel[unsafe_offset=i]:
            return False
    return True


def run_case(rows: Int, columns: Int):
    var element_count = rows * columns
    var input_storage = alloc(Layout[Float64](count=element_count))
    var serial_storage = alloc(Layout[Float64](count=rows))
    var parallel_storage = alloc(Layout[Float64](count=rows))

    var matrix = Matrix(Int(input_storage.unsafe_ptr()), rows, columns)
    var serial_address = Int(serial_storage.unsafe_ptr())
    var parallel_address = Int(parallel_storage.unsafe_ptr())
    fill_matrix(matrix.address, rows, columns)

    for _ in range(WARMUPS):
        row_sums_serial(matrix, serial_address)
        row_sums_parallel(matrix, parallel_address)

    var serial_start = perf_counter_ns()
    for _ in range(REPEATS):
        row_sums_serial(matrix, serial_address)
    var serial_elapsed = perf_counter_ns() - serial_start

    var parallel_start = perf_counter_ns()
    for _ in range(REPEATS):
        row_sums_parallel(matrix, parallel_address)
    var parallel_elapsed = perf_counter_ns() - parallel_start

    var matches = outputs_match(serial_address, parallel_address, rows)
    var serial_checksum = output_checksum(serial_address, rows)
    var parallel_checksum = output_checksum(parallel_address, rows)
    var serial_ns = Float64(serial_elapsed) / Float64(REPEATS)
    var parallel_ns = Float64(parallel_elapsed) / Float64(REPEATS)
    var speedup = serial_ns / parallel_ns

    print(
        "rows=",
        rows,
        " columns=",
        columns,
        " elements=",
        element_count,
        " serial_ns=",
        serial_ns,
        " parallel_ns=",
        parallel_ns,
        " speedup=",
        speedup,
        " match=",
        matches,
        " checksum=",
        serial_checksum,
        " parallel_checksum=",
        parallel_checksum,
    )

    dealloc(input_storage^)
    dealloc(serial_storage^)
    dealloc(parallel_storage^)


def main() raises:
    print("serial row loop vs parallelize row loop")
    print("warmups=", WARMUPS, " repeats=", REPEATS)
    print("Use cases with rows >= parallel_min_groups to estimate the row-length threshold.")

    # First sweep: hold the row length above the likely threshold and vary the
    # number of independent rows.  This exposes the minimum outer-group gate.
    run_case(1, 262_144)
    run_case(2, 262_144)
    run_case(4, 262_144)
    run_case(8, 262_144)
    run_case(16, 262_144)
    run_case(32, 262_144)
    run_case(64, 262_144)

    # Second sweep: hold enough rows for a worker pool and vary the row length.
    # The crossover is the useful evidence for parallel_threshold.
    run_case(16, 1_024)
    run_case(16, 4_096)
    run_case(16, 16_384)
    run_case(16, 24_576)
    run_case(16, 32_768)
    run_case(16, 49_152)
    run_case(16, 65_536)
    run_case(16, 131_072)
    run_case(16, 262_144)
    run_case(16, 524_288)
