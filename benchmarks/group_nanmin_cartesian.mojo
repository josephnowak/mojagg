"""Benchmark every width/unroll combination for grouped nanmin.

Each measurement uses the same grouped scatter workload and differs only in
the compile-time arguments passed to ``vectorize``.  The Cartesian product is:

* width: ``[1, 2, 4, 8, 16, 32]``;
* unroll factor: ``[1, 2, 4, 8, 16, 32]``.
"""

from std.algorithm import vectorize
from std.math import isnan
from std.memory import alloc, dealloc
from std.memory.alloc import Layout
from std.time import perf_counter_ns

comptime WARMUPS = 2
comptime REPEATS = 10
comptime DATA_BYTES = 4 << 30
comptime BYTES_PER_ROW = 16  # Float64 value plus Int64 label.
comptime NUM_GROUPS = 256


@always_inline
def nan_value() -> Float64:
    var zero = Float64(0)
    return zero / zero


@always_inline
def load_values[width: Int](
    pointer: Pointer[mut=False, Float64, ImmUntrackedOrigin], i: Int, evl: Int
) -> SIMD[DType.float64, width]:
    if evl == width:
        return pointer.unsafe_load[width=width](i)
    var block = SIMD[DType.float64, width](0)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block


@always_inline
def load_labels[width: Int](
    pointer: Pointer[mut=False, Int64, ImmUntrackedOrigin], i: Int, evl: Int
) -> SIMD[DType.int64, width]:
    if evl == width:
        return pointer.unsafe_load[width=width](i)
    var block = SIMD[DType.int64, width](-1)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block


@always_inline
def update_lane(
    destination: Pointer[mut=True, Float64, MutAnyOrigin], label: Int64, value: Float64
):
    if label < 0:
        return
    if isnan(value):
        return
    if isnan(destination[unsafe_offset=label]) or value < destination[unsafe_offset=label]:
        destination[unsafe_offset=label] = value


def fill_inputs(values_address: Int, labels_address: Int, length: Int):
    var values = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=values_address)
    var labels = Pointer[mut=True, Int64, MutAnyOrigin](unsafe_from_address=labels_address)
    for i in range(length):
        values[unsafe_offset=i] = Float64((i * 17) % 100_003) * 0.001
        if i % 29 == 0:
            values[unsafe_offset=i] = nan_value()
        labels[unsafe_offset=i] = Int64((i * 13) % NUM_GROUPS)
        if i % 127 == 0:
            labels[unsafe_offset=i] = -1


def initialize_output(output_address: Int):
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    for i in range(NUM_GROUPS):
        output[unsafe_offset=i] = nan_value()


def vectorized_variant[width: Int, unroll_factor: Int](
    values_address: Int, labels_address: Int, output_address: Int, length: Int
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var labels = Pointer[mut=False, Int64, ImmUntrackedOrigin](unsafe_from_address=labels_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, imm labels, mut output
    }:
        var value_block = load_values[width](values, i, evl)
        var label_block = load_labels[width](labels, i, evl)
        comptime for lane in range(width):
            if lane < evl:
                update_lane(output, label_block[lane], value_block[lane])

    vectorize[width, unroll_factor=unroll_factor](length, step)


def outputs_match(left_address: Int, right_address: Int) -> Bool:
    var left = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=left_address)
    var right = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=right_address)
    for i in range(NUM_GROUPS):
        if isnan(left[unsafe_offset=i]) and isnan(right[unsafe_offset=i]):
            continue
        if left[unsafe_offset=i] != right[unsafe_offset=i]:
            return False
    return True


def run_variant[width: Int, unroll_factor: Int](
    values: Int, labels: Int, output: Int, reference: Int, length: Int
):
    for _ in range(WARMUPS):
        initialize_output(output)
        vectorized_variant[width, unroll_factor](values, labels, output, length)

    var started = perf_counter_ns()
    for _ in range(REPEATS):
        initialize_output(output)
        vectorized_variant[width, unroll_factor](values, labels, output, length)
    var elapsed = perf_counter_ns() - started
    print(
        "vectorize width=", width,
        " unroll=", unroll_factor,
        " average_ns=", Float64(elapsed) / Float64(REPEATS),
        " match=", outputs_match(reference, output),
    )


def main() raises:
    comptime length = DATA_BYTES // BYTES_PER_ROW
    var values_storage = alloc(Layout[Float64](count=length))
    var labels_storage = alloc(Layout[Int64](count=length))
    var reference_storage = alloc(Layout[Float64](count=NUM_GROUPS))
    var output_storage = alloc(Layout[Float64](count=NUM_GROUPS))
    var values = Int(values_storage.unsafe_ptr())
    var labels = Int(labels_storage.unsafe_ptr())
    var reference = Int(reference_storage.unsafe_ptr())
    var output = Int(output_storage.unsafe_ptr())

    fill_inputs(values, labels, length)
    initialize_output(reference)
    vectorized_variant[1, 1](values, labels, reference, length)

    run_variant[1, 1](values, labels, output, reference, length)
    run_variant[1, 2](values, labels, output, reference, length)
    run_variant[1, 4](values, labels, output, reference, length)
    run_variant[1, 8](values, labels, output, reference, length)
    run_variant[1, 16](values, labels, output, reference, length)
    run_variant[1, 32](values, labels, output, reference, length)
    run_variant[2, 1](values, labels, output, reference, length)
    run_variant[2, 2](values, labels, output, reference, length)
    run_variant[2, 4](values, labels, output, reference, length)
    run_variant[2, 8](values, labels, output, reference, length)
    run_variant[2, 16](values, labels, output, reference, length)
    run_variant[2, 32](values, labels, output, reference, length)
    run_variant[4, 1](values, labels, output, reference, length)
    run_variant[4, 2](values, labels, output, reference, length)
    run_variant[4, 4](values, labels, output, reference, length)
    run_variant[4, 8](values, labels, output, reference, length)
    run_variant[4, 16](values, labels, output, reference, length)
    run_variant[4, 32](values, labels, output, reference, length)
    run_variant[8, 1](values, labels, output, reference, length)
    run_variant[8, 2](values, labels, output, reference, length)
    run_variant[8, 4](values, labels, output, reference, length)
    run_variant[8, 8](values, labels, output, reference, length)
    run_variant[8, 16](values, labels, output, reference, length)
    run_variant[8, 32](values, labels, output, reference, length)
    run_variant[16, 1](values, labels, output, reference, length)
    run_variant[16, 2](values, labels, output, reference, length)
    run_variant[16, 4](values, labels, output, reference, length)
    run_variant[16, 8](values, labels, output, reference, length)
    run_variant[16, 16](values, labels, output, reference, length)
    run_variant[16, 32](values, labels, output, reference, length)
    run_variant[32, 1](values, labels, output, reference, length)
    run_variant[32, 2](values, labels, output, reference, length)
    run_variant[32, 4](values, labels, output, reference, length)
    run_variant[32, 8](values, labels, output, reference, length)
    run_variant[32, 16](values, labels, output, reference, length)
    run_variant[32, 32](values, labels, output, reference, length)

    dealloc(values_storage^)
    dealloc(labels_storage^)
    dealloc(reference_storage^)
    dealloc(output_storage^)