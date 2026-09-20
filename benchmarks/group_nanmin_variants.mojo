"""Compare grouped nanmin loop shapes over the same scatter workload.

The five functions intentionally keep the update operation identical:

* ``vectorized_scalar`` uses ``vectorize[1, 8]``;
* ``direct_unroll`` performs the eight-way unroll explicitly;
* ``vectorized_pair`` uses ``vectorize[2, 4]``.
* ``vectorized_system_width`` uses the platform SIMD width;
* ``vectorized_system_width_x8`` uses eight times the platform SIMD width.

This isolates the loop/vectorization choice from allocation, label generation,
and the grouped output layout used by ``GroupNanMinMax``.
"""

from std.algorithm import vectorize
from std.math import isnan
from std.memory import alloc, dealloc
from std.memory.alloc import Layout
from std.sys.info import simd_width_of
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
def load_values[width: Int](pointer: Pointer[mut=False, Float64, ImmUntrackedOrigin], i: Int, evl: Int) -> SIMD[DType.float64, width]:
    if evl == width:
        return pointer.unsafe_load[width=width](i)
    var block = SIMD[DType.float64, width](0)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block


@always_inline
def load_labels[width: Int](pointer: Pointer[mut=False, Int64, ImmUntrackedOrigin], i: Int, evl: Int) -> SIMD[DType.int64, width]:
    if evl == width:
        return pointer.unsafe_load[width=width](i)
    var block = SIMD[DType.int64, width](-1)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block


@always_inline
def update_lane(destination: Pointer[mut=True, Float64, MutAnyOrigin], label: Int64, value: Float64):
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


def vectorized_scalar(values_address: Int, labels_address: Int, output_address: Int, length: Int):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var labels = Pointer[mut=False, Int64, ImmUntrackedOrigin](unsafe_from_address=labels_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, imm labels, mut output
    }:
        var value_block = load_values[1](values, i, evl)
        var label_block = load_labels[1](labels, i, evl)
        comptime for lane in range(1):
            if lane < evl:
                update_lane(output, label_block[lane], value_block[lane])

    vectorize[1, unroll_factor=8](length, step)


def direct_unroll(values_address: Int, labels_address: Int, output_address: Int, length: Int):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var labels = Pointer[mut=False, Int64, ImmUntrackedOrigin](unsafe_from_address=labels_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    var i = 0
    while i < length:
        # Mojo 1.1 rejects @unroll on loop statements; comptime for emits the
        # same fixed eight-way unrolled loop without vectorize.
        comptime for offset in range(8):
            if i + offset < length:
                update_lane(output, labels[unsafe_offset=i + offset], values[unsafe_offset=i + offset])
        i += 8


def vectorized_pair(values_address: Int, labels_address: Int, output_address: Int, length: Int):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var labels = Pointer[mut=False, Int64, ImmUntrackedOrigin](unsafe_from_address=labels_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, imm labels, mut output
    }:
        var value_block = load_values[2](values, i, evl)
        var label_block = load_labels[2](labels, i, evl)
        comptime for lane in range(2):
            if lane < evl:
                update_lane(output, label_block[lane], value_block[lane])

    vectorize[2, unroll_factor=4](length, step)


def vectorized_system_width(values_address: Int, labels_address: Int, output_address: Int, length: Int):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var labels = Pointer[mut=False, Int64, ImmUntrackedOrigin](unsafe_from_address=labels_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    comptime width = simd_width_of[Float64]()

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, imm labels, mut output
    }:
        var value_block = load_values[width](values, i, evl)
        var label_block = load_labels[width](labels, i, evl)
        comptime for lane in range(width):
            if lane < evl:
                update_lane(output, label_block[lane], value_block[lane])

    vectorize[width](length, step)


def vectorized_system_width_x8(values_address: Int, labels_address: Int, output_address: Int, length: Int):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var labels = Pointer[mut=False, Int64, ImmUntrackedOrigin](unsafe_from_address=labels_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    comptime width = simd_width_of[Float64]() * 8

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, imm labels, mut output
    }:
        var value_block = load_values[width](values, i, evl)
        var label_block = load_labels[width](labels, i, evl)
        comptime for lane in range(width):
            if lane < evl:
                update_lane(output, label_block[lane], value_block[lane])

    vectorize[width](length, step)


def outputs_match(left_address: Int, right_address: Int) -> Bool:
    var left = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=left_address)
    var right = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=right_address)
    for i in range(NUM_GROUPS):
        if isnan(left[unsafe_offset=i]) and isnan(right[unsafe_offset=i]):
            continue
        if left[unsafe_offset=i] != right[unsafe_offset=i]:
            return False
    return True


def run_variant(kind: Int, values: Int, labels: Int, output: Int, reference: Int, length: Int):
    for _ in range(WARMUPS):
        initialize_output(output)
        if kind == 0:
            vectorized_scalar(values, labels, output, length)
        elif kind == 1:
            direct_unroll(values, labels, output, length)
        elif kind == 2:
            vectorized_pair(values, labels, output, length)
        elif kind == 3:
            vectorized_system_width(values, labels, output, length)
        else:
            vectorized_system_width_x8(values, labels, output, length)

    var started = perf_counter_ns()
    for _ in range(REPEATS):
        initialize_output(output)
        if kind == 0:
            vectorized_scalar(values, labels, output, length)
        elif kind == 1:
            direct_unroll(values, labels, output, length)
        elif kind == 2:
            vectorized_pair(values, labels, output, length)
        elif kind == 3:
            vectorized_system_width(values, labels, output, length)
        else:
            vectorized_system_width_x8(values, labels, output, length)
    var elapsed = perf_counter_ns() - started
    if kind == 0:
        print("vectorize width=1 unroll=8", end="")
    elif kind == 1:
        print("direct unroll=8", end="")
    elif kind == 2:
        print("vectorize width=2 unroll=4", end="")
    elif kind == 3:
        print("vectorize width=system", end="")
    else:
        print("vectorize width=system*8", end="")
    print(" average_ns=", Float64(elapsed) / Float64(REPEATS), " match=", outputs_match(reference, output))


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
    vectorized_scalar(values, labels, reference, length)
    run_variant(0, values, labels, output, reference, length)
    run_variant(1, values, labels, output, reference, length)
    run_variant(2, values, labels, output, reference, length)
    run_variant(3, values, labels, output, reference, length)
    run_variant(4, values, labels, output, reference, length)

    dealloc(values_storage^)
    dealloc(labels_storage^)
    dealloc(reference_storage^)
    dealloc(output_storage^)