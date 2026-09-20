"""Benchmark every width/unroll combination for forward fill.

Each measurement uses the same contiguous floating-point scan and differs
only in the compile-time arguments passed to ``vectorize``.  The Cartesian
product is:

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
comptime BYTES_PER_VALUE = 8


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
    var block = SIMD[DType.float64, width](nan_value())
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block


@always_inline
def fill_lane(
    value: Float64, mut current: Float64, mut remaining: Int, allowed: Int
) -> Float64:
    if isnan(value):
        if remaining <= 0:
            current = nan_value()
        remaining -= 1
    else:
        current = value
        remaining = allowed
    return current


def fill_inputs(values_address: Int, length: Int):
    var values = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=values_address)
    for i in range(length):
        values[unsafe_offset=i] = Float64((i * 17) % 100_003) * 0.001
        if i % 29 < 4:
            values[unsafe_offset=i] = nan_value()


def vectorized_variant[width: Int, unroll_factor: Int](
    values_address: Int, output_address: Int, length: Int
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    var current = nan_value()
    var remaining = length

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, mut output, mut current, mut remaining, imm length
    }:
        var block = load_values[width](values, i, evl)
        comptime for lane in range(width):
            if lane < evl:
                block[lane] = fill_lane(block[lane], current, remaining, length)
        if evl == width:
            output.unsafe_store[width=width](i, block)
        else:
            comptime for lane in range(width):
                if lane < evl:
                    output[unsafe_offset=i + lane] = block[lane]

    vectorize[width, unroll_factor=unroll_factor](length, step)


def outputs_match(left_address: Int, right_address: Int, length: Int) -> Bool:
    var left = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=left_address)
    var right = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=right_address)
    for i in range(length):
        if isnan(left[unsafe_offset=i]) and isnan(right[unsafe_offset=i]):
            continue
        if left[unsafe_offset=i] != right[unsafe_offset=i]:
            return False
    return True


def run_variant[width: Int, unroll_factor: Int](
    values: Int, output: Int, reference: Int, length: Int
):
    for _ in range(WARMUPS):
        vectorized_variant[width, unroll_factor](values, output, length)

    var started = perf_counter_ns()
    for _ in range(REPEATS):
        vectorized_variant[width, unroll_factor](values, output, length)
    var elapsed = perf_counter_ns() - started
    print(
        "vectorize width=", width,
        " unroll=", unroll_factor,
        " average_ns=", Float64(elapsed) / Float64(REPEATS),
        " match=", outputs_match(reference, output, length),
    )


def main() raises:
    comptime length = DATA_BYTES // BYTES_PER_VALUE
    var values_storage = alloc(Layout[Float64](count=length))
    var reference_storage = alloc(Layout[Float64](count=length))
    var output_storage = alloc(Layout[Float64](count=length))
    var values = Int(values_storage.unsafe_ptr())
    var reference = Int(reference_storage.unsafe_ptr())
    var output = Int(output_storage.unsafe_ptr())

    fill_inputs(values, length)
    vectorized_variant[1, 1](values, reference, length)

    run_variant[1, 1](values, output, reference, length)
    run_variant[1, 2](values, output, reference, length)
    run_variant[1, 4](values, output, reference, length)
    run_variant[1, 8](values, output, reference, length)
    run_variant[1, 16](values, output, reference, length)
    run_variant[1, 32](values, output, reference, length)
    run_variant[2, 1](values, output, reference, length)
    run_variant[2, 2](values, output, reference, length)
    run_variant[2, 4](values, output, reference, length)
    run_variant[2, 8](values, output, reference, length)
    run_variant[2, 16](values, output, reference, length)
    run_variant[2, 32](values, output, reference, length)
    run_variant[4, 1](values, output, reference, length)
    run_variant[4, 2](values, output, reference, length)
    run_variant[4, 4](values, output, reference, length)
    run_variant[4, 8](values, output, reference, length)
    run_variant[4, 16](values, output, reference, length)
    run_variant[4, 32](values, output, reference, length)
    run_variant[8, 1](values, output, reference, length)
    run_variant[8, 2](values, output, reference, length)
    run_variant[8, 4](values, output, reference, length)
    run_variant[8, 8](values, output, reference, length)
    run_variant[8, 16](values, output, reference, length)
    run_variant[8, 32](values, output, reference, length)
    run_variant[16, 1](values, output, reference, length)
    run_variant[16, 2](values, output, reference, length)
    run_variant[16, 4](values, output, reference, length)
    run_variant[16, 8](values, output, reference, length)
    run_variant[16, 16](values, output, reference, length)
    run_variant[16, 32](values, output, reference, length)
    run_variant[32, 1](values, output, reference, length)
    run_variant[32, 2](values, output, reference, length)
    run_variant[32, 4](values, output, reference, length)
    run_variant[32, 8](values, output, reference, length)
    run_variant[32, 16](values, output, reference, length)
    run_variant[32, 32](values, output, reference, length)

    dealloc(values_storage^)
    dealloc(reference_storage^)
    dealloc(output_storage^)