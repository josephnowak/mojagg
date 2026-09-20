"""Benchmark every width/unroll combination for NaN-aware all predicate.

Each measurement uses the same contiguous floating-point predicate reduction
and differs only in the compile-time arguments passed to ``vectorize``.  The
Cartesian product is:

* width: ``[1, 2, 4, 8, 16, 32]``;
* unroll factor: ``[1, 2, 4, 8, 16, 32]``.

The input is all NaN so every vectorized block is visited instead of allowing
the all predicate to terminate early at the first non-NaN value.
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
    var block = SIMD[DType.float64, width](0)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block


def fill_inputs(values_address: Int, length: Int):
    var values = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=values_address)
    for i in range(length):
        values[unsafe_offset=i] = nan_value()


def vectorized_variant[width: Int, unroll_factor: Int](
    values_address: Int, output_address: Int, length: Int
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var output = Pointer[mut=True, Bool, MutAnyOrigin](unsafe_from_address=output_address)
    var acc = SIMD[DType.bool, width](fill=True)

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, mut acc
    }:
        var block = load_values[width](values, i, evl)
        acc = acc & isnan(block)

    vectorize[width, unroll_factor=unroll_factor](length, step)
    output[unsafe_offset=0] = acc.reduce_and()


def results_match(left_address: Int, right_address: Int) -> Bool:
    var left = Pointer[mut=False, Bool, ImmUntrackedOrigin](unsafe_from_address=left_address)
    var right = Pointer[mut=False, Bool, ImmUntrackedOrigin](unsafe_from_address=right_address)
    return left[unsafe_offset=0] == right[unsafe_offset=0]


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
        " match=", results_match(reference, output),
    )


def main() raises:
    comptime length = DATA_BYTES // BYTES_PER_VALUE
    var values_storage = alloc(Layout[Float64](count=length))
    var reference_storage = alloc(Layout[Bool](count=1))
    var output_storage = alloc(Layout[Bool](count=1))
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