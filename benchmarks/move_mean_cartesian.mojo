"""Benchmark every width/unroll combination for trailing moving mean.

Each measurement uses the same contiguous NaN-aware moving-mean recurrence and
differs only in the compile-time arguments passed to ``vectorize``.  The
Cartesian product is:

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
comptime WINDOW = 64
comptime MIN_COUNT = 32


@always_inline
def nan_value() -> Float64:
    var zero = Float64(0)
    return zero / zero


@always_inline
def load_values[width: Int](
    pointer: Pointer[mut=False, Float64, ImmUntrackedOrigin], i: Int, evl: Int
) -> SIMD[DType.float64, width]:
    var block = SIMD[DType.float64, width](0)
    if evl == width:
        return pointer.unsafe_load[width=width](i)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block


def fill_inputs(values_address: Int, length: Int):
    var values = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=values_address)
    for i in range(length):
        values[unsafe_offset=i] = Float64((i * 17) % 100_003) * 0.001
        if i % 29 == 0:
            values[unsafe_offset=i] = nan_value()


@always_inline
def sequential_variant(values_address: Int, output_address: Int, length: Int):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    var input_offset = 0
    var total = Float64(0)
    var count = Float64(0)
    var threshold = Float64(MIN_COUNT)

    while input_offset < length:
        var active = min(1, length - input_offset)
        if input_offset < WINDOW:
            active = min(active, WINDOW - input_offset)
        var entering = SIMD[DType.float64, 1](0)
        entering[0] = values[unsafe_offset=input_offset]
        var expiring = SIMD[DType.float64, 1](0)
        if input_offset >= WINDOW:
            expiring[0] = values[unsafe_offset=input_offset - WINDOW]
        var entering_value = entering[0]
        if isnan(entering_value):
            entering_value = 0.0
        else:
            count += 1.0
        var expiring_value = expiring[0]
        if input_offset >= WINDOW:
            if isnan(expiring_value):
                expiring_value = 0.0
            else:
                count -= 1.0
        total += entering_value - expiring_value
        if count >= threshold:
            output[unsafe_offset=input_offset] = total / count
        else:
            output[unsafe_offset=input_offset] = nan_value()
        input_offset += active


def vectorized_variant[width: Int, unroll_factor: Int](
    values_address: Int, output_address: Int, length: Int
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    var total = Float64(0)
    var count = Float64(0)
    var threshold = Float64(MIN_COUNT)

    def step[vector_width: Int](i: Int, evl: Int) {
        imm values, mut output, mut total, mut count, imm threshold
    }:
        var entering = load_values[width](values, i, evl)
        var expiring = SIMD[DType.float64, width](0)
        if i >= WINDOW:
            expiring = load_values[width](values, i - WINDOW, evl)
        comptime for lane in range(width):
            if lane < evl:
                var entering_value = entering[lane]
                if isnan(entering_value):
                    entering_value = 0.0
                else:
                    count += 1.0
                var expiring_value = expiring[lane]
                if i >= WINDOW:
                    if isnan(expiring_value):
                        expiring_value = 0.0
                    else:
                        count -= 1.0
                total += entering_value - expiring_value
                if count >= threshold:
                    output[unsafe_offset=i + lane] = total / count
                else:
                    output[unsafe_offset=i + lane] = nan_value()

    vectorize[width, unroll_factor=unroll_factor](length, step)


def outputs_match(left_address: Int, right_address: Int, length: Int) -> Bool:
    var left = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=left_address)
    var right = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=right_address)
    for i in range(length):
        var left_value = left[unsafe_offset=i]
        var right_value = right[unsafe_offset=i]
        if isnan(left_value) and isnan(right_value):
            continue
        if left_value != right_value:
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


def run_sequential(values: Int, output: Int, reference: Int, length: Int):
    for _ in range(WARMUPS):
        sequential_variant(values, output, length)

    var started = perf_counter_ns()
    for _ in range(REPEATS):
        sequential_variant(values, output, length)
    var elapsed = perf_counter_ns() - started
    print(
        "implementation=old_sequential average_ns=",
        Float64(elapsed) / Float64(REPEATS),
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
    sequential_variant(values, reference, length)

    run_sequential(values, output, reference, length)
    print("implementation=new_vectorized width=8 unroll=8")
    run_variant[8, 8](values, output, reference, length)
    print("implementation=cartesian_variants")
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