"""Benchmark every width/unroll combination for ``nancovmatrix``.

The input is a contiguous ``(N_VARS, N_OBS)`` matrix and the output is a
contiguous ``(N_VARS, N_VARS)`` covariance matrix.  The Cartesian product is:

* width: ``[1, 2, 4, 8, 16, 32]``;
* unroll factor: ``[1, 2, 4, 8, 16, 32]``.

The output allocation is deliberately bounded: with the dimensions below it
uses ``N_VARS * N_VARS * 8`` bytes, or 32 KiB, well below the 4 GiB limit.
"""

from std.algorithm import vectorize
from std.math import isnan
from std.memory import alloc, dealloc
from std.memory.alloc import Layout
from std.time import perf_counter_ns

comptime WARMUPS = 2
comptime REPEATS = 10
comptime N_VARS = 64
comptime N_OBS = 32_768
comptime BYTES_PER_VALUE = 8
comptime MAX_OUTPUT_BYTES = 4 << 30
comptime OUTPUT_BYTES = N_VARS * N_VARS * BYTES_PER_VALUE


@always_inline
def nan_value() -> Float64:
    var zero = Float64(0)
    return zero / zero


@always_inline
def load_values[width: Int](
    pointer: Pointer[mut=False, Float64, ImmUntrackedOrigin],
    index: Int,
    evl: Int,
) -> SIMD[DType.float64, width]:
    if evl == width:
        return pointer.unsafe_load[width=width](index)
    var block = SIMD[DType.float64, width](0)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=index + lane]
    return block


def fill_inputs(values_address: Int):
    var values = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=values_address
    )
    for variable in range(N_VARS):
        for observation in range(N_OBS):
            var value = Float64(
                ((variable + 3) * (observation + 11) % 100_003)
            ) * 0.001
            if (variable * 13 + observation) % 37 == 0:
                value = nan_value()
            values[unsafe_offset=variable * N_OBS + observation] = value


def covariance_pair[width: Int, unroll_factor: Int](
    values_address: Int, row_i: Int, row_j: Int, output_address: Int
):
    var values_i = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address + row_i * N_OBS * BYTES_PER_VALUE
    )
    var values_j = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address + row_j * N_OBS * BYTES_PER_VALUE
    )
    var output = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=output_address
    )
    var count = SIMD[DType.float64, width](0)
    var sum_x = SIMD[DType.float64, width](0)
    var sum_y = SIMD[DType.float64, width](0)
    var sum_xy = SIMD[DType.float64, width](0)
    comptime zero = SIMD[DType.float64, width](0)
    comptime one = SIMD[DType.float64, width](1)

    def step[vector_width: Int](index: Int, evl: Int) {
        imm values_i,
        imm values_j,
        mut count,
        mut sum_x,
        mut sum_y,
        mut sum_xy,
    }:
        var raw_x = load_values[width](values_i, index, evl)
        var raw_y = load_values[width](values_j, index, evl)
        var missing = isnan(raw_x) | isnan(raw_y)
        var x = missing.select(zero, raw_x)
        var y = missing.select(zero, raw_y)
        count += missing.select(zero, one)
        sum_x += x
        sum_y += y
        sum_xy += x * y

    vectorize[width, unroll_factor=unroll_factor](N_OBS, step)

    var n = count.reduce_add()
    var result = nan_value()
    if n > 1.0:
        var mean_x = sum_x.reduce_add() / n
        var mean_y = sum_y.reduce_add() / n
        result = (sum_xy.reduce_add() / n - mean_x * mean_y) * n / (n - 1.0)
    output[unsafe_offset=row_i * N_VARS + row_j] = result


def matrix_variant[width: Int, unroll_factor: Int](
    values_address: Int, output_address: Int
):
    for row_i in range(N_VARS):
        for row_j in range(row_i, N_VARS):
            covariance_pair[width, unroll_factor](
                values_address, row_i, row_j, output_address
            )
            var output = Pointer[mut=True, Float64, MutAnyOrigin](
                unsafe_from_address=output_address
            )
            output[unsafe_offset=row_j * N_VARS + row_i] = output[
                unsafe_offset=row_i * N_VARS + row_j
            ]


def outputs_match(left_address: Int, right_address: Int) -> Bool:
    var left = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=left_address
    )
    var right = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=right_address
    )
    for index in range(N_VARS * N_VARS):
        var left_value = left[unsafe_offset=index]
        var right_value = right[unsafe_offset=index]
        if isnan(left_value) and isnan(right_value):
            continue
        var difference = left_value - right_value
        if difference < 0.0:
            difference = -difference
        if difference > 1.0e-8:
            return False
    return True


def run_variant[width: Int, unroll_factor: Int](
    values: Int, output: Int, reference: Int
):
    for _ in range(WARMUPS):
        matrix_variant[width, unroll_factor](values, output)

    var started = perf_counter_ns()
    for _ in range(REPEATS):
        matrix_variant[width, unroll_factor](values, output)
    var elapsed = perf_counter_ns() - started
    print(
        "vectorize width=", width,
        " unroll=", unroll_factor,
        " average_ns=", Float64(elapsed) / Float64(REPEATS),
        " match=", outputs_match(reference, output),
    )


def main() raises:
    var values_storage = alloc(Layout[Float64](count=N_VARS * N_OBS))
    var reference_storage = alloc(Layout[Float64](count=N_VARS * N_VARS))
    var output_storage = alloc(Layout[Float64](count=N_VARS * N_VARS))
    var values = Int(values_storage.unsafe_ptr())
    var reference = Int(reference_storage.unsafe_ptr())
    var output = Int(output_storage.unsafe_ptr())

    fill_inputs(values)
    matrix_variant[1, 1](values, reference)

    run_variant[1, 1](values, output, reference)
    run_variant[1, 2](values, output, reference)
    run_variant[1, 4](values, output, reference)
    run_variant[1, 8](values, output, reference)
    run_variant[1, 16](values, output, reference)
    run_variant[1, 32](values, output, reference)
    run_variant[2, 1](values, output, reference)
    run_variant[2, 2](values, output, reference)
    run_variant[2, 4](values, output, reference)
    run_variant[2, 8](values, output, reference)
    run_variant[2, 16](values, output, reference)
    run_variant[2, 32](values, output, reference)
    run_variant[4, 1](values, output, reference)
    run_variant[4, 2](values, output, reference)
    run_variant[4, 4](values, output, reference)
    run_variant[4, 8](values, output, reference)
    run_variant[4, 16](values, output, reference)
    run_variant[4, 32](values, output, reference)
    run_variant[8, 1](values, output, reference)
    run_variant[8, 2](values, output, reference)
    run_variant[8, 4](values, output, reference)
    run_variant[8, 8](values, output, reference)
    run_variant[8, 16](values, output, reference)
    run_variant[8, 32](values, output, reference)
    run_variant[16, 1](values, output, reference)
    run_variant[16, 2](values, output, reference)
    run_variant[16, 4](values, output, reference)
    run_variant[16, 8](values, output, reference)
    run_variant[16, 16](values, output, reference)
    run_variant[16, 32](values, output, reference)
    run_variant[32, 1](values, output, reference)
    run_variant[32, 2](values, output, reference)
    run_variant[32, 4](values, output, reference)
    run_variant[32, 8](values, output, reference)
    run_variant[32, 16](values, output, reference)
    run_variant[32, 32](values, output, reference)

    dealloc(values_storage^)
    dealloc(reference_storage^)
    dealloc(output_storage^)