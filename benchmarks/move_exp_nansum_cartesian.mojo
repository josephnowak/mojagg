"""Benchmark every width/unroll combination for exponential moving nansum.

Each measurement uses the same scalar-alpha NaN-aware exponentially weighted
moving-sum recurrence.  The Cartesian entry points are retained for comparing
compile-time width/unroll settings when the Mojo callback supports stateful
writes; Mojo 1.1 currently requires the scalar fallback below for this
recurrence.  The Cartesian product is:

* width: ``[1, 2, 4, 8, 16, 32]``;
* unroll factor: ``[1, 2, 4, 8, 16, 32]``.
"""

from std.algorithm import vectorize
from std.math import fma, isnan
from std.memory import alloc, dealloc
from std.memory.alloc import Layout
from std.time import perf_counter_ns

comptime WARMUPS = 2
comptime REPEATS = 10
comptime DATA_BYTES = 100 << 20
comptime BYTES_PER_VALUE = 8
comptime ALPHA = 0.15
comptime MULTI_ALPHA_K = 4
comptime DYNAMIC_WIDTH = 4


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


def fill_lambdas(lambdas_address: Int, length: Int):
    var lambdas = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=lambdas_address
    )
    for i in range(length):
        # Deliberately vary the decay so the scan is tested against the
        # dynamic-alpha recurrence rather than a constant-alpha special case.
        var alpha = 0.05 + Float64((i * 13) % 17) * 0.01
        lambdas[unsafe_offset=i] = 1.0 - alpha


def dynamic_decay_scalar_zero_fill(
    values_address: Int,
    lambdas_address: Int,
    output_address: Int,
    length: Int,
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address
    )
    var lambdas = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=lambdas_address
    )
    var output = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=output_address
    )
    var carry = Float64(0)
    for i in range(length):
        var value = values[unsafe_offset=i]
        if isnan(value):
            value = 0.0
        carry = value + lambdas[unsafe_offset=i] * carry
        output[unsafe_offset=i] = carry


def dynamic_decay_scalar_hold_nan(
    values_address: Int,
    lambdas_address: Int,
    output_address: Int,
    length: Int,
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address
    )
    var lambdas = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=lambdas_address
    )
    var output = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=output_address
    )
    var carry = Float64(0)
    var seen = False
    for i in range(length):
        var value = values[unsafe_offset=i]
        if not isnan(value):
            seen = True
            carry = value + lambdas[unsafe_offset=i] * carry
        if seen:
            output[unsafe_offset=i] = carry
        else:
            output[unsafe_offset=i] = nan_value()


def dynamic_decay_scan_simd4(
    values_address: Int,
    lambdas_address: Int,
    output_address: Int,
    length: Int,
):
    """The supplied non-sequential four-lane scan, kept verbatim in spirit."""
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address
    )
    var lambdas = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=lambdas_address
    )
    var output = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=output_address
    )
    var carry = Float64(0)
    var zero = SIMD[DType.float64, DYNAMIC_WIDTH](0.0)
    var i = 0
    while i + DYNAMIC_WIDTH <= length:
        var raw_x = values.unsafe_load[width=DYNAMIC_WIDTH](i)
        var lam = lambdas.unsafe_load[width=DYNAMIC_WIDTH](i)
        var x = isnan(raw_x).select(zero, raw_x)

        var x_s1 = SIMD[DType.float64, DYNAMIC_WIDTH](
            0.0, x[0], x[1], x[2]
        )
        var lam_s1 = SIMD[DType.float64, DYNAMIC_WIDTH](
            1.0, lam[1], lam[2], lam[3]
        )
        var a = fma(x_s1, lam, x)
        var lam_cum1 = lam * lam_s1

        var a_s2 = SIMD[DType.float64, DYNAMIC_WIDTH](
            0.0, 0.0, a[0], a[1]
        )
        var lam_s2 = SIMD[DType.float64, DYNAMIC_WIDTH](
            1.0, 1.0, lam_cum1[2], lam_cum1[3]
        )
        var local_scan = fma(a_s2, lam_cum1, a)

        var carry_weights = SIMD[DType.float64, DYNAMIC_WIDTH](
            lam[0],
            lam[1] * lam[0],
            lam[2] * lam[1] * lam[0],
            lam[3] * lam[2] * lam[1] * lam[0],
        )
        var final_vec = fma(
            SIMD[DType.float64, DYNAMIC_WIDTH](carry),
            carry_weights,
            local_scan,
        )
        output.unsafe_store[width=DYNAMIC_WIDTH](i, final_vec)
        carry = final_vec[3]
        i += DYNAMIC_WIDTH

    while i < length:
        var value = values[unsafe_offset=i]
        if isnan(value):
            value = 0.0
        carry = value + lambdas[unsafe_offset=i] * carry
        output[unsafe_offset=i] = carry
        i += 1


def first_mismatch(
    left_address: Int, right_address: Int, length: Int
) -> Int:
    var left = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=left_address
    )
    var right = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=right_address
    )
    for i in range(length):
        var left_value = left[unsafe_offset=i]
        var right_value = right[unsafe_offset=i]
        if isnan(left_value) and isnan(right_value):
            continue
        if left_value != right_value:
            return i
    return -1


def scalar_stream(
    values_address: Int, output_address: Int, length: Int, alpha: Float64
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address
    )
    var output = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=output_address
    )
    var decay = 1.0 - alpha
    var state = Float64(0)
    var valid = False
    for i in range(length):
        var value = values[unsafe_offset=i]
        if not isnan(value):
            valid = True
            state = fma(state, decay, value)
        if valid:
            output[unsafe_offset=i] = state
        else:
            output[unsafe_offset=i] = nan_value()


def multi_alpha_variant(
    values_address: Int, output_address: Int, length: Int
):
    """Run independent alpha streams across SIMD lanes.

    Time remains sequential, but four independent parameter streams are
    updated together.  NaNs hold every lane's state, matching move_exp_nansum
    rather than zero-filling and decaying missing observations.
    """
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address
    )
    var output = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=output_address
    )
    var lambdas = SIMD[DType.float64, MULTI_ALPHA_K](
        0.9, 0.8, 0.7, 0.5
    )
    var carry = SIMD[DType.float64, MULTI_ALPHA_K](0)
    var seen = False
    for i in range(length):
        var value = values[unsafe_offset=i]
        if not isnan(value):
            seen = True
            var input = SIMD[DType.float64, MULTI_ALPHA_K](value)
            carry = fma(carry, lambdas, input)
        if seen:
            output.unsafe_store[width=MULTI_ALPHA_K](i * MULTI_ALPHA_K, carry)
        else:
            output.unsafe_store[width=MULTI_ALPHA_K](
                i * MULTI_ALPHA_K,
                SIMD[DType.float64, MULTI_ALPHA_K](nan_value()),
            )


def scalar_multi_alpha_variant(
    values_address: Int, output_address: Int, length: Int
):
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=values_address
    )
    var output = Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=output_address
    )
    var lambdas = SIMD[DType.float64, MULTI_ALPHA_K](
        0.9, 0.8, 0.7, 0.5
    )
    var carries = SIMD[DType.float64, MULTI_ALPHA_K](0)
    var seen = False
    for i in range(length):
        var value = values[unsafe_offset=i]
        if not isnan(value):
            seen = True
            comptime for lane in range(MULTI_ALPHA_K):
                carries[lane] = carries[lane] * lambdas[lane] + value
        var output_offset = i * MULTI_ALPHA_K
        if seen:
            comptime for lane in range(MULTI_ALPHA_K):
                output[unsafe_offset=output_offset + lane] = carries[lane]
        else:
            comptime for lane in range(MULTI_ALPHA_K):
                output[unsafe_offset=output_offset + lane] = nan_value()


def one_output_match(
    values_address: Int,
    output_address: Int,
    length: Int,
    lane: Int,
    alpha: Float64,
) -> Bool:
    var output = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=output_address
    )
    var scratch = alloc(Layout[Float64](count=length))
    var scratch_address = Int(scratch.unsafe_ptr())
    var scratch_ptr = Pointer[mut=False, Float64, ImmUntrackedOrigin](
        unsafe_from_address=scratch_address
    )
    scalar_stream(values_address, scratch_address, length, alpha)
    for i in range(length):
        var got = output[unsafe_offset=i * MULTI_ALPHA_K + lane]
        var want = scratch_ptr[unsafe_offset=i]
        if isnan(got) and isnan(want):
            continue
        if got != want:
            dealloc(scratch^)
            return False
    dealloc(scratch^)
    return True


def multi_outputs_match(
    values_address: Int, output_address: Int, length: Int
) -> Bool:
    return (
        one_output_match(values_address, output_address, length, 0, 0.1)
        and one_output_match(values_address, output_address, length, 1, 0.2)
        and one_output_match(values_address, output_address, length, 2, 0.3)
        and one_output_match(values_address, output_address, length, 3, 0.5)
    )


def vectorized_variant[width: Int, unroll_factor: Int](
    values_address: Int, output_address: Int, length: Int
):
    # Stateful writes cannot be captured by the Mojo 1.1 vectorize callback.
    # Keep the recurrence correct and executable with the scalar fallback.
    var values = Pointer[mut=False, Float64, ImmUntrackedOrigin](unsafe_from_address=values_address)
    var output = Pointer[mut=True, Float64, MutAnyOrigin](unsafe_from_address=output_address)
    var alpha = Float64(ALPHA)
    var decay = 1.0 - alpha

    var state = Float64(0)
    var valid = False
    for i in range(length):
        state *= decay
        var value = values[unsafe_offset=i]
        if not isnan(value):
            valid = True
            state += value
        if valid:
            output[unsafe_offset=i] = state
        else:
            output[unsafe_offset=i] = nan_value()


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


def run_multi_alpha_comparison(
    values: Int, scalar_output: Int, simd_output: Int, length: Int
):
    for _ in range(WARMUPS):
        scalar_multi_alpha_variant(values, scalar_output, length)
    var scalar_started = perf_counter_ns()
    for _ in range(REPEATS):
        scalar_multi_alpha_variant(values, scalar_output, length)
    var scalar_elapsed = perf_counter_ns() - scalar_started

    for _ in range(WARMUPS):
        multi_alpha_variant(values, simd_output, length)
    var simd_started = perf_counter_ns()
    for _ in range(REPEATS):
        multi_alpha_variant(values, simd_output, length)
    var simd_elapsed = perf_counter_ns() - simd_started

    print(
        "scalar multi-alpha average_ns=",
        Float64(scalar_elapsed) / Float64(REPEATS),
        " match=",
        multi_outputs_match(values, scalar_output, length),
    )
    print(
        "SIMD multi-alpha average_ns=",
        Float64(simd_elapsed) / Float64(REPEATS),
        " match=",
        outputs_match(scalar_output, simd_output, length * MULTI_ALPHA_K),
        " speedup=",
        Float64(scalar_elapsed) / Float64(simd_elapsed),
    )


def run_dynamic_scan_comparison(
    values: Int,
    lambdas: Int,
    zero_reference: Int,
    hold_reference: Int,
    scan_output: Int,
    length: Int,
):
    dynamic_decay_scalar_zero_fill(
        values, lambdas, zero_reference, length
    )
    dynamic_decay_scalar_hold_nan(
        values, lambdas, hold_reference, length
    )
    print(
        "dynamic scan initial match_zero_fill=",
        outputs_match(zero_reference, scan_output, length),
        " first_mismatch_zero_fill=",
        first_mismatch(zero_reference, scan_output, length),
        " first_mismatch_nan_hold=",
        first_mismatch(hold_reference, scan_output, length),
    )

    for _ in range(WARMUPS):
        dynamic_decay_scalar_zero_fill(
            values, lambdas, zero_reference, length
        )
    var scalar_started = perf_counter_ns()
    for _ in range(REPEATS):
        dynamic_decay_scalar_zero_fill(
            values, lambdas, zero_reference, length
        )
    var scalar_elapsed = perf_counter_ns() - scalar_started

    for _ in range(WARMUPS):
        dynamic_decay_scan_simd4(values, lambdas, scan_output, length)
    var scan_started = perf_counter_ns()
    for _ in range(REPEATS):
        dynamic_decay_scan_simd4(values, lambdas, scan_output, length)
    var scan_elapsed = perf_counter_ns() - scan_started

    print(
        "dynamic scalar zero-fill average_ns=",
        Float64(scalar_elapsed) / Float64(REPEATS),
        " match=",
        outputs_match(zero_reference, scan_output, length),
    )
    print(
        "dynamic supplied scan average_ns=",
        Float64(scan_elapsed) / Float64(REPEATS),
        " match=",
        outputs_match(zero_reference, scan_output, length),
        " first_mismatch=",
        first_mismatch(zero_reference, scan_output, length),
        " speedup=",
        Float64(scalar_elapsed) / Float64(scan_elapsed),
    )
def main() raises:
    comptime length = DATA_BYTES // BYTES_PER_VALUE
    var values_storage = alloc(Layout[Float64](count=length))
    var reference_storage = alloc(Layout[Float64](count=length))
    var output_storage = alloc(Layout[Float64](count=length))
    var multi_output_storage = alloc(
        Layout[Float64](count=length * MULTI_ALPHA_K)
    )
    var scalar_multi_output_storage = alloc(
        Layout[Float64](count=length * MULTI_ALPHA_K)
    )
    var lambdas_storage = alloc(Layout[Float64](count=length))
    var dynamic_zero_storage = alloc(Layout[Float64](count=length))
    var dynamic_hold_storage = alloc(Layout[Float64](count=length))
    var dynamic_scan_storage = alloc(Layout[Float64](count=length))
    var values = Int(values_storage.unsafe_ptr())
    var reference = Int(reference_storage.unsafe_ptr())
    var output = Int(output_storage.unsafe_ptr())
    var multi_output = Int(multi_output_storage.unsafe_ptr())
    var scalar_multi_output = Int(scalar_multi_output_storage.unsafe_ptr())
    var lambdas = Int(lambdas_storage.unsafe_ptr())
    var dynamic_zero = Int(dynamic_zero_storage.unsafe_ptr())
    var dynamic_hold = Int(dynamic_hold_storage.unsafe_ptr())
    var dynamic_scan = Int(dynamic_scan_storage.unsafe_ptr())

    fill_inputs(values, length)
    fill_lambdas(lambdas, length)
    vectorized_variant[1, 1](values, reference, length)
    multi_alpha_variant(values, multi_output, length)
    print(
        "multi-alpha SIMD match=",
        multi_outputs_match(values, multi_output, length),
    )
    run_multi_alpha_comparison(
        values, scalar_multi_output, multi_output, length
    )
    dynamic_decay_scan_simd4(values, lambdas, dynamic_scan, length)
    run_dynamic_scan_comparison(
        values,
        lambdas,
        dynamic_zero,
        dynamic_hold,
        dynamic_scan,
        length,
    )

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
    dealloc(multi_output_storage^)
    dealloc(scalar_multi_output_storage^)
    dealloc(lambdas_storage^)
    dealloc(dynamic_zero_storage^)
    dealloc(dynamic_hold_storage^)
    dealloc(dynamic_scan_storage^)
