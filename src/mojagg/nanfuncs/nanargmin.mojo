"""NaN-aware first-occurrence argmin and argmax operations."""

from std.algorithm import vectorize
from std.collections import Span
from std.sys.info import simd_width_of

from mojagg.core.numeric import (
    load_block_or_identity,
    nan_or_zero,
    neg_inf_or_min,
    pos_inf_or_max,
)
from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def nan_arg_extreme_contiguous[
    dtype: DType,
    is_min: Bool,
](values: Span[Scalar[dtype], ImmUntrackedOrigin]) -> Int64:
    """Find the first SIMD-reduced minimum or maximum index.

    Iterates in reverse order (from end of array to start) so that earlier
    occurrences (smaller indices) encountered later in time naturally overwrite
    later occurrences via relational `<=` (min) or `>=` (max) comparisons.
    This also handles identity values (+inf/-inf) without any boundary fallback
    checks, and leaves all-NaN slices with index -1.
    """
    if len(values) == 0:
        return -1

    comptime width = simd_width_of[DType.int64]()
    comptime identity = (
        pos_inf_or_max[dtype]() if is_min else neg_inf_or_min[dtype]()
    )

    comptime pad_val = (
        nan_or_zero[dtype]() if dtype.is_floating_point() else identity
    )

    var global_best = SIMD[dtype, width](identity)
    var global_best_base_i = SIMD[DType.int64, width](-1)
    var pointer = values.unsafe_ptr()

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {imm pointer, mut global_best, mut global_best_base_i}:
        var block = load_block_or_identity[dtype, width](
            pointer, i, evl, pad_val
        )
        var is_better: SIMD[DType.bool, width]
        comptime if is_min:
            is_better = block.le(global_best)
        else:
            is_better = block.ge(global_best)

        if evl < width:
            comptime for lane in range(width):
                if lane >= evl:
                    is_better[lane] = False

        global_best = is_better.select(block, global_best)
        global_best_base_i = is_better.select(
            SIMD[DType.int64, width](Int64(i)), global_best_base_i
        )

    var n = len(values)
    var tail = n % width
    var start = n - tail

    if tail > 0:
        step[width](start, tail)

    for i in range(start - width, -1, -width):
        step[width](i, width)

    # Post-loop finalization across the `width` lanes with deferred lane addition
    var best_val: Scalar[dtype] = identity
    var best_pos: Int64 = -1

    for lane in range(width):
        if global_best_base_i[lane] >= 0:
            var abs_pos = global_best_base_i[lane] + Int64(lane)
            var is_better: Bool
            comptime if is_min:
                is_better = global_best[lane] < best_val
            else:
                is_better = global_best[lane] > best_val
            if is_better or best_pos < 0:
                best_val = global_best[lane]
                best_pos = abs_pos
            elif global_best[lane] == best_val and abs_pos < best_pos:
                best_pos = abs_pos

    return best_pos


@fieldwise_init
struct NanArgExtrema[
    dtype: DType,
    is_min: Bool,
](GUFuncKernel, ImplicitlyCopyable):
    comptime value_dtype = Self.dtype
    comptime out_dtype = DType.int64
    comptime Signature = Tuple[
        GUTensor[Self.value_dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.out_dtype, True, CoreSpec[]],
    ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        var index = nan_arg_extreme_contiguous[Self.dtype, Self.is_min](
            input.read_span()
        )
        output.write_span()[0] = index


comptime NanArgMin[dtype: DType] = NanArgExtrema[dtype, True]
