"""NaN-aware first-occurrence argmax reduction."""

from std.collections import Span
from std.math import isnan, iota
from std.sys.info import simd_width_of

from mojagg.core.numeric import neg_inf_or_min
from mojagg.core.reduce1d import Reduction1D
from mojagg.nanfuncs.nanargmin import ArgState


struct NanArgMax[dtype: DType](Reduction1D):
    comptime value_dtype = Self.dtype
    comptime out_dtype = DType.int64
    comptime State = ArgState[Self.dtype]

    @staticmethod
    def identity() -> Self.State:
        return Self.State(neg_inf_or_min[Self.dtype](), -1)

    @staticmethod
    def _accept(value: Scalar[Self.dtype]) -> Bool:
        comptime if Self.dtype.is_floating_point():
            return not isnan(value)
        else:
            return True

    @staticmethod
    def contig(data: Span[Scalar[Self.dtype], _]) -> Self.State:
        var n = len(data)
        if n == 0:
            return Self.identity()

        comptime W = simd_width_of[Self.dtype]()
        var ptr = data.unsafe_ptr()
        var global_values = SIMD[Self.dtype, W](neg_inf_or_min[Self.dtype]())
        var global_indices = SIMD[DType.int64, W](-1)
        var curr_indices = iota[DType.int64, W]()

        var vec_n = (n // W) * W
        var i = 0
        while i < vec_n:
            var curr_values = ptr.unsafe_load[width=W](i)
            var clean_values: SIMD[Self.dtype, W]
            var take: SIMD[DType.bool, W]
            comptime if Self.dtype.is_floating_point():
                var is_nan = isnan(curr_values)
                clean_values = is_nan.select(
                    SIMD[Self.dtype, W](neg_inf_or_min[Self.dtype]()),
                    curr_values,
                )
                var is_valid = ~is_nan
                take = clean_values.gt(global_values) | (
                    global_indices.eq(-1) & is_valid
                )
            else:
                clean_values = curr_values
                take = clean_values.gt(global_values) | global_indices.eq(-1)

            global_values = take.select(clean_values, global_values)
            global_indices = take.select(curr_indices, global_indices)
            curr_indices += Scalar[DType.int64](W)
            i += W

        var best_val = neg_inf_or_min[Self.dtype]()
        var best_idx = Int64(-1)

        # Collapse SIMD lanes
        var has_valid = not Bool(global_indices.eq(-1).reduce_and())
        if has_valid:
            var valid_values = global_indices.eq(-1).select(
                SIMD[Self.dtype, W](neg_inf_or_min[Self.dtype]()),
                global_values,
            )
            best_val = valid_values.reduce_max()
            var matching = valid_values.eq(
                SIMD[Self.dtype, W](best_val)
            ) & global_indices.ne(-1)
            var min_indices = matching.select(
                global_indices, Scalar[DType.int64].MAX_FINITE
            )
            best_idx = min_indices.reduce_min()

        # Check tail elements
        while i < n:
            var elem = ptr[unsafe_offset=i]
            if Self._accept(elem):
                if best_idx < 0 or elem > best_val:
                    best_val = elem
                    best_idx = Int64(i)
            i += 1

        return Self.State(best_val, best_idx)

    @staticmethod
    def strided(
        ptr: Pointer[mut=True, Scalar[Self.dtype], MutAnyOrigin],
        n: Int,
        stride: Int,
    ) -> Self.State:
        var best_val = neg_inf_or_min[Self.dtype]()
        var best_idx = Int64(-1)
        var off = 0
        for i in range(n):
            var elem = ptr[unsafe_offset=off]
            if Self._accept(elem):
                if best_idx < 0 or elem > best_val:
                    best_val = elem
                    best_idx = Int64(i)
            off += stride
        return Self.State(best_val, best_idx)

    @staticmethod
    def _merge(acc: Self.State, partial: Self.State, offset: Int) -> Self.State:
        if partial.index < 0:
            return acc.copy()
        if acc.index < 0:
            return Self.State(partial.value, partial.index + Int64(offset))
        var partial_index = partial.index + Int64(offset)
        if partial.value > acc.value or (
            partial.value == acc.value and partial_index < acc.index
        ):
            return Self.State(partial.value, partial_index)
        return acc.copy()

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        return Self._merge(acc, partial, 0)

    @staticmethod
    def combine_at(
        acc: Self.State, partial: Self.State, offset: Int
    ) -> Self.State:
        return Self._merge(acc, partial, offset)

    def __init__(out self):
        pass

    def finalize(self, state: Self.State) -> Scalar[DType.int64]:
        return state.index
