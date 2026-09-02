"""nansum kernel — NaN-skipping sum over a contiguous 1-D span.

Parity semantics (see SKILL.md §2): NaN values are skipped; an all-NaN or
empty slice sums to 0.0 (matching numpy.nansum / numbagg.nansum).

Performance patterns (SKILL.md §3): raw pointer walk, SIMD vector accumulator
with branch-free NaN masking (v != v), in-register accumulation, single
horizontal reduce at the end. No bounds checks, no allocation.
"""

from std.collections import Span
from std.sys.info import simd_width_of


@always_inline
def _simd_width[dtype: DType]() -> Int:
    """Hardware SIMD lanes for this dtype (f32 packs 2x f64 automatically)."""
    return simd_width_of[dtype]()


@always_inline
def nansum_core[dtype: DType](data: Span[Scalar[dtype], _]) -> Scalar[dtype]:
    """Sum non-NaN elements of a contiguous span. Branch-free NaN mask."""
    comptime W = _simd_width[dtype]()
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var acc = SIMD[dtype, W](0.0)
    var zero = SIMD[dtype, W](0.0)

    var i = 0
    while i + W <= n:
        var v = ptr.unsafe_load[width=W](i)
        # NaN mask: a value equals itself iff it is not NaN. select() picks
        # v where valid else 0, so NaNs contribute nothing — no branches.
        var is_valid = v.eq(v)
        acc += is_valid.select(v, zero)
        i += W

    var total = acc.reduce_add()

    # Scalar tail
    while i < n:
        var v = ptr[unsafe_offset=i]
        if v == v:  # not NaN
            total += v
        i += 1

    return total


def nansum_f64(data: Span[Float64, _]) -> Float64:
    return nansum_core[DType.float64](data)


def nansum_f32(data: Span[Float32, _]) -> Float32:
    return nansum_core[DType.float32](data)


@always_inline
def sum_core[dtype: DType](data: Span[Scalar[dtype], _]) -> Scalar[dtype]:
    """Plain SIMD sum for integer dtypes (no NaN concept, no masking)."""
    comptime W = simd_width_of[dtype]()
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var acc = SIMD[dtype, W](0)

    var i = 0
    while i + W <= n:
        acc += ptr.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += ptr[unsafe_offset=i]
        i += 1
    return total


def sum_i64(data: Span[Int64, _]) -> Int64:
    return sum_core[DType.int64](data)


def sum_i32(data: Span[Int32, _]) -> Int32:
    return sum_core[DType.int32](data)
