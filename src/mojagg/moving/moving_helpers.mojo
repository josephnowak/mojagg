"""Shared low-level helpers for SIMD moving-window kernels."""

from std.collections import Span


@always_inline
def load_block[
    dtype: DType, width: Int
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    offset: Int,
    active: Int,
) -> SIMD[dtype, width]:
    """Load a full SIMD block or guard each lane of a partial tail."""

    var block = SIMD[dtype, width](0)
    var pointer = values.unsafe_ptr()
    if active == width:
        return pointer.unsafe_load[width=width](offset)

    comptime for lane in range(width):
        if lane < active:
            block[lane] = pointer[unsafe_offset=offset + lane]
    return block
