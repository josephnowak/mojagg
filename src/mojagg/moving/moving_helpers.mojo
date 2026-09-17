"""Shared low-level helpers for SIMD moving-window kernels."""

from std.collections import Span
from std.math import isnan


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


@fieldwise_init
struct MaskedBlock[dtype: DType, width: Int](ImplicitlyCopyable):
    """One NaN-masked window block.

    Missing lanes carry a zero payload and a zero count, so entering blocks and
    expiring blocks combine into window deltas without extra masking. The
    all-zero block is the identity, which lets the warmup phase subtract a block
    that does not exist yet.
    """

    var native: SIMD[Self.dtype, Self.width]
    var values: SIMD[DType.float64, Self.width]
    var counts: SIMD[DType.float64, Self.width]

    @always_inline
    @staticmethod
    def zeros() -> Self:
        """Return the neutral block used before the window is full."""
        return Self(
            SIMD[Self.dtype, Self.width](0),
            SIMD[DType.float64, Self.width](0.0),
            SIMD[DType.float64, Self.width](0.0),
        )

    @always_inline
    @staticmethod
    def masked(
        block: SIMD[Self.dtype, Self.width],
        missing: SIMD[DType.bool, Self.width],
    ) -> Self:
        """Zero the lanes selected by `missing` and drop them from the count."""
        var native = missing.select(SIMD[Self.dtype, Self.width](0), block)
        return Self(
            native,
            native.cast[DType.float64](),
            missing.select(
                SIMD[DType.float64, Self.width](0.0),
                SIMD[DType.float64, Self.width](1.0),
            ),
        )

    @always_inline
    def squares(self) -> SIMD[DType.float64, Self.width]:
        """Lane squares, multiplied in the input dtype like the scalar path."""
        return (self.native * self.native).cast[DType.float64]()

    @always_inline
    def products(self, other: Self) -> SIMD[DType.float64, Self.width]:
        """Lane products of two blocks that already share one NaN mask."""
        return (self.native * other.native).cast[DType.float64]()


@always_inline
def load_masked_block[
    dtype: DType, width: Int
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    offset: Int,
    active: Int,
) -> MaskedBlock[dtype, width]:
    """Load one block and mask its NaN lanes."""

    var block = load_block[dtype, width](values, offset, active)
    return MaskedBlock[dtype, width].masked(block, isnan(block))


@always_inline
def load_masked_pair[
    dtype: DType, width: Int
](
    a_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    b_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    offset: Int,
    active: Int,
) -> Tuple[MaskedBlock[dtype, width], MaskedBlock[dtype, width]]:
    """Load two aligned blocks under the joint NaN mask of both operands."""

    var a_block = load_block[dtype, width](a_values, offset, active)
    var b_block = load_block[dtype, width](b_values, offset, active)
    var missing = isnan(a_block) | isnan(b_block)
    return (
        MaskedBlock[dtype, width].masked(a_block, missing),
        MaskedBlock[dtype, width].masked(b_block, missing),
    )


@always_inline
def load_expiring_block[
    dtype: DType, width: Int
](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
    input_offset: Int,
    window: Int,
    active: Int,
) -> MaskedBlock[dtype, width]:
    """Block leaving the window, or the neutral block while warming up."""

    if input_offset < window:
        return MaskedBlock[dtype, width].zeros()
    return load_masked_block[dtype, width](
        values, input_offset - window, active
    )


@always_inline
def load_expiring_pair[
    dtype: DType, width: Int
](
    a_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    b_values: Span[Scalar[dtype], ImmUntrackedOrigin],
    input_offset: Int,
    window: Int,
    active: Int,
) -> Tuple[MaskedBlock[dtype, width], MaskedBlock[dtype, width]]:
    """Pair leaving the window, or neutral blocks while warming up."""

    if input_offset < window:
        return (
            MaskedBlock[dtype, width].zeros(),
            MaskedBlock[dtype, width].zeros(),
        )
    return load_masked_pair[dtype, width](
        a_values, b_values, input_offset - window, active
    )
