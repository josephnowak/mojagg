"""Shared numeric helpers for kernels.

Kernels need NaN/infinity sentinels without pulling heavyweight deps into hot
loops. All helpers are comptime-specialized and inline to nothing.
"""


def _dtype_name[dtype: DType]() -> String:
    """Return the NumPy spelling for a dtype supported by mojagg."""

    comptime if dtype == DType.float64:
        return "float64"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unsupported"


@always_inline
def nan_or_zero[dtype: DType]() -> Scalar[dtype]:
    """Quiet NaN for float dtypes; 0 for ints (callers guarantee the int path
    is unreachable: integer kernels only reach this on empty slices, which the
    binding layer rejects first)."""
    comptime if dtype.is_floating_point():
        var z = Scalar[dtype](0.0)
        return z / z  # IEEE 0/0 = NaN at runtime
    else:
        return Scalar[dtype](0)


@always_inline
def pos_inf_or_max[dtype: DType]() -> Scalar[dtype]:
    """+inf for floats, MAX for ints — identity element for min-reductions."""
    comptime if dtype.is_floating_point():
        var z = Scalar[dtype](0.0)
        return Scalar[dtype](1.0) / z
    else:
        return Scalar[dtype].MAX


@always_inline
def neg_inf_or_min[dtype: DType]() -> Scalar[dtype]:
    """-inf for floats, MIN for ints — identity element for max-reductions."""
    comptime if dtype.is_floating_point():
        var z = Scalar[dtype](0.0)
        return Scalar[dtype](-1.0) / z
    else:
        return Scalar[dtype].MIN


@always_inline
def load_block_or_identity[
    dtype: DType,
    width: Int,
    origin: Origin[mut=False] = ImmUntrackedOrigin,
](
    pointer: Pointer[mut=False, Scalar[dtype], origin],
    i: Int,
    evl: Int,
    identity: Scalar[dtype],
) -> SIMD[dtype, width]:
    """Load a SIMD block from pointer, pre-filling lanes past evl with identity.
    """
    if evl == width:
        return pointer.unsafe_load[width=width](i)
    var block = SIMD[dtype, width](identity)
    comptime for lane in range(width):
        if lane < evl:
            block[lane] = pointer[unsafe_offset=i + lane]
    return block
