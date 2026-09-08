"""Shared numeric helpers for kernels.

Kernels need NaN/infinity sentinels without pulling heavyweight deps into hot
loops. All helpers are comptime-specialized and inline to nothing.
"""


@always_inline
def nan_or_zero[dtype: DType]() -> Scalar[dtype]:
    """Quiet NaN for float dtypes; 0 for ints (callers guarantee the int path
    is unreachable: integer kernels only reach this on empty slices, which the
    binding layer rejects first)."""
    comptime if dtype == DType.float64 or dtype == DType.float32:
        var z = Scalar[dtype](0.0)
        return z / z  # IEEE 0/0 = NaN at runtime
    else:
        return Scalar[dtype](0)


@always_inline
def pos_inf_or_max[dtype: DType]() -> Scalar[dtype]:
    """+inf for floats, MAX for ints — identity element for min-reductions."""
    comptime if dtype == DType.float64 or dtype == DType.float32:
        var z = Scalar[dtype](0.0)
        return Scalar[dtype](1.0) / z
    else:
        return Scalar[dtype].MAX


@always_inline
def neg_inf_or_min[dtype: DType]() -> Scalar[dtype]:
    """-inf for floats, MIN for ints — identity element for max-reductions."""
    comptime if dtype == DType.float64 or dtype == DType.float32:
        var z = Scalar[dtype](0.0)
        return Scalar[dtype](-1.0) / z
    else:
        return Scalar[dtype].MIN
