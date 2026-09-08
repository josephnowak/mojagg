"""NaN-aware maximum reduction."""

from mojagg.nanfuncs.nanmin import NanExtrema

comptime NanMax[
    dtype: DType,
    result_dtype: DType = dtype,
] = NanExtrema[dtype, False, result_dtype]
