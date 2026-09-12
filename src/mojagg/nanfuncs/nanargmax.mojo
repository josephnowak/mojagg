"""NaN-aware first-occurrence argmax operation."""

from mojagg.nanfuncs.nanargmin import NanArgExtrema


comptime NanArgMax[dtype: DType] = NanArgExtrema[dtype, False]
