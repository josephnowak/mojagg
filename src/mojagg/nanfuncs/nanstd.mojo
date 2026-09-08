"""NaN-aware standard deviation.

The implementation is shared with nanvar; the compile-time flag only changes
the final square-root operation.
"""

from mojagg.nanfuncs.nanvar import NanVar


comptime NanStd = NanVar
