"""Public nanfuncs API — numbagg-compatible signatures.

Each op is ONE line of table + a thin wrapper around the shared `_reduce_axis`
driver (see `_reduce.py`). Adding a new nanfunc = add its flat bindings to the
native module, then add a table + wrapper here. No axis/promotion logic is
duplicated per op.
"""

from __future__ import annotations

import numpy as np

from mojagg import _native
from mojagg._reduce import _promote_nansum_like, _reduce_axis

_NANSUM_KERNELS = {
    np.dtype(np.float64): _native.nansum_f64_flat,
    np.dtype(np.float32): _native.nansum_f32_flat,
    np.dtype(np.int64): _native.sum_i64_flat,
    np.dtype(np.int32): _native.sum_i32_flat,
}


def nansum(arr, axis=None):
    """Sum of array elements over a given axis, treating NaN as zero.

    Matches numbagg.nansum / numpy.nansum semantics.
    """
    return _reduce_axis(arr, axis, _NANSUM_KERNELS, _promote_nansum_like, "nansum")
