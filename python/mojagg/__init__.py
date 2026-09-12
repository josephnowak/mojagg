"""mojagg — NaN-aware aggregations, grouped reductions, rolling windows.

numbagg-compatible API backed by AOT-compiled Mojo kernels.

This package is the PUBLIC CONTRACT: function names, signatures, and NaN/NaT
semantics must match numbagg exactly. See tests/python/ (the executable spec).
"""

from mojagg.compat import is_registered, patch, register, unregister
from mojagg.config import MojaggConfig, config, get_config, set_config
from mojagg.fill import bfill, ffill
from mojagg.groupby import (
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nanstd,
    group_nansum,
    group_nansum_of_squares,
    group_nanvar,
)
from mojagg.matrix import nancorrmatrix, nancovmatrix
from mojagg.nanfuncs import (
    allnan,
    anynan,
    count,
    nanargmax,
    nanargmin,
    nancount,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanprod,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
)

__version__ = "0.1.0"

__all__ = [
    "MojaggConfig",
    "__version__",
    "config",
    "get_config",
    "set_config",
    # nanfuncs
    "allnan",
    "anynan",
    "count",
    "nanargmax",
    "nanargmin",
    "nancount",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    # fill
    "bfill",
    "ffill",
    # matrix
    "nancorrmatrix",
    "nancovmatrix",
    # groupby
    "group_nanall",
    "group_nanany",
    "group_nanargmax",
    "group_nanargmin",
    "group_nancount",
    "group_nanfirst",
    "group_nanlast",
    "group_nanmax",
    "group_nanmean",
    "group_nanmin",
    "group_nanprod",
    "group_nanstd",
    "group_nansum",
    "group_nansum_of_squares",
    "group_nanvar",
    # compat
    "is_registered",
    "patch",
    "register",
    "unregister",
]
