"""mojagg — NaN-aware aggregations, grouped reductions, rolling windows.

numbagg-compatible API backed by AOT-compiled Mojo kernels.

This package is the PUBLIC CONTRACT: function names, signatures, and NaN/NaT
semantics must match numbagg exactly. See tests/python/ (the executable spec).
"""

from mojagg.compat import is_registered, patch, register, unregister
from mojagg.config import MojaggConfig, config, get_config, set_config
from mojagg.funcs import (
    allnan,
    anynan,
    bfill,
    count,
    ffill,
    nanargmax,
    nanargmin,
    nancorrmatrix,
    nancount,
    nancovmatrix,
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
from mojagg.grouped import (
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
from mojagg.moving import (
    move_corr,
    move_corrmatrix,
    move_cov,
    move_covmatrix,
    move_mean,
    move_std,
    move_sum,
    move_var,
)
from mojagg.moving_exp import (
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
)
from mojagg.moving_matrix import move_exp_nancorrmatrix, move_exp_nancovmatrix

__version__ = "0.1.0"

GROUPED_FUNCS = [
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
]

MOVE_EXP_FUNCS = [
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
]

MOVE_EXP_MATRIX_FUNCS = [move_exp_nancorrmatrix, move_exp_nancovmatrix]

MOVE_FUNCS = [move_corr, move_cov, move_mean, move_std, move_sum, move_var]

MOVE_MATRIX_FUNCS = [move_corrmatrix, move_covmatrix]

AGGREGATION_FUNCS = [
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
    nanquantile,
    nanstd,
    nansum,
    nanvar,
]

MATRIX_FUNCS = [nancorrmatrix, nancovmatrix]

OTHER_FUNCS = [bfill, ffill]

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
    # moving
    "move_corr",
    "move_cov",
    "move_corrmatrix",
    "move_covmatrix",
    "move_exp_nancorr",
    "move_exp_nancount",
    "move_exp_nancov",
    "move_exp_nancorrmatrix",
    "move_exp_nancovmatrix",
    "move_exp_nanmean",
    "move_exp_nanstd",
    "move_exp_nansum",
    "move_exp_nanvar",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
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
