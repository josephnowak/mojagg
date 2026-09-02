"""mojagg — NaN-aware aggregations, grouped reductions, rolling windows.

numbagg-compatible API backed by AOT-compiled Mojo kernels.

This package is the PUBLIC CONTRACT: function names, signatures, and NaN/NaT
semantics must match numbagg exactly. See tests/python/ (the executable spec).
"""

from mojagg.config import MojaggConfig, config, get_config, set_config
from mojagg.nanfuncs import nansum

__version__ = "0.1.0"

__all__ = [
    "MojaggConfig",
    "__version__",
    "config",
    "get_config",
    "set_config",
    "nansum",
    # Function families are re-exported here as their Mojo bindings land:
    # nanfuncs: nanmean nanstd nanvar nanmin nanmax nancount ...
    # groupby:  group_nansum group_nanmean group_nanvar ...
    # rolling:  move_sum move_mean move_std move_var move_cov move_corr
    # exp:      move_exp_nansum move_exp_nanmean ...
    # fill:     ffill bfill
]
