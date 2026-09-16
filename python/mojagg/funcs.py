"""Compatibility import surface matching :mod:`numbagg.funcs`."""

from mojagg.fill import bfill, ffill
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

__all__ = [
    "allnan",
    "anynan",
    "bfill",
    "count",
    "ffill",
    "nanargmax",
    "nanargmin",
    "nancorrmatrix",
    "nancount",
    "nancovmatrix",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
]
