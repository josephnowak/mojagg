"""Compatibility and registration layer for drop-in numbagg replacement.

Allows mojagg to dynamically overwrite numbagg functions so downstream libraries
and upstream test suites execute against mojagg's Mojo kernels directly.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mojagg

_REGISTERED_NAMES = [
    "allnan",
    "anynan",
    "count",
    "nancount",
    "nansum",
    "nanmean",
    "nanvar",
    "nanstd",
    "nanmin",
    "nanmax",
    "nanargmin",
    "nanargmax",
    "nanquantile",
    "nanmedian",
    "nanprod",
    "bfill",
    "ffill",
    "nancorrmatrix",
    "nancovmatrix",
]

_ORIGINAL_NUMBAGG: dict[str, Any] = {}
_ORIGINAL_FUNCS: dict[str, Any] = {}
_ORIGINAL_LISTS: dict[str, list[Any]] = {}
_COMPARISONS_BY_NAME: dict[str, Any] = {}
_IS_REGISTERED = False


def register(target: Any = None) -> None:
    """Register mojagg functions as drop-in replacements in numbagg.

    Replaces functions in `numbagg`, `numbagg.funcs`, and test fixtures so that
    upstream numbagg tests and downstream libraries run with mojagg kernels.

    Parameters
    ----------
    target : module, optional
        Target numbagg module. If None, imports `numbagg`.
    """
    global _IS_REGISTERED
    if _IS_REGISTERED:
        return

    if target is None:
        try:
            import numbagg
        except ImportError:
            return
        target = numbagg

    # Backup and replace top-level attributes
    for name in _REGISTERED_NAMES:
        if hasattr(mojagg, name):
            mojagg_fn = getattr(mojagg, name)
            if hasattr(target, name):
                _ORIGINAL_NUMBAGG[name] = getattr(target, name)
            setattr(target, name, mojagg_fn)

    # Patch numbagg.funcs
    try:
        import numbagg.funcs as funcs_mod

        for name in _REGISTERED_NAMES:
            if hasattr(mojagg, name):
                mojagg_fn = getattr(mojagg, name)
                if hasattr(funcs_mod, name):
                    _ORIGINAL_FUNCS[name] = getattr(funcs_mod, name)
                setattr(funcs_mod, name, mojagg_fn)
    except Exception:
        pass

    # Patch collection lists
    for list_name in ("AGGREGATION_FUNCS", "OTHER_FUNCS", "MATRIX_FUNCS"):
        if hasattr(target, list_name):
            target_list = getattr(target, list_name)
            _ORIGINAL_LISTS[list_name] = list(target_list)
            target_list[:] = [getattr(target, f.__name__, f) for f in target_list]

    # Patch test comparisons
    try:
        import numbagg.test.conftest as conftest_mod

        _patch_conftest(conftest_mod)
    except Exception:
        pass

    _IS_REGISTERED = True


def _patch_conftest(conftest: Any) -> None:
    """Patch numbagg.test.conftest COMPARISONS table and module functions."""
    if not hasattr(conftest, "COMPARISONS"):
        return
    comparisons = conftest.COMPARISONS

    # Harvest all existing comparisons by name
    for k, comp in list(comparisons.items()):
        name = getattr(k, "__name__", "")
        if name and name not in _COMPARISONS_BY_NAME:
            _COMPARISONS_BY_NAME[name] = comp

    for name in _REGISTERED_NAMES:
        if hasattr(mojagg, name):
            m_fn = getattr(mojagg, name)
            if hasattr(conftest, name):
                setattr(conftest, name, m_fn)
            comp = _COMPARISONS_BY_NAME.get(name)
            if comp is not None:
                comparisons[m_fn] = comp


def unregister() -> None:
    """Restore original numbagg functions and collection lists."""
    global _IS_REGISTERED
    if not _IS_REGISTERED:
        return

    if "numbagg" in sys.modules:
        target = sys.modules["numbagg"]
        for name, fn in _ORIGINAL_NUMBAGG.items():
            setattr(target, name, fn)
        for list_name, orig_list in _ORIGINAL_LISTS.items():
            if hasattr(target, list_name):
                getattr(target, list_name)[:] = orig_list

    if "numbagg.funcs" in sys.modules:
        funcs_mod = sys.modules["numbagg.funcs"]
        for name, fn in _ORIGINAL_FUNCS.items():
            setattr(funcs_mod, name, fn)

    if "numbagg.test.conftest" in sys.modules:
        conftest = sys.modules["numbagg.test.conftest"]
        if hasattr(conftest, "COMPARISONS"):
            # Restore original function keys in comparisons
            for name, orig_fn in _ORIGINAL_NUMBAGG.items():
                comp = _COMPARISONS_BY_NAME.get(name)
                if comp is not None and orig_fn is not None:
                    conftest.COMPARISONS[orig_fn] = comp
            # Remove mojagg function keys
            for name in _REGISTERED_NAMES:
                if hasattr(mojagg, name):
                    conftest.COMPARISONS.pop(getattr(mojagg, name), None)

    _ORIGINAL_NUMBAGG.clear()
    _ORIGINAL_FUNCS.clear()
    _ORIGINAL_LISTS.clear()
    _IS_REGISTERED = False


def is_registered() -> bool:
    """Return True if mojagg is currently registered into numbagg."""
    return _IS_REGISTERED


@contextmanager
def patch(target: Any = None) -> Generator[None, None, None]:
    """Context manager to temporarily overwrite numbagg with mojagg."""
    register(target)
    try:
        yield
    finally:
        unregister()
