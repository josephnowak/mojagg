"""Compatibility and registration layer for drop-in numbagg replacement.

Allows mojagg to dynamically overwrite numbagg functions so downstream libraries
and upstream test suites execute against mojagg's Mojo kernels directly.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Generator
from contextlib import contextmanager
from functools import wraps
from importlib.machinery import ModuleSpec
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
    "move_corr",
    "move_corrmatrix",
    "move_cov",
    "move_covmatrix",
    "move_exp_nancorr",
    "move_exp_nancorrmatrix",
    "move_exp_nancount",
    "move_exp_nancov",
    "move_exp_nancovmatrix",
    "move_exp_nanmean",
    "move_exp_nanstd",
    "move_exp_nansum",
    "move_exp_nanvar",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
    "group_nancount",
    "group_nanall",
    "group_nanany",
    "group_nanargmax",
    "group_nanargmin",
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
]

_ORIGINAL_NUMBAGG: dict[str, Any] = {}
_ORIGINAL_FUNCS: dict[str, Any] = {}
_ORIGINAL_GROUPED: dict[str, Any] = {}
_ORIGINAL_MODULES: dict[str, dict[str, Any]] = {}
_ORIGINAL_XARRAY: dict[str, Any] = {}
_ORIGINAL_XARRAY_ROLLING: dict[str, tuple[Any, Any]] = {}
_SHIM_MODULES: set[str] = set()
_ORIGINAL_LISTS: dict[str, list[Any]] = {}
_COMPARISONS_BY_NAME: dict[str, Any] = {}
_IS_REGISTERED = False

_MODULE_NAMES = {
    "numbagg.moving": (
        "move_corr",
        "move_corrmatrix",
        "move_cov",
        "move_covmatrix",
        "move_mean",
        "move_std",
        "move_sum",
        "move_var",
    ),
    "numbagg.moving_exp": (
        "move_exp_nancorr",
        "move_exp_nancount",
        "move_exp_nancov",
        "move_exp_nanmean",
        "move_exp_nanstd",
        "move_exp_nansum",
        "move_exp_nanvar",
    ),
    "numbagg.moving_matrix": (
        "move_corrmatrix",
        "move_covmatrix",
        "move_exp_nancorrmatrix",
        "move_exp_nancovmatrix",
    ),
}

# xarray normally imports these operations from ``numbagg`` at call time, so
# replacing the numbagg module is sufficient.  Its nanops implementation of
# nansum is the exception: it uses ``sum_where`` directly and never looks up
# numbagg.nansum.
_XARRAY_NANOPS_DIRECT = ("nansum",)
_XARRAY_ROLLING_FUNCS = {
    "sum": "move_sum",
    "std": "move_std",
    "var": "move_var",
}


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
        except ModuleNotFoundError as exc:
            if exc.name != "numbagg":
                raise
            target = _create_numbagg_shim()
        else:
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

    # Patch family modules as well as the top-level re-exports. Consumers often
    # import moving operations directly from these modules.
    for module_name, names in _MODULE_NAMES.items():
        try:
            module = __import__(module_name, fromlist=["*"])
            originals = _ORIGINAL_MODULES.setdefault(module_name, {})
            for name in names:
                if hasattr(mojagg, name) and hasattr(module, name):
                    originals[name] = getattr(module, name)
                    setattr(module, name, getattr(mojagg, name))
        except Exception:
            pass

    # Patch numbagg.grouped for the grouped upstream tests and consumers.
    try:
        import numbagg.grouped as grouped_mod

        for name in _REGISTERED_NAMES:
            if hasattr(mojagg, name) and hasattr(grouped_mod, name):
                _ORIGINAL_GROUPED[name] = getattr(grouped_mod, name)
                setattr(grouped_mod, name, getattr(mojagg, name))
    except Exception:
        pass

    # Patch xarray paths that bypass its normal dynamic numbagg lookup.  The
    # wrapper deliberately resolves the target attribute on every call so a
    # downstream caller can still patch numbagg.nansum after registration.
    try:
        import xarray.computation.nanops as xarray_nanops

        for name in _XARRAY_NANOPS_DIRECT:
            if not hasattr(xarray_nanops, name) or not hasattr(target, name):
                continue
            original = getattr(xarray_nanops, name)
            _ORIGINAL_XARRAY[name] = original

            @wraps(original)
            def registered_nanop(*args: Any, _name=name, **kwargs: Any) -> Any:
                return getattr(target, _name)(*args, **kwargs)

            setattr(xarray_nanops, name, registered_nanop)
    except Exception:
        pass

    # Xarray creates rolling methods with the numbagg function captured in a
    # closure at module import time. Update those cached references as well.
    try:
        import xarray.computation.rolling as xarray_rolling

        for method_name, numbagg_name in _XARRAY_ROLLING_FUNCS.items():
            if not hasattr(mojagg, numbagg_name):
                continue
            method = getattr(xarray_rolling.Rolling, method_name, None)
            if method is None or method.__closure__ is None:
                continue
            for cell in method.__closure__:
                if cell.cell_contents is getattr(target, numbagg_name, None):
                    _ORIGINAL_XARRAY_ROLLING[method_name] = (
                        cell,
                        cell.cell_contents,
                    )
                    cell.cell_contents = getattr(mojagg, numbagg_name)
                    break
    except Exception:
        pass

    # Patch collection lists
    for list_name in (
        "AGGREGATION_FUNCS",
        "GROUPED_FUNCS",
        "OTHER_FUNCS",
        "MATRIX_FUNCS",
        "MOVE_FUNCS",
        "MOVE_EXP_FUNCS",
        "MOVE_MATRIX_FUNCS",
        "MOVE_EXP_MATRIX_FUNCS",
    ):
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


def _create_numbagg_shim() -> types.ModuleType:
    """Create an importable numbagg-compatible module tree without numbagg."""
    target = types.ModuleType("numbagg", "mojagg compatibility shim for numbagg")
    target.__version__ = mojagg.__version__
    target.__path__ = []
    target.__spec__ = ModuleSpec("numbagg", loader=None, is_package=True)
    sys.modules["numbagg"] = target
    _SHIM_MODULES.add("numbagg")

    for name in _REGISTERED_NAMES:
        if hasattr(mojagg, name):
            setattr(target, name, getattr(mojagg, name))

    for list_name in (
        "AGGREGATION_FUNCS",
        "GROUPED_FUNCS",
        "OTHER_FUNCS",
        "MATRIX_FUNCS",
        "MOVE_FUNCS",
        "MOVE_EXP_FUNCS",
        "MOVE_MATRIX_FUNCS",
        "MOVE_EXP_MATRIX_FUNCS",
    ):
        setattr(target, list_name, list(getattr(mojagg, list_name)))

    funcs = _create_shim_module("numbagg.funcs", _REGISTERED_NAMES)
    grouped = _create_shim_module("numbagg.grouped", _REGISTERED_NAMES)
    target.funcs = funcs
    target.grouped = grouped

    for module_name, names in _MODULE_NAMES.items():
        module = _create_shim_module(module_name, names)
        setattr(target, module_name.rsplit(".", 1)[1], module)

    return target


def _create_shim_module(module_name: str, names: tuple[str, ...] | list[str]) -> types.ModuleType:
    module = types.ModuleType(module_name, "mojagg compatibility shim for numbagg")
    module.__spec__ = ModuleSpec(module_name, loader=None)
    for name in names:
        if hasattr(mojagg, name):
            setattr(module, name, getattr(mojagg, name))
    sys.modules[module_name] = module
    _SHIM_MODULES.add(module_name)
    return module


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

    if "numbagg.grouped" in sys.modules:
        grouped_mod = sys.modules["numbagg.grouped"]
        for name, fn in _ORIGINAL_GROUPED.items():
            setattr(grouped_mod, name, fn)

    if "xarray.computation.nanops" in sys.modules:
        xarray_nanops = sys.modules["xarray.computation.nanops"]
        for name, fn in _ORIGINAL_XARRAY.items():
            setattr(xarray_nanops, name, fn)

    for cell, fn in _ORIGINAL_XARRAY_ROLLING.values():
        cell.cell_contents = fn

    for module_name, originals in _ORIGINAL_MODULES.items():
        if module_name in sys.modules:
            module = sys.modules[module_name]
            for name, fn in originals.items():
                setattr(module, name, fn)

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

    for module_name in _SHIM_MODULES:
        sys.modules.pop(module_name, None)

    _ORIGINAL_NUMBAGG.clear()
    _ORIGINAL_FUNCS.clear()
    _ORIGINAL_GROUPED.clear()
    _ORIGINAL_MODULES.clear()
    _ORIGINAL_XARRAY.clear()
    _ORIGINAL_XARRAY_ROLLING.clear()
    _SHIM_MODULES.clear()
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
