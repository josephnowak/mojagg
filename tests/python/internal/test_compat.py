"""Test registration and monkeypatching of numbagg."""

import importlib
from unittest.mock import patch

import numbagg
import numpy as np
import pytest
from numbagg import moving, moving_exp, moving_matrix

import mojagg


def test_register_and_unregister():
    # Registration was triggered in conftest
    assert mojagg.is_registered()
    assert numbagg.nansum is mojagg.nansum
    assert numbagg.funcs.nansum is mojagg.nansum
    assert numbagg.group_nansum is mojagg.group_nansum
    assert numbagg.grouped.group_nansum is mojagg.group_nansum
    assert moving.move_mean is mojagg.move_mean
    assert moving_exp.move_exp_nanmean is mojagg.move_exp_nanmean
    assert moving_matrix.move_exp_nancovmatrix is mojagg.move_exp_nancovmatrix

    # Test unregister
    mojagg.unregister()
    assert not mojagg.is_registered()
    assert numbagg.nansum is not mojagg.nansum
    assert numbagg.group_nansum is not mojagg.group_nansum
    assert numbagg.grouped.group_nansum is not mojagg.group_nansum
    assert moving.move_mean is not mojagg.move_mean
    assert moving_exp.move_exp_nanmean is not mojagg.move_exp_nanmean
    assert moving_matrix.move_exp_nancovmatrix is not mojagg.move_exp_nancovmatrix

    # Test context manager
    with mojagg.patch():
        assert mojagg.is_registered()
        assert numbagg.nansum is mojagg.nansum
        assert numbagg.group_nansum is mojagg.group_nansum
        assert moving.move_mean is mojagg.move_mean
        assert moving_exp.move_exp_nanmean is mojagg.move_exp_nanmean
        assert moving_matrix.move_exp_nancovmatrix is mojagg.move_exp_nancovmatrix

    assert not mojagg.is_registered()
    assert numbagg.nansum is not mojagg.nansum
    assert moving.move_mean is not mojagg.move_mean

    # Restore registration for subsequent tests
    mojagg.register()
    assert mojagg.is_registered()
    assert numbagg.nansum is mojagg.nansum
    assert numbagg.group_nansum is mojagg.group_nansum
    assert moving.move_mean is mojagg.move_mean


def test_xarray_sum_uses_registered_mojagg_function():
    xarray = pytest.importorskip("xarray")

    mojagg.register()
    importlib.import_module("xarray.core.nputils")

    values = xarray.DataArray([1.0, np.nan, 3.0])
    with patch("numbagg.nansum", wraps=mojagg.nansum) as registered_nansum:
        result = values.sum(skipna=True)

    registered_nansum.assert_called_once()
    np.testing.assert_equal(result.item(), 4.0)


@pytest.mark.parametrize(
    ("method", "numbagg_name"),
    [
        ("mean", "nanmean"),
        ("prod", "nanprod"),
        ("min", "nanmin"),
        ("max", "nanmax"),
        ("argmin", "nanargmin"),
        ("argmax", "nanargmax"),
    ],
)
def test_xarray_nan_reductions_use_registered_mojagg_functions(method, numbagg_name):
    xarray = pytest.importorskip("xarray")

    mojagg.register()
    values = xarray.DataArray([1.0, np.nan, 3.0])
    with patch(f"numbagg.{numbagg_name}", wraps=getattr(mojagg, numbagg_name)) as fn:
        getattr(values, method)(skipna=True)

    fn.assert_called_once()


@pytest.mark.parametrize(
    ("rolling_method", "numbagg_name"),
    [
        ("sum", "move_sum"),
        ("std", "move_std"),
        ("var", "move_var"),
    ],
)
def test_xarray_rolling_uses_registered_mojagg_functions(rolling_method, numbagg_name):
    pytest.importorskip("xarray")

    rolling = importlib.import_module("xarray.computation.rolling")
    method = getattr(rolling.Rolling, rolling_method)
    cell = next(
        cell
        for cell in method.__closure__ or ()
        if cell.cell_contents is getattr(mojagg, numbagg_name)
    )
    assert cell.cell_contents is getattr(mojagg, numbagg_name)


@pytest.mark.parametrize(
    ("rolling_method", "numbagg_name"),
    [("sum", "move_sum"), ("std", "move_std"), ("var", "move_var")],
)
def test_xarray_rolling_uses_mojagg_when_xarray_was_imported_first(rolling_method, numbagg_name):
    pytest.importorskip("xarray")

    mojagg.unregister()
    importlib.import_module("xarray.computation.rolling")
    method = getattr(
        importlib.import_module("xarray.computation.rolling").Rolling,
        rolling_method,
    )
    cell = next(
        cell
        for cell in method.__closure__ or ()
        if cell.cell_contents is getattr(mojagg, numbagg_name)
    )
    assert cell.cell_contents is getattr(mojagg, numbagg_name)
    mojagg.unregister()
    mojagg.register()


@pytest.mark.parametrize(
    ("rolling_method", "numbagg_name"),
    [
        ("mean", "move_exp_nanmean"),
        ("sum", "move_exp_nansum"),
        ("std", "move_exp_nanstd"),
        ("var", "move_exp_nanvar"),
        ("cov", "move_exp_nancov"),
        ("corr", "move_exp_nancorr"),
    ],
)
def test_xarray_rolling_exp_uses_registered_mojagg_functions(rolling_method, numbagg_name):
    xarray = pytest.importorskip("xarray")

    mojagg.register()
    values = xarray.DataArray([1.0, np.nan, 3.0], dims="x")
    rolling = values.rolling_exp(x=2)
    args = (values,) if rolling_method in {"cov", "corr"} else ()
    with patch(f"numbagg.{numbagg_name}", wraps=getattr(mojagg, numbagg_name)) as fn:
        getattr(rolling, rolling_method)(*args)

    fn.assert_called_once()
