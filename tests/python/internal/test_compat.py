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
