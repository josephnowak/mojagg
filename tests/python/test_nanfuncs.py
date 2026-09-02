"""Parity tests for nanfuncs — differential vs numbagg and numpy.

Mirrors numbagg's test_funcs.py pattern: iterate adversarial arrays, sweep
every axis (including None and negative), compare values AND dtypes.

As more nanfuncs land, add them to FUNCTIONS; the sweep covers them uniformly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mojagg
from tests.python.util import arrays

try:
    import numbagg
except ImportError:  # pragma: no cover
    numbagg = None

# (mojagg_fn, numpy_ref, mojagg_name_for_array_gen, rtol)
FUNCTIONS = [
    (mojagg.nansum, np.nansum, "nansum", 1e-6),
]


@pytest.mark.parametrize("mojagg_fn,ref_fn,name,rtol", FUNCTIONS)
def test_parity_vs_numpy(mojagg_fn, ref_fn, name, rtol):
    msg = "\nfunc %s | dtype %s | shape %s | axis %s\nInput:\n%s\n"
    for arr in arrays(name):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            with np.errstate(invalid="ignore"):
                desired = ref_fn(arr, axis=axis)
                actual = mojagg_fn(arr, axis=axis)

            err = msg % (name, arr.dtype, arr.shape, axis, arr)
            actual = np.asarray(actual)
            desired = np.asarray(desired)

            if np.isfinite(arr.astype(np.float64)).sum() > 0:
                assert_allclose(
                    actual, desired, rtol=rtol, atol=1e-8, equal_nan=True, err_msg=err
                )
            else:
                assert_array_equal(actual, desired, err_msg=err)

            # dtype parity: mojagg follows NUMBAGG's promotion rules, which
            # differ from numpy for float16 (numbagg -> float32). So assert
            # against numbagg's dtype when available; otherwise only check
            # dtype for the natively-supported float32/float64 inputs.
            if numbagg is not None:
                num_dtype = np.asarray(getattr(numbagg, name)(arr, axis=axis)).dtype
                assert actual.dtype == num_dtype, (
                    f"{err}\ndtype mismatch vs numbagg {actual.dtype} vs {num_dtype}"
                )
            elif arr.dtype in (np.float32, np.float64):
                assert actual.dtype == desired.dtype, (
                    f"{err}\ndtype mismatch {actual.dtype} vs {desired.dtype}"
                )


@pytest.mark.skipif(numbagg is None, reason="numbagg not installed")
@pytest.mark.parametrize("mojagg_fn,ref_fn,name,rtol", FUNCTIONS)
def test_parity_vs_numbagg(mojagg_fn, ref_fn, name, rtol):
    num_fn = getattr(numbagg, name)
    for arr in arrays(name):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            with np.errstate(invalid="ignore"):
                desired = num_fn(arr, axis=axis)
                actual = mojagg_fn(arr, axis=axis)
            assert_allclose(
                np.asarray(actual),
                np.asarray(desired),
                rtol=rtol,
                atol=1e-8,
                equal_nan=True,
            )


def test_nansum_all_nan():
    assert mojagg.nansum(np.array([np.nan, np.nan])) == 0.0


def test_nansum_empty():
    assert mojagg.nansum(np.array([])) == 0.0


def test_nansum_unsupported_dtype_raises():
    with pytest.raises((TypeError, ValueError, Exception)):
        mojagg.nansum(np.array(["a", "b"]))
