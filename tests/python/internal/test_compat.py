"""Test registration and monkeypatching of numbagg."""

import numbagg

import mojagg


def test_register_and_unregister():
    # Registration was triggered in conftest
    assert mojagg.is_registered()
    assert numbagg.nansum is mojagg.nansum
    assert numbagg.funcs.nansum is mojagg.nansum

    # Test unregister
    mojagg.unregister()
    assert not mojagg.is_registered()
    assert numbagg.nansum is not mojagg.nansum

    # Test context manager
    with mojagg.patch():
        assert mojagg.is_registered()
        assert numbagg.nansum is mojagg.nansum

    assert not mojagg.is_registered()
    assert numbagg.nansum is not mojagg.nansum

    # Restore registration for subsequent tests
    mojagg.register()
    assert mojagg.is_registered()
    assert numbagg.nansum is mojagg.nansum
