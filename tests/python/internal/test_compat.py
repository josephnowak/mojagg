"""Test registration and monkeypatching of numbagg."""

import numbagg

import mojagg


def test_register_and_unregister():
    # Registration was triggered in conftest
    assert mojagg.is_registered()
    assert numbagg.nansum is mojagg.nansum
    assert numbagg.funcs.nansum is mojagg.nansum
    assert numbagg.group_nansum is mojagg.group_nansum
    assert numbagg.grouped.group_nansum is mojagg.group_nansum

    # Test unregister
    mojagg.unregister()
    assert not mojagg.is_registered()
    assert numbagg.nansum is not mojagg.nansum
    assert numbagg.group_nansum is not mojagg.group_nansum
    assert numbagg.grouped.group_nansum is not mojagg.group_nansum

    # Test context manager
    with mojagg.patch():
        assert mojagg.is_registered()
        assert numbagg.nansum is mojagg.nansum
        assert numbagg.group_nansum is mojagg.group_nansum

    assert not mojagg.is_registered()
    assert numbagg.nansum is not mojagg.nansum

    # Restore registration for subsequent tests
    mojagg.register()
    assert mojagg.is_registered()
    assert numbagg.nansum is mojagg.nansum
    assert numbagg.group_nansum is mojagg.group_nansum
