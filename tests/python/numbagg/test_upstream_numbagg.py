"""Execute upstream numbagg test suite directly against registered mojagg implementations.

Runs upstream test modules directly without copying, pasting, or vendoring.
"""

import numpy as np
import pandas as pd
import pytest
from numbagg.test.test_funcs import *  # noqa: F403
from numbagg.test.test_grouped import *  # noqa: F403
from numbagg.test.test_matrix_functions import *  # noqa: F403
from numbagg.test.test_moving import *  # noqa: F403
from numbagg.test.test_moving_exp import *  # noqa: F403


def _patch_pandas_all_nan_idx_reduction():
    """Restore the pre-pandas-3 result used by the upstream snapshot.

    The snapshot expects an all-NaN group to produce the ``-1``/NaN sentinel,
    while current pandas raises before the numbagg result can be compared.
    """
    groupby_type = pd.core.groupby.generic.SeriesGroupBy
    for name in ("idxmin", "idxmax"):
        original = getattr(groupby_type, name)

        def compatible(self, *args, _original=original, _name=name, **kwargs):
            try:
                return _original(self, *args, **kwargs)
            except ValueError as error:
                if "encountered all NA values in a group" not in str(error):
                    raise
                return self.apply(
                    lambda values: (
                        getattr(values.dropna(), _name)(*args, **kwargs)
                        if values.notna().any()
                        else np.nan
                    )
                )

        setattr(groupby_type, name, compatible)


_patch_pandas_all_nan_idx_reduction()


@pytest.fixture(scope="module")
def rs():
    """Provide the module-scoped RNG required by upstream grouped fixtures."""
    return np.random.RandomState(0)
