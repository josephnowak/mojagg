"""Execute upstream numbagg test suite directly against registered mojagg implementations.

Runs upstream test modules directly without copying, pasting, or vendoring.
"""

from numbagg.test.test_funcs import *  # noqa: F403
from numbagg.test.test_matrix_functions import TestCorrelationCovarianceMatrices  # noqa: F401
