"""Root pytest fixtures and setup for mojagg test suite."""

import mojagg

# Automatically register mojagg into numbagg so that any tests
# importing or executing numbagg operations use mojagg kernels.
mojagg.register()

pytest_plugins = ["numbagg.test.conftest"]
