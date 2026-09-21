"""Test registration when numbagg is not installed."""

import os
import subprocess
import sys
from pathlib import Path


def test_register_provides_importable_numbagg_shim_without_dependency():
    project_root = Path(__file__).parents[3]
    script = """
import importlib.abc
import importlib.util
import sys


class BlockNumbagg(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numbagg":
            raise ModuleNotFoundError("No module named 'numbagg'", name="numbagg")
        return None


sys.meta_path.insert(0, BlockNumbagg())
sys.path.insert(0, r"PROJECT_ROOT/python")

import mojagg
import numpy as np

mojagg.register()
import numbagg
from numbagg import moving

assert importlib.util.find_spec("numbagg") is not None
assert numbagg.nansum is mojagg.nansum
assert moving.move_mean is mojagg.move_mean
assert numbagg.nansum(np.array([1.0, float("nan")])) == 1.0
mojagg.unregister()
assert "numbagg" not in sys.modules
""".replace("PROJECT_ROOT", str(project_root))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
