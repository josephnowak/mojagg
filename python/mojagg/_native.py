"""Loader for the compiled Mojo native extension.

The extension is built out-of-band by `pixi run build-ext` (mojo build
--emit shared-lib) and placed on the package path. This module locates and
imports it, re-exporting the bound functions at module scope so facade code
does `from mojagg import _native; _native.nansum_f64(...)`.

During development, `mojo.importer` can auto-compile from source instead.
"""

from __future__ import annotations

import importlib
import importlib.util
import sysconfig
from pathlib import Path

_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _load(name: str):
    """Import a compiled native submodule (e.g. 'nanfuncs_native')."""
    # 1) Already-importable on sys.path (installed wheel or PYTHONPATH=src).
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    # 2) Locate the shared object next to this package or in src/.
    here = Path(__file__).resolve().parent
    candidates = [
        here / f"{name}{_EXT_SUFFIX}",
        here / f"{name}.so",
        here.parent.parent / "src" / f"{name}.so",
        Path.cwd() / f"{name}.so",
    ]
    for so in candidates:
        if so.exists():
            spec = importlib.util.spec_from_file_location(name, so)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    raise ImportError(
        f"mojagg native extension '{name}' not found. "
        "Build it with `pixi run build-ext` (compiles src/mojagg/python/"
        f"{name}.mojo to a shared library), or install a prebuilt wheel."
    )


# Native submodule handles. Each is the compiled Mojo PythonModuleBuilder module.
try:
    nanfuncs = _load("nanfuncs_native")
except ImportError:  # pragma: no cover - allows importing mojagg before build
    nanfuncs = None

# Native binding entry points (thin; see facade for axis-aware public API).
# Re-export every public binding (e.g. `nansum_f64`) at module scope so
# facade code does `from mojagg import _native; _native.nansum_f64(...)`.
if nanfuncs is not None:
    for _name in dir(nanfuncs):
        if not _name.startswith("_"):
            globals()[_name] = getattr(nanfuncs, _name)
    del _name
else:  # pragma: no cover

    def __getattr__(name: str):
        raise ImportError("mojagg native extension not built. Run `pixi run build-ext`.")
