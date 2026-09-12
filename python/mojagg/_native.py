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
import os
import platform
import sys
import sysconfig
from pathlib import Path

_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _detect_cpu_target() -> str | None:
    """Detect if the host CPU supports higher microarchitecture tiers (e.g. AVX-512 / x86-64-v4).

    Can be forced or disabled via MOJAGG_CPU_DISPATCH ('v4', 'v3', 'baseline').
    """
    dispatch_override = os.environ.get("MOJAGG_CPU_DISPATCH", "").lower()
    if dispatch_override in ("v4", "x86-64-v4", "avx512"):
        return "v4"
    if dispatch_override in ("v3", "x86-64-v3", "baseline", "none"):
        return None

    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        return None

    # 1) NumPy CPU feature detection (numpy >= 2.0)
    try:
        from numpy._core import _multiarray_umath

        features = getattr(_multiarray_umath, "__cpu_features__", {})
        if features.get("X86_V4", False) or (
            features.get("AVX512F", False)
            and features.get("AVX512BW", False)
            and features.get("AVX512DQ", False)
            and features.get("AVX512VL", False)
        ):
            return "v4"
    except Exception:
        pass

    # 2) Fallback on Linux to /proc/cpuinfo flags
    if sys.platform.startswith("linux"):
        try:
            with Path("/proc/cpuinfo").open(encoding="utf-8") as f:
                content = f.read()
            flags = set(content.split())
            if {"avx512f", "avx512bw", "avx512dq", "avx512vl"}.issubset(flags):
                return "v4"
        except Exception:
            pass

    return None


def _load(name: str):
    """Import a compiled native submodule (e.g. 'nanfuncs_native').

    Supports dynamic CPU dispatch: if an AVX-512 (v4) variant is bundled and supported
    by the host CPU, it will be loaded for maximum SIMD throughput; otherwise falls back
    to the baseline (v3 / Apple Silicon / ARM) extension.
    """
    target = _detect_cpu_target()
    variants = [f"{name}_{target}", name] if target else [name]

    here = Path(__file__).resolve().parent

    for variant in variants:
        # 1) Already-importable on sys.path
        try:
            return importlib.import_module(variant)
        except ImportError:
            pass

        # 2) Locate the shared object next to this package or in src/.
        candidates = [
            here / f"{variant}{_EXT_SUFFIX}",
            here / f"{variant}.so",
            here / f"{variant}.dylib",
            here.parent.parent / "src" / f"{variant}.so",
            Path.cwd() / f"{variant}.so",
        ]
        for so in candidates:
            if so.exists():
                try:
                    spec = importlib.util.spec_from_file_location(name, so)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module
                except Exception:
                    # If an optimized variant fails (e.g. unexpected ABI or illegal instruction),
                    # allow fallback to the next (baseline) candidate.
                    if variant != name:
                        continue
                    raise

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

try:
    groupby = _load("groupby_native")
except ImportError:  # pragma: no cover
    groupby = None

# Native binding entry points (thin; see facade for axis-aware public API).
# Re-export every public binding (e.g. `nansum_f64`) at module scope so
# facade code does `from mojagg import _native; _native.nansum_f64(...)`.
for _module in (nanfuncs, groupby):
    if _module is not None:
        for _name in dir(_module):
            if not _name.startswith("_"):
                globals()[_name] = getattr(_module, _name)
del _module

if nanfuncs is None or groupby is None:  # pragma: no cover

    def __getattr__(name: str):
        raise ImportError("mojagg native extension not built. Run `pixi run build-ext`.")
