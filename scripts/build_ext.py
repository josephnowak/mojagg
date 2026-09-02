"""Build the mojagg Mojo native extension(s).

Compiles each src/mojagg/python/<name>.mojo binding into a shared library and
places it where `mojagg._native` can import it (next to the Python package).

Usage:
    pixi run build-ext            # build all bindings
    pixi run build-wheel          # build + package a wheel (future)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BINDINGS_DIR = SRC / "mojagg" / "python"
OUT_DIR = ROOT / "python" / "mojagg"  # where _native.py looks first

EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# Binding modules to compile (one per family).
BINDINGS = ["nanfuncs_native"]


def mojo() -> str:
    exe = shutil.which("mojo")
    if exe:
        return exe
    # Fall back to the pixi/venv layout used in WSL.
    for cand in (ROOT / ".venv" / "bin" / "mojo", ROOT / ".pixi"):
        if cand.is_file():
            return str(cand)
    raise SystemExit("mojo toolchain not found on PATH; run inside pixi env")


def build_binding(name: str, mojo_exe: str) -> Path:
    src = BINDINGS_DIR / f"{name}.mojo"
    out = OUT_DIR / f"{name}{EXT_SUFFIX}"
    cmd = [
        mojo_exe,
        "build",
        "--emit",
        "shared-lib",
        "-I",
        str(SRC),
        "-O3",
        "-o",
        str(out),
        str(src),
    ]
    print("build:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def main() -> int:
    wheel = "--wheel" in sys.argv
    exe = mojo()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in BINDINGS:
        out = build_binding(name, exe)
        print("built:", out)
    if wheel:
        print("(wheel packaging not yet implemented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
