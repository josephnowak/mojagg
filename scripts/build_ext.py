"""Build the mojagg Mojo native extension(s).

Compiles each src/mojagg/python/<name>.mojo binding into a shared library and
places it where `mojagg._native` can import it (next to the Python package).

Usage:
    pixi run build-ext            # build all bindings
    pixi run build-wheel          # build + package a wheel (future)
"""

from __future__ import annotations

import contextlib
import os
import platform
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


def get_target_mcpu() -> str | None:
    """Determine the compilation target CPU microarchitecture.

    Similar to NumPy's baseline strategy:
    - x86_64: defaults to 'x86-64-v3' (AVX2, FMA3, BMI1, BMI2 - Haswell/Zen and newer, 2013+),
      dropping legacy pre-AVX2 hardware while maximizing modern SIMD performance.
    - aarch64/arm64: defaults to standard ARMv8-A baseline.
    Can be overridden via MOJAGG_TARGET_CPU env var or --mcpu command line option.
    """
    env_target = os.environ.get("MOJAGG_TARGET_CPU")
    if env_target:
        return env_target

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86-64-v3"
    if machine in ("aarch64", "arm64"):
        return "generic"
    return None


def parse_mcpu_arg() -> str | None:
    for i, arg in enumerate(sys.argv):
        if arg == "--mcpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--mcpu="):
            return arg.split("=", 1)[1]
    return None


def mojo() -> str:
    exe = shutil.which("mojo")
    if exe:
        return exe
    # Fall back to the pixi/venv layout used in WSL.
    for cand in (ROOT / ".venv" / "bin" / "mojo", ROOT / ".pixi"):
        if cand.is_file():
            return str(cand)
    raise SystemExit("mojo toolchain not found on PATH; run inside pixi env")


def build_binding(name: str, mojo_exe: str, mcpu: str | None = None) -> Path:
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
    ]
    if mcpu:
        cmd.extend(["--mcpu", mcpu])
    cmd.extend(
        [
            "-o",
            str(out),
            str(src),
        ]
    )
    print("build:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def main() -> int:
    wheel = "--wheel" in sys.argv
    mcpu = parse_mcpu_arg() or get_target_mcpu()
    exe = mojo()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in BINDINGS:
        # Clean stale shared libraries from other Python versions before building.
        for old_so in OUT_DIR.glob(f"{name}*.so"):
            with contextlib.suppress(OSError):
                old_so.unlink()
        out = build_binding(name, exe, mcpu=mcpu)
        print("built:", out)
    if wheel:
        dist_dir = ROOT / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        wheel_cmd = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-w",
            str(dist_dir),
            str(ROOT),
        ]
        print("package wheel:", " ".join(wheel_cmd), flush=True)
        subprocess.run(wheel_cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
