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
BINDINGS = ["nanfuncs_native", "groupby_native"]


def get_target_mcpu() -> str | None:
    """Determine the compilation target CPU microarchitecture.

    Aligns with NumPy's CPU optimization strategy across supported devices:
    - macOS Apple Silicon (arm64/aarch64): defaults to 'apple-m1' (enables ARMv8.5-A,
      full FP16 vector arithmetic, dot product, and advanced NEON SIMD supported by 100% of
      Apple Silicon Macs M1-M5, replacing LLVM's restricted 'generic' baseline).
    - x86_64: defaults to 'x86-64-v3' (AVX2, FMA3, BMI1, BMI2 - Haswell/Zen and newer, 2013+),
      dropping legacy pre-AVX2 hardware while maximizing modern SIMD performance.
    - Linux aarch64: defaults to standard ARMv8-A baseline ('generic') or cloud server targets.
    Can be overridden via MOJAGG_TARGET_CPU env var or --mcpu command line option.
    """
    env_target = os.environ.get("MOJAGG_TARGET_CPU")
    if env_target:
        return env_target

    machine = platform.machine().lower()
    system = platform.system().lower()

    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "apple-m1"

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


def build_binding(
    name: str,
    mojo_exe: str,
    mcpu: str | None = None,
    out_name: str | None = None,
) -> Path:
    src = BINDINGS_DIR / f"{name}.mojo"
    target_name = out_name or name
    out = OUT_DIR / f"{target_name}{EXT_SUFFIX}"
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
    all_targets = "--all-targets" in sys.argv or os.environ.get("MOJAGG_ALL_TARGETS") == "1"
    explicit_mcpu = parse_mcpu_arg()
    mcpu = explicit_mcpu or get_target_mcpu()
    exe = mojo()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    machine = platform.machine().lower()
    # Multi-target compilation for x86_64 when packaging wheels or explicitly requested:
    # builds both baseline (x86-64-v3 / AVX2) and high-performance (x86-64-v4 / AVX-512)
    # extensions for runtime CPU dispatch without sacrificing compatibility.
    should_build_multi_target = (
        (wheel or all_targets)
        and machine in ("x86_64", "amd64")
        and explicit_mcpu is None
        and os.environ.get("MOJAGG_TARGET_CPU") is None
    )

    for name in BINDINGS:
        # Clean stale shared libraries from other Python versions before building.
        for old_so in (*OUT_DIR.glob(f"{name}*.so"), *OUT_DIR.glob(f"{name}*.dylib")):
            with contextlib.suppress(OSError):
                old_so.unlink()

        out = build_binding(name, exe, mcpu=mcpu)
        print("built:", out)

        if should_build_multi_target:
            # Build AVX-512 (x86-64-v4) variant for modern CPUs (Skylake-X, Zen 4/5, etc.)
            v4_out = build_binding(name, exe, mcpu="x86-64-v4", out_name=f"{name}_v4")
            print("built (AVX-512 variant):", v4_out)

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
