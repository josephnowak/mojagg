"""Project linter orchestrator.

Runs ruff (Python), `mojo format --check` (Mojo, once sources exist), and the
kernel-convention rules in `lint_mojo.py`. Invoked as `pixi run lint`.

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(name: str, cmd: list[str]) -> bool:
    print(f"--- {name} ---", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode == 0


def main() -> int:
    ok = True

    ok &= _run("ruff check", ["ruff", "check", "."])
    ok &= _run("ruff format check", ["ruff", "format", "--check", "."])

    mojo_sources = list((ROOT / "src").rglob("*.mojo"))
    if mojo_sources:
        ok &= _run(
            "mojo format",
            ["mojo", "format", "-q", "src/", "tests/mojo/"],
        )
        ok &= _run("mojo conventions", [sys.executable, "scripts/lint_mojo.py"])
    else:
        print("--- mojo checks skipped (no .mojo sources yet) ---")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
