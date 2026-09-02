"""Fail if any built wheel exceeds a size budget.

Cheap guard against instantiation-matrix binary bloat (see SKILL.md §4).
Usage: python scripts/check_wheel_size.py --max-mb 20 dist/*.whl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--max-mb", type=float, default=20.0)
    args = parser.parse_args()

    limit = args.max_mb * 1024 * 1024
    ok = True
    for wheel in args.wheels:
        size = wheel.stat().st_size
        status = "ok" if size <= limit else "OVER BUDGET"
        print(f"{wheel.name}: {size / 1024 / 1024:.1f} MB ({status}, budget {args.max_mb} MB)")
        ok &= size <= limit
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
