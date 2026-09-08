"""Mojo kernel-convention linter.

There is no real Mojo linter yet (the compiler catches type/safety errors and
`mojo format` handles style). This script enforces PROJECT conventions that the
compiler cannot: the performance rules in .claude/skills/mojagg/SKILL.md §3.

It is intentionally simple regex-based checking. When the Mojo ecosystem ships
a proper linter, replace this.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# (pattern, message). Applied to kernel/driver files only (not bindings).
KERNEL_RULES = [
    (re.compile(r"\bprint\("), "no print() in kernels/drivers"),
    (
        re.compile(r"\bList\[[^\]]+\]\(\)"),
        "no ad-hoc List() allocation in kernels (allocate up front)",
    ),
    (re.compile(r"\bappend\("), "no append() (dynamic growth) in kernels"),
]

# Files where the kernel rules apply.
KERNEL_DIRS = ("nanfuncs", "groupby", "rolling", "fill", "drivers", "core")


def _is_kernel_file(path: Path) -> bool:
    rel = path.relative_to(SRC).as_posix()
    return any(f"/{d}/" in f"/{rel}" for d in KERNEL_DIRS)


def main() -> int:
    failures: list[str] = []
    for path in sorted(SRC.rglob("*.mojo")):
        if not _is_kernel_file(path):
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "unsafe" in line or line.strip().startswith("#"):
                # allow-list: lines already marked unsafe are intentional;
                # comments ignored
                pass
            for rule, msg in KERNEL_RULES:
                if rule.search(line):
                    failures.append(f"{path.relative_to(ROOT)}:{lineno}: {msg}: {line.strip()}")

    if failures:
        print("Mojo convention violations:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 1

    print("mojo conventions: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
