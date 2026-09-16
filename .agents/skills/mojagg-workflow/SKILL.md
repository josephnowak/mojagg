---
name: mojagg-workflow
description: Use when working in the mojagg repository; it defines the WSL-first environment, scoped source discovery, build order, and verification commands.
---

# mojagg project workflow

This repository is developed through Ubuntu WSL. The Windows host Python, pytest, Mojo, and Pixi installations are not project runtimes.

## Entry point

Use the single wrapper for every local operation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 doctor
```

Run `install` on a new WSL distro. The wrapper installs Pixi when it is missing, resolves the locked `pixi.lock` environment, and runs commands inside WSL. Do not invoke `python`, `pytest`, `mojo`, or `pixi` directly from the Windows host. Do not open source files through the IDE to run commands.

Use `--env py310`, `--env py311`, `--env py312`, `--env py313`, or `--env py314` when the verification must use a specific locked Python environment. `default` selects the repository default environment.

## Source boundary

The default project scope is:

- `python/` — Python facade and public API.
- `src/` — Mojo drivers, kernels, and bindings.
- `tests/` — Mojo and Python verification.

Use the wrapper’s `files` and `search` commands for discovery. Treat `.pixi/`, `.analysis/`, `.venv/`, `.tmp/`, `benchmarks/`, `examples/`, and scratch files as generated, exploratory, or secondary material unless the task explicitly names them. Do not delete `.pixi/` or `.analysis/` to improve search results; scope the search instead.

## Change verification

- Python facade change: run `test-python`.
- Mojo kernel, driver, or binding change: run `test-mojo` and `test-python`.
- Cross-layer change: run `test`.
- Style or convention change: run `lint`.
- Performance change: run `bench-quick` and compare against the numbagg reference when applicable.

The Python test commands build the native extension before running parity tests. A missing or stale native binary is an environment failure; run `doctor` or `clean-native`, then `build` through the wrapper.

Before modifying a kernel, driver, binding, or benchmark, read `.agents/skills/mojagg/SKILL.md` (or the mirrored `.claude/skills/mojagg/SKILL.md`) and follow its parity and performance rules.

If `doctor` cannot find WSL, Pixi, Mojo, Python, or pytest, stop and repair the WSL environment with `install`. Do not fall back to a host interpreter.
