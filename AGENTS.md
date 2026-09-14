# AGENTS.md — mojagg

Guidance for AI agents (and humans) working in this repo. Follow `.agents/skills/mojagg-workflow/SKILL.md` for the WSL-first workflow, and read `.claude/skills/mojagg/SKILL.md` for the full function catalog, parity semantics, and performance patterns before touching kernels.

## Canonical local workflow

All local commands go through `scripts/mojagg.ps1`. It always dispatches into Ubuntu WSL, uses the locked Pixi environment, and prevents the host Windows Python or Mojo installation from being selected accidentally.

Run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 doctor` before changing code. Use `install` once on a new WSL distro. Use `search` and `files` for source discovery; they are intentionally limited to `python/`, `src/`, and `tests/`.

## What this is

mojagg replicates [numbagg](https://github.com/numbagg/numbagg) 1:1 (API + semantics) in Mojo — NaN-aware reductions, grouped aggregations, rolling/exp-moving windows — faster via explicit SIMD, cache-aware parallelism, and (planned) GPU backends. Python-first package backed by a compiled Mojo extension.

## Commands (all through the WSL wrapper)

| Task | Command |
|---|---|
| Build extension | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 build` |
| Mojo unit tests | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-mojo` |
| Python parity tests | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-python` |
| Lint everything | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 lint` |
| Format | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 format` |
| Quick bench | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-quick` |
| Numbagg reference matrix | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-reference` |
| Full bench matrix | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-full` |

Use `--env py310`, `--env py311`, `--env py312`, `--env py313`, or `--env py314` when a specific locked Python environment is required.

## Repo layout

```
src/mojagg/            Mojo package
  core/                config (MojaggConfig, GlobalConfig), dispatch (DispatchPolicy), registry (dtype matrix), simd utils
  drivers/             reduce_axis, group_reduce, rolling_axis, rolling_exp_axis, rolling_matrix
  nanfuncs/  groupby/  rolling/  fill/   one file per op
  python/              PythonModuleBuilder bindings (one module per family)
python/mojagg/         Python facade: numbagg-identical signatures, config ctx manager, dtype validation
tests/mojo/  tests/python/    unit tests / parity tests vs numbagg
benchmarks/            pytest+codspeed benches, full_matrix.py, baselines/
scripts/               lint_mojo.py, release helpers
.github/workflows/     ci.yml, codspeed.yml, release.yml, dependabot, mojo-watch
```

## Performance rules (enforced in review)

- Kernels: zero-copy, no bounds checks, no allocs, SIMD with masked NaN handling, padded per-worker accumulators, prefetch on scatter loops. See SKILL.md §3.
- Dispatch thresholds come from benchmark data, never guesses; changes to `DispatchPolicy` require a benchmark diff in the PR.
- Never silently copy/cast user input. Unsupported dtype → raise with the supported list.

## Testing contract

- Parity = `np.testing.assert_allclose` against numbagg (equal-NaN) across: empty, all-NaN, min_count/ddof boundaries, negative labels, axis None/int/tuple, f32+f64.
- `tests/python/` is the executable spec of "100% numbagg compatibility".
- Reuse pinned tests/fixtures cloned from numbagg's GitHub repository, with
  source revision and license preserved. Compare results, shapes, dtypes and
  exception types directly against numbagg, not only NumPy. Benchmark the same
  inputs against numbagg too; clearly label operations without a numbagg equivalent.
- GPU-marked tests: `pytest -m gpu`, skipped without hardware.

## Conventions

- Public function names match numbagg exactly (`group_nansum`, `move_exp_nanvar`...). One op = one Mojo file.
- Internal APIs may change incompatibly. Prefer one current driver contract
  over compatibility overloads; public numerical semantics must still match numbagg.
- ruff for Python; `mojo format` for Mojo; `scripts/lint_mojo.py` enforces kernel conventions (naming, no prints, no allocs in hot loops).
- Conventional commits (`feat:`, `fix:`, `perf:`, ...) — the changelog is generated from them.
