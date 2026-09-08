# AGENTS.md — mojagg

Guidance for AI agents (and humans) working in this repo. Read `.claude/skills/mojagg/SKILL.md` for the full function catalog, parity semantics, and performance patterns before touching kernels.

## What this is

mojagg replicates [numbagg](https://github.com/numbagg/numbagg) 1:1 (API + semantics) in Mojo — NaN-aware reductions, grouped aggregations, rolling/exp-moving windows — faster via explicit SIMD, cache-aware parallelism, and (planned) GPU backends. Python-first package backed by a compiled Mojo extension.

## Commands (all via pixi)

| Task | Command |
|---|---|
| Build extension | `pixi run build-ext` |
| Mojo unit tests | `pixi run test-mojo` |
| Python parity tests | `pixi run test` (runs pytest vs numbagg) |
| Lint everything | `pixi run lint` (ruff + mojo format --check + scripts/lint_mojo.py) |
| Format | `pixi run format` |
| Quick bench | `pixi run bench-quick` |
| Numbagg reference matrix | `pixi run bench-reference` (`--full` for 1M-element cases) |
| Full bench matrix | `python benchmarks/full_matrix.py` |

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
