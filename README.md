# mojagg

**NaN-aware aggregations, grouped reductions, and rolling windows — numbagg's API, Mojo's speed.**

```python
import mojagg

mojagg.nansum(a, axis=1)
mojagg.group_nanmean(values, labels, axis=0)
mojagg.move_mean(a, window=30, min_count=5)
```

If you know [numbagg](https://github.com/numbagg/numbagg), you already know mojagg: same functions, same signatures, same NaN semantics — reimplemented from scratch in [Mojo](https://www.modular.com/mojo) instead of numba, and pushed further.

## Why mojagg?

- **Drop-in**: 100% API- and semantics-compatible with numbagg. Change your import, keep your code.
- **Faster**: explicit SIMD (not compiler-hoped-for), cache-line-padded parallel reductions, branch-free NaN masking, and software-prefetched group-by scatter. No JIT warmup — kernels are AOT-compiled into the wheel.
- **Tunable**: every dispatch decision (parallel thresholds, worker counts, backend) is configurable per-call, globally, or by env var.
- **Honest**: every speedup claim below is produced by `benchmarks/full_matrix.py` on pinned hardware, against numbagg *and* numpy. Reproduce it yourself.

## Benchmarks

> Numbers below are placeholders until the first AWS calibration run (`c7i.8xlarge`, pinned CPU governor). The full matrix — size × cardinality × NaN-density × dtype — is regenerated per release and committed to `benchmarks/results/`.

| Function | numpy | numbagg | mojagg | vs numbagg |
|---|---|---|---|---|
| `nansum` (1e7 f64) | 1.0× | 8× | **TBD** | TBD |
| `group_nansum` (1e7 rows, 1e4 groups) | 1.0× (groupies) | 15× | **TBD** | TBD |
| `move_mean` (1e7 f64, w=100) | — | 20× | **TBD** | TBD |

Run them: `python benchmarks/full_matrix.py` · Continuous per-PR performance tracking via [CodSpeed](https://codspeed.io).

## Install

```bash
pip install mojagg
```

Prebuilt wheels for Linux x86_64/aarch64 and macOS arm64. Python ≥ 3.11, NumPy ≥ 2.0. No Mojo toolchain needed — kernels ship compiled.

## Functions

| Family | Functions |
|---|---|
| Aggregations | `nansum nanmean nanstd nanvar nanmin nanmax nancount nanargmin nanargmax nanmedian nanquantile allnan anynan count` |
| Grouped | `group_nansum group_nanmean group_nanprod group_nanvar group_nanstd group_nancount group_nanmin group_nanmax group_nanargmin group_nanargmax group_nanfirst group_nanlast group_nanany group_nanall group_nansum_of_squares` |
| Rolling | `move_sum move_mean move_std move_var move_cov move_corr` |
| Exp-weighted | `move_exp_nansum move_exp_nanmean move_exp_nancount move_exp_nanvar move_exp_nanstd move_exp_nancov move_exp_nancorr` |
| Matrix | `nancovmatrix nancorrmatrix move_covmatrix move_corrmatrix move_exp_nancovmatrix move_exp_nancorrmatrix` |
| Fill | `ffill bfill` |

Supported dtypes: values `float64`/`float32`, labels `int64`/`int32`. Zero-copy: non-contiguous or unsupported inputs raise instead of silently copying.

## Configuration

```python
with mojagg.config(parallel_threshold=50_000, threads=8):
    mojagg.group_nansum(values, labels)

mojagg.set_config(backend="cpu")          # global
# or env: MOJAGG_PARALLEL_THRESHOLD=50000 MOJAGG_THREADS=8
```

Context manager > global > env var > tuned defaults (benchmark-derived).

## Philosophy

1. **Parity before speed.** A result that doesn't match numbagg is a bug, however fast. The test suite *is* the spec.
2. **No hidden work.** No JIT warmup, no silent casts, no secret copies. If mojagg can't go fast on your data as-is, it tells you.
3. **Measure or revert.** Performance changes land only with benchmark evidence.
4. **Design for the next hardware.** Kernels are written against a backend abstraction; GPU targets slot in without API changes.

## Built with AI, built for AI

mojagg is designed and maintained with AI agents as first-class contributors — and first-class users:

- **Machine-readable design docs** (`AGENTS.md`, `skills/`) encode the architecture, parity semantics, and performance rules, so agent-generated contributions are consistent by construction.
- **Self-verifying**: parity tests + benchmark gates give agents (and humans) objective acceptance criteria for every change.
- **Agent-friendly API**: predictable naming, explicit errors, structured config — easy for codegen tools to call correctly.

Contributions from humans and agents alike are welcome. See `AGENTS.md`.

## License

BSD 3-Clause — same as numbagg. mojagg is and will remain 100% free and open source.

## Acknowledgments

Inspired by and API-compatible with [numbagg](https://github.com/numbagg/numbagg) (BSD-3). Group-label conventions follow [numpy-groupies](https://github.com/ml31415/numpy-groupies).
