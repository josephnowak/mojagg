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
- **Honest**: reproducible comparisons against numbagg and NumPy, with development/WSL timings clearly separated from native-hardware calibration.

## Benchmarks

### Upstream-backed verification

The test corpus is vendored byte-for-byte from
[numbagg/numbagg@c73d4661b66cbfdee69e2834adb7f08d0d12af24](https://github.com/numbagg/numbagg/commit/c73d4661b66cbfdee69e2834adb7f08d0d12af24)
under `tests/vendor/numbagg`, with the upstream licenses and SHA-256 provenance.
`python scripts/vendor_numbagg_tests.py` refreshes the pinned snapshot;
`python scripts/vendor_numbagg_tests.py --check` verifies it offline.

Parity tests use the complete upstream array corpus, including million-element
cases, and execute the upstream allnan/anynan edge-case class unchanged through
differential wrappers. Supported operations must match numbagg's values,
shapes, dtypes and exception types; missing numbagg is an error, not a skip.
`nanprod` has no standalone numbagg equivalent and is explicitly NumPy-only.

`pixi run bench-reference --save comparison.json` compares public APIs using
the same upstream inputs, with correctness checks and JIT warmup before timing.
Add `--full` for million-element matrices, or `--ops nanmean nansum` to narrow
the run. Reports include fixture commit, runtime numbagg/NumPy versions, thread
configuration, individual samples and new/reference time ratios. Smaller ratios
are faster. Integer `nanprod` uses NumPy's explicit input dtype, matching this
extension's result-dtype contract.

Development verification (2026-09-06, WSL2, numbagg 0.9.4, 16 Numba threads):
153 Python tests passed. Both 10K- and 1M-element matrices completed: 208
numbagg comparisons plus 40 explicitly labeled NumPy-only nanprod comparisons.
Performance is not uniformly faster: the full matrix included nanmean at
1.43x and nancount at 1.59x numbagg's time in their slowest cases. These are
development measurements, not native-Linux dispatch calibration.

> Numbers below are placeholders until the first AWS calibration run (`c7i.8xlarge`, pinned CPU governor). The full matrix — size × cardinality × NaN-density × dtype — is regenerated per release and committed to `benchmarks/results/`.

| Function | numpy | numbagg | mojagg | vs numbagg |
|---|---|---|---|---|
| `nansum` (1e7 f64) | 1.0× | 8× | **TBD** | TBD |
| `group_nansum` (1e7 rows, 1e4 groups) | 1.0× (groupies) | 15× | **TBD** | TBD |
| `move_mean` (1e7 f64, w=100) | — | 20× | **TBD** | TBD |

Run them: `python benchmarks/full_matrix.py` · Continuous per-PR performance tracking via [CodSpeed](https://codspeed.io).

### Reduction development

`allnan`, `nansum` and `nanmean` implement `NaNReduction1D` in
`src/mojagg/core/reduce1d.mojo`. Shared scanners handle contiguous SIMD and
scalar strides; the axis driver merges partial states and finalizes once per
output slice. Native vector width and arithmetic accumulator-chain count are
separate. Sum retains its eight-chain EVL accumulator and scalar strided sum.

`allnan` checks `isnan(values).reduce_and()` after each native-width block.
Its terminal state also stops the multi-axis odometer, never other output
slices. Compile-time capabilities erase those checks for sum and bypass input
traversal entirely for integer `allnan`. Other reductions pass `HookReduction`
explicitly to the same `reduce_axis[Op: Reduction1D]` entry point. There is no
legacy four-hook overload. `HookReduction` describes a result-valued operation;
the axis driver still owns N-D planning, slice traversal and parallelism.

`nanmean` carries a float64 sum and int64 valid count through both scanners
and multi-axis merging; it divides only at finalization. Empty/all-NaN slices
return NaN. Float32 input remains zero-copy with a float32 result, but uses
float64 arithmetic internally to match numbagg (including finite values whose
float32 sum would overflow). The facade follows the reference's safe-casting
order: bool/small integers/float16 -> float32; 32/64-bit integers -> float64.
Unsupported nonnumeric dtypes are rejected instead of coerced.

Mean's width sweep is reproducible with
`pixi run mojo run -O3 -I src benchmarks/mean_widths.mojo`. The seven-trial,
alternating 1/2/4/8-chain comparison selected one chain for float32 and four
for float64. On the development WSL host, at 100,003 elements their median
kernel times were 41.5 us and 36.5 us respectively; float64's one-chain
baseline took 50.1 us. Four float64 chains trade roughly 8 ns on 17-element
runs for faster long scans. Recalibrate on native hardware before claiming
portable speedups. Python benchmark rows compare both dtypes against NumPy
and numbagg.

The local quick public-API comparison was mixed: full 1M-element means took
about 0.56/0.52 ms (f32/f64) versus numbagg's 6.27/7.88 ms, but float64 tiny
rows and column reductions remained slower than numbagg (2.44 vs 0.99 ms and
2.32 vs 1.32 ms respectively). These WSL measurements are not a claim that
every shape is faster.

Run `pixi run test-mojo` to check actual traversal counts, including parallel
output independence, and `pixi run test` for Python parity.
`benchmarks/reduction_contract.py` times the native boundary with allocations
and reference calculations excluded:

```bash
pixi run build-ext
PYTHONPATH=python pixi run python benchmarks/reduction_contract.py --save before.json
# Preserve a copy of the built extension before editing, then rebuild.
PYTHONPATH=python pixi run python benchmarks/reduction_contract.py --baseline-library before.so --save paired.json
```

The paired mode alternates old/new libraries on identical inputs, checks equal
results and reports new/baseline time ratios (below 1 means faster). It covers
f32/f64/i32/i64, SIMD tails, strided/multi-axis slices, all-NaN scans and early
or late decisive values. Use `--filter nansum --min-time 0.1` for longer sum
checks. WSL timings are development comparisons, not native-Linux dispatch
calibration; no dispatch thresholds are changed by this refactor.

Development comparison (2026-09-06, WSL2 x86_64, Mojo 1.0.0): nine alternating
old/new samples per case. Longer sum runs used at least 100 ms per sample.
These are time ratios against the pre-refactor working-tree extension, not
speedups against NumPy/numbagg:

| Workload | New / baseline time |
|---|---|
| Sum, 28 dtype/layout/size cases | 0.884–1.049 |
| Float allnan, first value decisive, contiguous | 0.117–0.199 |
| Float allnan, first block decisive in every MULTI slice | 0.104–0.143 |
| Float allnan, first value decisive in every strided MULTI slice | 0.039–0.045 |
| Integer allnan, MULTI layouts | 0.235–0.391 |

All measured sum cases stayed within a 5% regression tolerance; the float32
MULTI case improved by about 12%. Recheck on native Linux before treating
these development timings as portable performance claims.

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

`nanvar` and `nanstd` use numbagg's default `ddof=1` and accept an integer
`ddof` keyword. `nanquantile`/`nanmedian` use a non-streaming selection path
with NumPy-compatible linear interpolation; scalar quantiles are returned as
scalars and vector quantiles occupy the leading axis.

Native reduction kernels instantiate `float64`/`float32`/`int64`/`int32`;
the facade visibly promotes numbagg-compatible small and integer inputs where
required. Reduction inputs remain zero-copy, except for documented promotion
and big-endian normalization.

## Configuration

```python
with mojagg.config(parallel_threshold=50_000, threads=8):
    mojagg.group_nansum(values, labels)

mojagg.set_config(backend="cpu")  # global
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
