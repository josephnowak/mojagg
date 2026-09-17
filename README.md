# mojagg

## Development workflow

Development on Windows always runs through Ubuntu WSL. Use the single wrapper
for setup, source discovery, compilation, tests, and linting:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 install
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 doctor
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test
```

The wrapper scopes normal source searches to `python/`, `src/`, and `tests/`.
See `.agents/skills/mojagg-workflow/SKILL.md` for the AI workflow.

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

### GUFunc driver

Every operation declares one fixed-arity native tuple of typed `GUTensor`
descriptors. Dtypes, read/write capabilities, and core dimensions are
compile-time specializations; there is no boxed runtime dtype dispatch. The
binding resolves symbolic core sizes, the driver broadcasts only outer
dimensions, and `GUFuncKernel.__call__` receives one prepared core per outer
position. Non-contiguous read cores use worker-local scratch, while writable
cores are validated for direct writes. `DispatchPolicy` parallelizes the outer
slice domain only after both the outer-group and input-core thresholds pass.

See the full [guvectorize driver reference](docs/guvectorize.md) for the
Numba gufunc model, `GUTensor` ownership, core-axis flattening, output layout,
broadcasting examples, native binding flow, scratch behavior, and contribution
rules.

Run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-mojo` for the native tuple and scratch-path smoke tests, and
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-python` for Python parity against numbagg.

## Performance measurement

The repository keeps two separate performance paths. The small
`benchmarks/codspeed/` suite runs on trusted pushes and pull requests through
[CodSpeed](https://codspeed.io). It covers representative reduction, groupby,
matrix, rolling, exponential, and fill paths with moderate deterministic
inputs, so it can detect regressions without running the publication matrix.

The manual comparison in `benchmarks/public_benchmark.py` is intended for an
occasional AWS run. It compares mojagg with numbagg and available pandas
adapters, records time and peak Python-tracked allocation, verifies results,
and writes a self-contained HTML dashboard. The default profile is the larger
`Public` suite; `Quick` is useful while editing the configuration:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-public --profile quick --only reduction:nansum,nanmean
```

On the publication host, the same runner is portable:

```bash
python benchmarks/public_benchmark.py --profile public --output docs/benchmarks/latest/index.html
```

For a programmatic run, instantiate `Public`, `Quick`, or `Stress` and mutate
their public case lists and function lists. Each `ReductionTest`,
`GroupByTest`, `MatrixTest`, `RollingTest`, `ExponentialTest`, and `FillTest`
owns its shape, dtype, axes, NaN settings, and family-specific parameters.
Missing or semantically unsupported adapters are shown as `N/A` in the report.

The default output is `docs/benchmarks/latest/index.html` with a companion
`results.json`. Once a manual run is complete, commit or upload those files to
that directory. A later static documentation site can embed the report from
`latest/index.html` without rerunning the expensive benchmark in CI.

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
Grouped operations expect dense, non-negative factorization labels.

## Configuration

```python
with mojagg.config(parallel_threshold=50_000, parallel_min_groups=64, threads=8):
    mojagg.group_nansum(values, labels)

mojagg.set_config(backend="cpu")  # global
# or env: MOJAGG_PARALLEL_THRESHOLD=50000 MOJAGG_THREADS=8
```

Parallel dispatch requires both thresholds: enough outer slices and enough
elements in each input core. Context manager > global > env var > tuned
defaults (benchmark-derived).

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

## Personal experience using Mojo

I really liked the syntax and the general idea behind Mojo, but it is still
hard to develop with it if you come from a Python background. The language has
several syntax and metaprogramming limitations, especially around trait
parametrization, variadic generic arguments, tuple manipulation, and
compile-time type transformations.

The original design of this project was more generic. The operation signature
would be built once, the binding would inspect it to identify the outputs, and
the kernel would be able to unpack the signature directly in `__call__`. In
an ideal version, the same generic code would build the complete execution
plan, allocate every output, bind the NumPy addresses, and pass the resulting
signature to the operation without requiring operation-specific helper
functions.

In practice, Mojo currently makes some of these patterns difficult or
impossible. Traits cannot express the parameterized variadic interfaces
needed for arbitrary operation signatures. A generic function can work with a
tuple when its element pack is explicitly available as `*Args`, but it cannot
recover that pack from an opaque associated type such as
`Operation.Signature`. For example, an `empty_signature[Operation]()` function
can return `Operation.Signature`, but the result cannot be passed to another
generic function that expects `Tuple[*Args]`, because the compiler cannot
infer the element pack from the associated type.

The language also has limited support for generic tuple transformations.
Filtering the output tensors, constructing a new output-only signature, or
forwarding an arbitrary tuple through several generic layers is only possible
when the concrete tuple types have already been exposed to the compiler. This
forced the project to construct grouped signatures explicitly and to use a
small number of helpers for the one-output, two-output, and three-output
cases. Output initialization therefore had to be supplied separately instead
of being encoded in a fully generic signature schema.

The Python boundary adds another layer of manual work. NumPy arrays cannot be
converted automatically into the borrowed tensor descriptors used by the
Mojo driver. The binding must map Mojo dtypes to NumPy dtypes, build the
planned shape, allocate the arrays, bind their memory addresses, and keep the
Python owners alive while the native call runs. These operations are possible,
but the language does not currently provide a simple reflection or ownership
abstraction that makes this boundary as generic as the original design
intended.

AI models also tend to make many mistakes when writing Mojo. This may be
related to the language being new and having fewer examples available for
training. Even after using the skills provided for Mojo development, I tried
multiple models, from Luna to K3 to Astra, and all of them required several
iterations to verify syntax and compiler behavior. This translated into
additional token usage and development time.

This is not a message saying that Mojo should not be used. It is a message
that, for many general use cases, the language still lacks important
functionality, and I would not consider it an ideal language at this point in
its development. I expect it to improve significantly in the future, and I
also expect this project to become simpler and more generic as those features
arrive. The Modular team has already explained that Mojo is still under active
development and that much of the current effort is focused on AI integration,
hardware support, and related priorities.

## License

BSD 3-Clause — same as numbagg. mojagg is and will remain 100% free and open source.

## Acknowledgments

Inspired by and API-compatible with [numbagg](https://github.com/numbagg/numbagg) (BSD-3). Group-label conventions follow [numpy-groupies](https://github.com/ml31415/numpy-groupies).
