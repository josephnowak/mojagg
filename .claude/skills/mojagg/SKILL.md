---
name: mojagg
description: How mojagg (Mojo reimplementation of numbagg) is designed and how every function must be implemented. Consult BEFORE writing or modifying any kernel, driver, binding, or benchmark in this repo. Encodes the function catalog, the two-driver architecture, mandatory performance patterns, parity semantics, and the new-op checklist.
---

# mojagg — Implementation Skill

mojagg replicates **numbagg 100%** (same public API, same semantics) in Mojo, faster, with CPU SIMD + parallelism now and GPU backends planned. Every function is callable from Python with a numbagg-identical signature.

Dtype matrix (the ONLY supported combinations — enforced, never widened/copied):

- values: `float64`, `float32`
- labels/groups: `int64`, `int32`
- Anything else → clear Python error. No hidden casts in hot paths.

---

## 1. Architecture: five drivers, all kernels pure 1-D

Numbagg's decorators map to five Mojo gufunc-drivers in `src/mojagg/drivers/`. A kernel NEVER knows about ndim/axis/Python; drivers handle that once.

| numbagg decorator | mojagg driver | gufunc shape | used by |
|---|---|---|---|
| `ndaggregate` | `reduce_axis` | `(n)->()` | all nanfuncs |
| `groupndreduce` | `group_reduce` | `(n),(n)->(m)` | all group_* |
| `ndmove` | `rolling_axis` | `(n),(),()->(n)` | move_* |
| `ndmoveexp` | `rolling_exp_axis` | `(n),(n),()->(n)` | move_exp_* |
| `ndmovematrix` | `rolling_matrix` | `(m,n),(),()->(m,n,n)` | *_matrix |

Axis rules (replicate exactly):
- `reduce_axis`: axis=None → all axes; int/tuple → move those axes last, then iterate outer slices parallelized.
- `group_reduce`: axis=int → labels are 1-D of len `values.shape[axis]`; axis=tuple → labels broadcast on those axes; axis=None → labels shape == values shape. Output shape = outer shape + `(num_labels,)`; `num_labels` defaults to `labels.max()+1`.
- rolling drivers: `np.moveaxis(axis, -1)` equivalent; kernel sees contiguous last axis.

---

## 2. Function catalog & parity semantics (the spec)

### nanfuncs (`src/mojagg/nanfuncs/`, driver `reduce_axis`)
`nansum nanmean nanmin nanmax nanstd nanvar nancount nanargmin nanargmax nanmedian nanquantile allnan anynan count`
- `nanstd/nanvar` take `ddof=1`; denominators `count - ddof <= 0` → NaN.
- `allnan/anynan/count` accept bool+int+float.
- `nanquantile/nanmedian` are NOT streaming: selection-based; treat as special case (sort/introselect per slice).
- All-NaN slice → NaN (numbagg warns; mojagg returns NaN silently).

### groupby (`src/mojagg/groupby/`, driver `group_reduce`)
`group_nansum group_nanmean group_nanprod group_nancount group_nanmin group_nanmax group_nanargmin group_nanargmax group_nanfirst group_nanlast group_nanany group_nanall group_nanvar group_nanstd group_nansum_of_squares`
- **Labels < 0 are SKIPPED, never an error.**
- NaN values are skipped per-op as in numbagg's grouped.py.
- `group_nanvar/std`: three accumulators (sum, sumsq, count) per group; `denom = count - ddof <= 0 → NaN`.
- `group_nanmean`: count+sum per group; `count==0 → NaN`.
- `group_nanfirst/last`: need a per-group `seen` bitmap (first) or plain overwrite (last).
- `group_nanargmin/max`: track best value + flat index; groups with no valid data → NaN.
- `group_nanany/all`: write 0/1 into out; treat NaN as missing.
- int values are PROMOTED to f64 by the facade for ops numbagg marks `supports_ints=False` (nanmean/nanvar/nanstd). Do this visibly in Python, never hidden in kernels.

### rolling (`src/mojagg/rolling/`, driver `rolling_axis`)
`move_sum move_mean move_std move_var move_cov move_corr`
- Single-pass running accumulator; O(n), NOT O(n·window).
- `min_count`: mean/sum use `max(min_count,1)`; var/std `max(...,2)`; out is NaN until `count >= min_count`.
- std/var maintain `sum` and `sum_sq`: `var = (sum_sq - sum²/count)/(count-1)`.
- cov/corr keep `asum bsum prodsum (asum_sq bsum_sq) count`, counting only pairwise-valid rows.

### exp rolling (`rolling_exp_axis`)
`move_exp_nansum move_exp_nanmean move_exp_nancount move_exp_nanvar move_exp_nanstd move_exp_nancov move_exp_nancorr`
- Per-step: `decay = 1 - alpha_i`; accumulators *= decay; current value added if valid.
- var/std bias correction: `bias = 1 - sumw2/sumw²`; `var_out = var_biased / bias`.
- nansum tracks `zero_count` until first valid value.

### matrix (`rolling_matrix`)
`move_covmatrix move_corrmatrix move_exp_nancovmatrix move_exp_nancorrmatrix` + static `nancovmatrix nancorrmatrix`
- **Pairwise accumulators per (i,j)** — each pair uses only rows where BOTH series valid. This is mandatory for parity.
- Per-variable shift (first valid observation / leading-mean) for numerical stability; float64 accumulators even for float32 input.
- Clamp negative rounded variances to 0; clip correlation to [-1, 1].

### fill/transform
`ffill bfill` — `(n),()->(n)`, limit semantics per numbagg ndfill.

---

## 3. Mandatory performance patterns (non-negotiable in kernels)

1. **Zero-copy boundary**: binding takes `arr.ctypes.data` → `UnsafePtr` → `Span`. Validate dtype/ndim/C-contiguity/alignment ONCE; then never touch Python until return. Never call `ascontiguousarray` silently — reject instead.
2. **No bounds checks, no allocations, no refcounting inside kernels.** All buffers (output, per-worker partials, bitmaps) allocated once up front.
3. **SIMD**: vector accumulators in-register; masked NaN handling via `v != v` unordered compare; horizontal reduce once at the end. f32 must genuinely process 2× f64 lanes.
4. **Parallel reductions**: per-worker accumulator slices, **cache-line padded (64 B)** to avoid false sharing; single-thread tree merge.
5. **Groupby scatter**: scalar stores (inherent), but software-PREFETCH `labels[i+P]`'s output cache line while processing `i`. Partials-per-worker when parallel (see thresholds).
6. **Strided fallback**: contiguous fast path and strided path are separate comptime instantiations; the hot loop never branches on stride.
7. Every optimization must show up in the benchmark matrix or be reverted.

## 4. Config & dispatch

- Mojo side is **stateless value config**: `MojaggConfig` (plain Int fields) resolved per call, passed BY VALUE to kernels. Never read globals inside a kernel.
- `GlobalConfig` singleton holds global/env state; Python thread-local context manager overrides win at resolution time.
- Env vars read once at import: `MOJAGG_BACKEND`, `MOJAGG_THREADS`, `MOJAGG_PARALLEL_THRESHOLD`, `MOJAGG_PARALLEL_MIN_GROUPS`, `MOJAGG_GPU_MIN_BYTES`, `MOJAGG_SIMD_WIDTH`.
- Parallel heuristics live ONLY in `core/dispatch.mojo` (`DispatchPolicy`), defaults derived from benchmark sweeps, documented with data.

## 5. Adding a new op — checklist

1. Find its numbagg semantics above (or in the cloned numbagg source); write the parity test FIRST (`tests/python/test_<op>.py` vs numbagg output, incl. NaN/edge cases: empty, all-NaN, min_count boundaries, negative labels, axis variants).
2. Pick the driver; implement ONE kernel file, layout: validation → core kernel → dispatch → binding.
3. Instantiate exactly the supported dtypes via the registry (`core/registry.mojo`).
4. Wire into the Python facade with the exact numbagg signature.
5. Add benchmark row (`benchmarks/test_bench_*.py`) vs numbagg and numpy where a numpy equivalent exists.
6. Run: `pixi run lint`, `pixi run test`, `pixi run bench-quick`. All green = done.

## 6. What NOT to do

- No dtype widening/casting inside kernels. No negative-label errors (skip). No hidden allocations. No reading env/globals in kernels. No GPU code paths until `gpu/` lands — keep `Backend` an enum with only `CPU` active.
