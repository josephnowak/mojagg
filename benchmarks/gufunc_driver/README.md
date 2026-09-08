# Gufunc driver design benchmark

Decision experiment: which architecture should mojagg's `reduce_axis` driver
(the numba.guvectorize equivalent) use? All Mojo variants wrap the **same**
masked-SIMD sum kernel, so only the driver differs. f64, ~15% NaN.

Reproduce: `bash benchmarks/gufunc_driver/run.sh` (inside the pixi env / WSL).

## Variants

| variant | driver architecture |
|---|---|
| `numpy` | `np.nansum` reference |
| `numbagg` | numba gufunc: compiled outer loop, native strides, `target="parallel"` |
| `mojagg-current` | Python per-row loop + `ascontiguousarray` (hidden copy) |
| `ffi-per-row*` | current approach minus the copy: Python loop, one FFI call per row |
| `mojo-contig` | single FFI call; Mojo loops rows of contiguous 2-D, SIMD kernel |
| `mojo-contig-par` | mojo-contig + `max.algorithm.parallelize` over rows |
| `mojo-view` | NuMojo-style: heap-backed view struct per row (lower bound) |
| `mojo-strided` | single FFI call; N-D odometer over (shape, strides), no copies |
| `mojo-strided-par` | mojo-strided + parallelize over coarse chunks |

## Results (ms, x86_64, 16 cores, Mojo 1.0.0, numbagg 0.9.4, numpy 2.5.2)

| case | numpy | numbagg | mojagg-current | ffi-per-row* | mojo-contig | mojo-contig-par | mojo-view | mojo-strided | mojo-strided-par |
|---|---|---|---|---|---|---|---|---|---|
| big-rows (10000x1000) axis=1 | 81.7 | 6.18 | 102.8 | 101.8 | 7.62 | **3.59** | 30.3 | 7.95 | 3.65 |
| tiny-rows (1000000x8) axis=1 | 93.9 | 7.38 | 9379 | 9242 | 8.41 | **3.54** | 132.9 | 10.4 | 21.7 |
| strided (1000x10000) axis=0 | 80.8 | 14.0 | 142.6 | 101.1 | — | — | — | 36.9 | **12.9** |
| full (2000x5000) axis=None | 82.6 | 31.2 | 7.95 | 8.43 | **7.61** | 8.24 | 28.7 | 8.52 | 8.15 |

## Conclusions (drove the `drivers/reduce_axis` design)

1. **Per-row FFI is catastrophically slow** (~10 µs/call in validation +
   attribute lookups): 16–1270× slower than numbagg. The gufunc outer loop
   MUST live in Mojo — one FFI call per public op.
2. **Single-call Mojo driver + `parallelize` beats numbagg everywhere**:
   1.7× on big rows, 2.1× on tiny rows, 1.1× on strided axis=0, 4.1× on full
   reductions. numbagg's numba codegen is the thing to beat, and Mojo's
   explicit SIMD + thread pool beats it.
3. **NuMojo's slice-view-per-iteration pattern is rejected** for reductions:
   5–18× slower than numbagg on row-heavy cases (heap metadata per slice).
   Their NDArray metadata structs are fine as inspiration; their reduction
   loop structure is not.
4. **The odometer driver handles any axis with zero copies** (numbagg's
   ndreduce property). On contiguous input it is within ~5–40% of the
   specialized contiguous driver single-threaded; parallel closes the gap.
   A per-row `stride == 1` branch selects the SIMD kernel (SKILL.md §3.6).
5. **Parallel needs a threshold**: on tiny rows, coarse chunked parallel is
   2.9× *slower* than numbagg while fine-grained row-parallel is 2.1× faster.
   `DispatchPolicy` must gate parallelism on `outer_count * n` and prefer
   fine-grained flat-index decomposition (divmod fast-forward per task).
6. `mojagg-current`'s `ascontiguousarray` copy is visible on the strided
   case (142ms vs 101ms driver-only) — copies are banned from the real
   driver per SKILL.md §3.1.

## Follow-up: parallel dispatch investigation (2026-09-03, WSL2)

After wiring the real driver, allnan showed numbagg "winning" some
early-exit cases. Isolation probes (`probe_dispatch.py`, `probe_pools.py`,
`probe_env.py`) found the cause is environmental, not architectural:

| measurement | result |
|---|---|
| `parallelize` no-op, 16 tasks | ~600 µs/call |
| `parallelize` no-op, 1M tasks | ~610 µs/call (i.e. ~pure fixed cost) |
| numba omp gufunc, same tiny workload, same window | **66 µs … 3062 µs — bimodal!** |
| mojagg serial slope, all-NaN worst case | clean ~18 ns/row |
| mojagg serial (10k×8) all-NaN | 195 µs — vs numbagg same window 2.5–4.6 ms |
| mojagg serial (100k×8) all-NaN | 1825 µs — vs numbagg 2999 µs (1.64× faster) |

**Conclusions:**
1. Thread-pool wake latency on WSL2 is inflated ~10–100× for EVERYONE.
   numba's OpenMP pool spin-waits briefly (OMP_WAIT_POLICY), so it is fast
   when calls land inside the spin window (66 µs) and pays a full sleep/wake
   (1–3 ms) otherwise — hence the bimodal numbagg numbers. max's
   `parallelize` appears to always sleep → consistent ~600 µs.
2. Our serial driver/kernel path is NOT the problem: clean 18 ns/row slope,
   and it beat numbagg's parallel at every measured size in the same window.
3. The earlier "numbagg faster on big-rows/strided" readings were taken in
   numba's lucky spin-window while we always paid pool wake — not a fair
   comparison.
4. **All parallel-dispatch thresholds must be calibrated on native Linux**
   (CI/AWS runner), never on WSL2. Until then `parallel_threshold=2M`
   elements (config.py) is a conservative placeholder.
5. Long-term options if native-Linux dispatch is still slow: keep-alive
   spinning worker mode (opt-in only — burns CPU), or gate parallel to
   ms-scale work only.
