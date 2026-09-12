# mojagg — session handoff (2026-09-02)

Everything a fresh session needs to continue this project. Read this file,
then `AGENTS.md` and `.claude/skills/mojagg/SKILL.md`, then the code.

---

## 1. Project goal

Replicate **numbagg 100%** (API + semantics) in Mojo, faster, via explicit
SIMD, cache-aware parallelism, and (later) GPU. Python-first package with a
compiled Mojo extension. Reference: https://github.com/numbagg/numbagg
(cloned at `.analysis/numbagg`, also installed in the pixi env).
NuMojo (https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) cloned
at `.analysis/NuMojo` for inspiration (see §6 for what to take/reject).

## 2. Agreed design decisions (from both sessions — settled, don't reopen)

- **Dtype matrix**: values f64/f32/i64/i32; group labels i64/i32. Nothing
  else — unsupported dtypes raise with the supported list. **No hidden
  copies/casts/widening ever** (promotions happen visibly in the Python
  facade only). No i8/f8 (upcast rejected; would need native instantiation).
  The `mojagg-full` extended-dtype wheel idea was dropped; one wheel.
- **Struct-per-op, dtype as comptime param**: `struct NanSum[dtype: DType]`,
  `struct AllNan[dtype: DType]`… — one definition, N monomorphizations, NOT
  per-dtype free functions. Backend will be a comptime param too (CPU only
  for now; GPU later).
- **Op authors write only math hooks**, never loops: `State` struct +
  `step`/`step_simd`/`result`. Loops (contiguous SIMD, strided scalar,
  odometer) are shared in `core/` + `drivers/`. (See §7.)
- **numba.guvectorize equivalent = a Mojo driver per family**
  (`drivers/gufunc.mojo` etc.) — one FFI call per public op, N-D
  iteration in compiled code, parallelized. NEVER a Python per-row loop
  (measured: 16–1270× slower than numbagg).
- **Config**: dask-style layered — thread-local context manager >
  `set_config` global > env vars > defaults; resolved per call in Python,
  passed BY VALUE to Mojo (`MojaggConfig` plain Int fields). Kernels never
  read globals. Env vars: `MOJAGG_BACKEND`, `MOJAGG_THREADS`,
  `MOJAGG_PARALLEL_THRESHOLD`, `MOJAGG_PARALLEL_MIN_GROUPS`,
  `MOJAGG_GPU_MIN_BYTES`, `MOJAGG_SIMD_WIDTH`. Parallel heuristics live only
  in `core/dispatch.mojo` (`DispatchPolicy`), thresholds from benchmark data.
- **Naming/API**: public Python names = numbagg's exactly
  (`group_nansum`, `move_exp_nanvar`, …). One op = one Mojo file.
- **License**: BSD-3-Clause (same as numbagg). Repo will be public.
- **CI**: pixi-based; ruff + `mojo format --check` + `scripts/lint_mojo.py`;
  CodSpeed per-PR (free for OSS, needs `CODSPEED_TOKEN`); regression gate vs
  committed baseline; publishable benchmark numbers from a manual AWS run
  (`c7i.8xlarge`, fixed governor/pinning, ~$1–2/release).
- **Release**: manual `workflow_dispatch` "button" → version bump + tag →
  wheel matrix → TestPyPI → PyPI via OIDC Trusted Publishing (no tokens).
  Never publish on merge. Dependabot + weekly `mojo-watch` bot (Mojo is a
  moving target). Conventional commits.

## 3. Environment / how to run (IMPORTANT — Windows quirks)

- Mojo toolchain is **linux-64 only**; the pixi env lives at
  `.pixi/envs/default` and must be used **from WSL** (`wsl -d Ubuntu`).
  `pixi`/`mojo` are NOT on Windows PATH.
- Working invocation pattern (from Windows PowerShell) — put commands in a
  bash script file with **LF endings** (CRLF breaks bash; write via
  `[IO.File]::WriteAllText` with `` `n `` joins) and run:
  `wsl -d Ubuntu -- bash /mnt/c/Users/usuario/PycharmProjects/PythonProject/mojagg/<script>.sh`
- Script must set:
  `cd /mnt/c/Users/usuario/PycharmProjects/PythonProject/mojagg`
  `export PATH="$PWD/.pixi/envs/default/bin:$PATH"`
  `export PYTHONPATH=python`   (for tests importing mojagg)
- Build: `python scripts/build_ext.py` (compiles
  `src/mojagg/python/nanfuncs_native.mojo` →
  `python/mojagg/nanfuncs_native.cpython-314-x86_64-linux-gnu.so`).
- Tests: `python -m pytest tests/python -q` (parity vs numpy AND numbagg).
- Known tool quirks: single-quoted `wsl bash -c '...'` commands with complex
  quoting get silently eaten (phantom success, no output) — use script
  files. Async shell completion sometimes swallows stdout — rerun sync or
  cat a log file. WSL file timestamps skew ~minutes from Windows.
- Mojo 1.0.0 (ed45d567), Python 3.14.7, numpy 2.5.2, numbagg 0.9.4, 16 cores.

## 4. Mojo 1.0 API facts (verified by compilation in this repo)

- `Pointer[mut=True, Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(py=arr.ctypes.data))`
  — `UnsafePointer` is deprecated (use `Pointer`); `UnsafePtr` unknown.
- Pointer arithmetic: `ptr.unsafe_offset(i)` (the `+` operator is deprecated).
- SIMD: `simd_width_of[dtype]()` from `std.sys.info`;
  `ptr.unsafe_load[width=W](i)`; `ptr[unsafe_offset=i]`; `SIMD[dtype, W]`;
  `.eq/.ne/.lt/.gt/.select/.reduce_add/.reduce_min/.reduce_max/.reduce_or`.
- `parallelize` is `from max.algorithm import parallelize` (NOT std.algorithm).
  Closures V1 (verified 2026-09-03): `def body(i: Int) {mut acc, imm x}: ...`
  with a capture list; **pass the closure as a runtime argument** —
  `parallelize(worker, num_items)` — via the
  `parallelize[FuncType: def(Int) -> None](func, num_work_items)` overload.
  `@parameter` on nested closures is forbidden (see mojo-syntax SKILL.md).
- `Array[Int, N]` (std.collections) is Mojo 1.0's fixed-length inline array
  (replacement for the old InlinedFixedVector): ctor `Array[Int, N](fill=0)`,
  len == N always, NO append/push/empty ctor — it is NOT a dynamic vector.
  `NDView`/`DimArray` use it for rank metadata.
- Struct fields CANNOT hold `Pointer[_, MutAnyOrigin]` ("struct fields cannot
  expose AnyOrigin in their type"). Store the raw `Int` address and
  reconstitute with `ptr()` accessors (NDView/ReducePlan pattern), or use
  MutUntrackedOrigin when an origin field is unavoidable.
- `out` is a reserved keyword — never name a parameter `out`.
- `Int` → `Int64` requires explicit `Int64(x)` (SIMD lane assignment too).
- PythonObject interop: `Int(py=arr.ndim)`, `Bool(py=arr.flags.c_contiguous)`,
  `String(py=arr.dtype.name)`, `Int(py=arr.shape[i])`, `arr.strides[i]`,
  `Int(py=arr.size)`, `Int(py=arr.ctypes.data)`.
- List: `List[Int](length=k, fill=0)`, `append`, `capacity=` ctor.
- `Span[Scalar[dtype], MutAnyOrigin](unsafe_ptr=ptr, length=n)`.
- Struct with pointer field must be origin-parameterized:
  `struct S[o: Origin[mut=True]]: var data: Pointer[mut=True, T, Self.o]`
  with explicit `__init__` (no auto memberwise init).
- **Mojo 1.0 MISCOMPILE (confirmed by bisection)**: vectorized `v.ne(v)` /
  `v.eq(v)` on SIMD floats **loaded from memory** (`unsafe_load`) can fold to
  all-False. Scalar `std.math.isnan(v[k])` is correct. Workaround used in
  allnan/anynan: per-lane scalar `isnan` unrolled over the vector.
  nansum's `eq().select()` pattern passes all parity tests (different
  lowering), but treat vectorized NaN compares with suspicion and keep the
  parity tests as the guardrail. Revisit on toolchain upgrade (mojo-watch).

## 5. Current code state

Committed: `29aa248 scaffold: mojagg project foundation` (docs, CI,
config.py, pixi.toml). Everything else is uncommitted work-in-progress.

**NEW ARCHITECTURE VERIFIED (2026-09-03, allnan vertical slice, 14/14 green):**
- `src/mojagg/core/tensor_view.mojo` — typed `LayoutTensor`/`RuntimeLayout` views:
  addr(Int) + ndim + `DimArray` (= `Array[Int, MAX_NDIM]`, the module-level
  `comptime MAX_NDIM = 8` shared by view/driver/bindings) sizes/strides
  (ELEMENT units); `from_numpy` validates dtype once; `ptr()` reconstitutes
  the typed pointer.
- `src/mojagg/drivers/gufunc.mojo` — the gufunc driver, SCENARIO-SPLIT
  (2026-09-03 refactor, re-verified 14/14 green): `ReducePlan.build`
  partitions dims into outer/reduced ONCE and classifies one of three
  scenarios — MERGED (reduced strides chain to 1 → single contig SIMD call),
  STRIDED (k==1 non-contiguous → strided walk), MULTI (non-chaining tuple →
  inner odometer + identity/combine). `ReducePlan.slice_base` locates a slice
  (ko==0 → 0, ko==1 → one multiply, else divmod odometer); `reduce_slice`
  dispatches to `_reduce_merged`/`_reduce_strided`/`_reduce_multi`. Parallel
  path uses a V1 closure passed as a RUNTIME arg (`parallelize(worker,
  NUM_WORKERS)`), coarse-chunked, gated by
  threshold. Comptime hook params are `def (...) thin -> T` (static struct
  methods are thin; `capturing thin` REJECTS them; bare `def` = thick
  closure). NOTE: `Bool` ≠ `Scalar[DType.bool]` in Mojo 1.0 — bool-out ops
  return `Scalar[DType.bool]` to keep the driver DType-typed.
- `src/mojagg/nanfuncs/allnan.mojo` — `AllNan[dtype]` with the 4 driver
  hooks (`contig`, `strided`, `identity`, `combine`).
- `src/mojagg/python/nanfuncs_native.mojo` — generic `allnan_binding[dtype]`
  instantiated 4× at PyInit (`m.def_function[allnan_binding[DType.float64]](
  "allnan_f64")`); `_axes_from_py` reads the facade-normalized tuple into a
  `DimArray`; `_out_addr[out_dtype]` validates the preallocated out and
  returns its raw Int address (drivers store addresses, not pointers). Old
  `*_flat` bindings remain for nansum/anynan until migrated.
- `python/mojagg/_reduce.py` — NEW `reduce_op_native` + `_resolve_axes`
  (None/int/tuple → validated stride-descending tuple; axis=() → full
  reduce per numbagg quirk; 0-d → reshape(1) view) + `_reduce_axis_native`
  (one FFI call, out preallocated, threshold from get_config()). Old
  `reduce_op`/`_reduce_axis` (per-row FFI) remain for nansum/anynan.
- Tests: parity sweep now includes tuple axes (mergeable/non-mergeable/
  reversed) + strided-view and threshold-config tests.

Perf verification + investigation (allnan vs numbagg, WSL2): after deeper
probes (benchmarks/gufunc_driver/probe_*.py + README "parallel dispatch
investigation"): the regressions were ENVIRONMENTAL. WSL2 inflates ALL
thread-pool wake costs 10–100×: numba's OpenMP pool is bimodal (66µs in its
spin window, 1–3ms after spin-down), max's `parallelize` pays a consistent
~600µs/call. mojagg's SERIAL driver is clean (~18ns/row slope) and beat
numbagg's parallel at every size in the same window (10k×8: 195µs vs
2500–4600µs; 100k×8: 1.64× faster). RULE: never tune parallel thresholds on
WSL2 — calibrate DispatchPolicy on native Linux (CI/AWS). Current
`parallel_threshold=2M` is a conservative placeholder.

## 6. Gufunc driver benchmark (decision data — `benchmarks/gufunc_driver/`)

nansum, f64, ~15% NaN. Absolute ms (lower better):

| case | numpy | numbagg | mojagg-current | ffi-per-row* | mojo-contig | mojo-contig-par | mojo-view | mojo-strided | mojo-strided-par |
|---|---|---|---|---|---|---|---|---|---|
| big-rows (10000x1000) axis=1 | 81.7 | 6.18 | 102.8 | 101.8 | 7.62 | **3.59** | 30.3 | 7.95 | 3.65 |
| tiny-rows (1000000x8) axis=1 | 93.9 | 7.38 | 9379 | 9242 | 8.41 | **3.54** | 132.9 | 10.4 | 21.7 |
| strided (1000x10000) axis=0 | 80.8 | 14.0 | 142.6 | 101.1 | — | — | — | 36.9 | **12.9** |
| full (2000x5000) axis=None | 82.6 | 31.2 | 7.95 | 8.43 | **7.61** | 8.24 | 28.7 | 8.52 | 8.15 |

Conclusions:
1. Per-row FFI ≈ 10µs/call → 16–1270× slower than numbagg. Outer loop MUST
   be compiled Mojo; one FFI call per public op; validate once.
2. Single-call Mojo driver + fine-grained `parallelize` beats numbagg:
   1.7× (big rows), 2.1× (tiny rows), 1.1× (strided axis=0), 4.1× (full).
3. NuMojo-style per-slice heap views: 5–18× slower — REJECTED for
   reductions. (Their NDArray metadata struct = OK inspiration; their numpy
   interop memcpys; zero-copy only via DLPack — maybe adopt DLPack later.)
4. Odometer strided driver handles any axis/layout with zero copies; on
   contiguous input it's within 5–40% of the specialized contig driver
   (serial), parallel closes it. `stride==1` per-slice branch picks SIMD.
5. Parallel needs a threshold (DispatchPolicy): coarse parallel on tiny
   rows is 2.9× SLOWER than numbagg; fine-grained is 2.1× faster. Gate on
   `outer_count × n`; prefer fine-grained flat-index decomposition with
   per-task divmod fast-forward.
6. `ascontiguousarray` normalization copies are banned (visible: 142ms vs
   101ms on the strided case).

Why contig wins: 1 FFI call + 1 validation; row address = one multiply;
SIMD masked-NaN kernel (~4× numba codegen, isolated by the full-reduce
case); parallelize = fine-grained work stealing, disjoint outputs. Its
limits: contiguous-last-axis only, 2-D only, reduced axis must be last →
it's a fast path, not the architecture.

Key concepts (explained in session): an **odometer** = per-dim counters
ticking like car digits with a running base offset (O(1) amortized);
**stride** = elements to jump per step on an axis (contiguous ⇔ stride==1;
covers F-order, transposes, slices, stride-0, negative — no "order" labels).

## 6b. MAX/Mojo ecosystem survey (2026-09-03) — what NOT to build/buy

- **numba.guvectorize equivalent**: does NOT exist in stdlib/MAX/NuMojo.
  MAX's reduce ops are graph-compiler-side only; NuMojo's
  `iter_along_axis` copies+allocates per step. Our `reduce_axis` driver IS
  the thing; keep it.
- **MAX graph engine** (`max.graph` + `InferenceEngine`): wrong tool for
  per-call CPU latency (execute() copies inputs to device, Buffer round
  trip). Potential fit LATER for the GPU backend: register our kernels via
  `ops.custom` + `@extensibility.register`, MEF compile caching. No rolling
  ops, no isnan op, no groupby semantics — we write custom kernels either
  way.
- **layout package** (`layout.mojoc`, verified via `mojo doc`): has
  Layout/RuntimeLayout/LayoutTensor/TileTensor + layout.math
  (sum/mean/variance). ALL require comptime rank/shape structure
  (`RuntimeLayout[l]` still needs static `Layout l`; only shape/stride
  VALUES are runtime). Our boundary receives runtime ndim → TileTensor
  would force per-(dtype×ndim) monomorphization or a flatten copy.
  REJECTED for the CPU path; revisit for GPU kernels where launch-time
  shapes are templated.
- **std.algorithm.vectorize** (`vectorize[width](size, fn)`): adopt in
  contig kernels instead of hand-written SIMD main+tail loops (validated
  idiom, same codegen, less code).
- WSL2 benchmarking is meaningless for parallel code (see §6): calibrate
  thresholds on native Linux (CodSpeed CI/AWS).

## 6c. max "experimental Tensor" + max.algorithm reductions eval (2026-09-04)

Question: adopt max's experimental Tensor / their reductions to replace
NDView + our kernels? Answer: **NO on both.** (Clone: `.analysis/max` @
93f66d1, 2026-09-03; probe: `.analysis/probes/bench_max_reduction.mojo`.)

- **"experimental Tensor" today is `max.experimental.Tensor` (Python only)**,
  `max/python/max/experimental/tensor.py`: eager API over the graph
  compiler — every op JIT-builds a graph and executes via the MAX runtime.
  Wrong layer for a compiled extension (we'd call back into Python/JIT per
  op), drags in the engine as a hard dep, and has NO NaN-aware ops
  (`grep def nan` → nothing), no groupby, no rolling, no min_count/ddof.
  The old Mojo-stdlib `experimental.tensor` is gone entirely (stdlib
  reorg → `std.*`; no such module anywhere in max history).
- **NDView stays.** Mojo-side candidates all serve other execution models:
  `LayoutTensor`/`TileTensor` (comptime-layout GPU tiles — already rejected
  §6b), `ManagedTensorSlice` (graph-compiler custom-op ABI). Nothing in max
  bridges numpy metadata zero-copy; NDView's 90 lines (Int addr + DimArray)
  exist only because Mojo 1.0 struct fields can't hold
  `Pointer[_, MutAnyOrigin]`. Nothing upstream solves that better.
- **max.algorithm reductions (sum/min/max/mean/product/variance/reduce/
  map_reduce) REJECTED**, three independent reasons:
  1. *Semantics*: numpy-style, NaN propagates (verified: `sum` returns NaN
     on 15%-NaN input); no nan* variants, no empty→0/NaN rules, no
     min_count, `mean` asserts non-empty + integer-divides ints, `variance`
     is two-pass with bare `assert`s. Cannot express numbagg parity.
  2. *Performance* (f64 AVX2, ns/elem, best-of; probes
     `bench_max_reduction.mojo` + `bench_max_inner_dim.mojo`): at n=100k
     (L2) their `sum` = 0.71–0.88, `reduce`+NaN-mask = 0.87–0.93, and even
     a DIRECT `_reduce_along_inner_dimension` call (no wrapper, correct
     arg order, verified numerically) = 0.60–0.93 — vs our NaN-aware
     `NanSum.contig` = **0.28–0.30**. The cost is INSIDE their machinery:
     per-call `sync_parallelize` dispatch (grain 32768, even for 1-D) +
     IndexList/StaticTuple plumbing, not the public wrapper. At n=10M all
     converge ~0.6 (RAM-bound). They do LESS work (no mask) and still lose
     ~3× at cache-resident sizes.
  3. *API*: generic `reduce`/`map_reduce` are callable on pinned 1.0.0 ONLY
     via the legacy `@parameter` decorator (docs: deprecated, will be
     removed) PLUS a real capture in the body (a capture-less closure is
     non-capturing → "failed to infer parameter '__origins__'"). And a
     TRAP, verified 2026-09-04: `reduce`'s fn is called as
     `(load_value, acc)` — NEW VALUE FIRST, opposite of `map_reduce`'s
     `(acc, val)` and of its own parameter names. Invisible for their
     commutative built-ins; a NaN-masking fn silently computed 37266.0
     instead of 21247500.0 (n-independent garbage) until the mask was
     applied to the FIRST arg. N-D generator forms additionally need
     comptime `reduce_dim` + dense row-major `Coord` shapes: no runtime
     axes, no arbitrary/negative strides (our MERGED/STRIDED/MULTI driver
     handles all of these).
- **Watch (not adopt)**: max/kernels HEAD has a newer monoid framework
  (`algorithm/reduce_op`: ReduceSum/Max/Min/Product, ArgMin/ArgMax, Welford,
  MinMax + `rowwise` driver, `test_arg_reduce_nan.mojo`) for graph-compiler
  kernels over `ManagedTensorSlice`. Not standalone-usable today; revisit
  when the GPU backend lands.
- **`vectorize` + fixed-width SIMD accumulator: the precise 1.0.0 contract**
  (2026-09-04, probes `probe_vectorize_user_pattern.mojo`,
  `bench_vectorize_contig.mojo`, `bench_vectorize_prealign.mojo`):
  V1 closures type-check the body ONCE with `width` unbound, so the body
  can NEVER mix a captured `SIMD[_, W]` accumulator with the loaded
  `SIMD[_, width]` value — `comptime if width == …` does NOT prune (dead
  branches still type-checked), and a second "border" accumulator can't fix
  it. Compile error, not a runtime-alignment issue. What DOES work:
  (a) width-generic body (allnan-style per-lane scalar work, or per-block
      `reduce_add` = 0.52 ns/elem) — vectorize owns main+tail;
  (b) width-IGNORING body (loads always comptime `W`) over a pre-aligned
      range `nvec = n - n % (k*W)` + hand-rolled tail — vectorize drives
      the main loop only. With 4 accs via `vectorize[4*W]` stepping this
      ties the hand loop (0.29 vs 0.29 ns/elem @100k f64; equal at 10M).
      A 1-acc version is 0.45 (FADD-latency-bound); `unroll_factor=8` does
      NOT help one accumulator (serial FADD chain, measured identical).
  nansum keeps the hand loop for now (max perf, tail is hand-written in
  every variant anyway); the §7 shared-`drive_contig` refactor remains the
  right de-duplication point.
- **Shared helper ADOPTED (user decision, 2026-09-04)**:
  `core/reduce1d.reduce1d` — ONE `vectorize[W, unroll_factor=4]` call over
  the W-aligned prefix (width-ignoring closure, single accumulator), `< W`
  remainder gathered per-lane into an identity-padded `SIMD[W]`, op's
  `finalize(acc, rem)` combines (nansum: `acc.reduce_add() +
  rem.reduce_add()`). Hooks per op: `map_simd`/`map_scalar`/`fold_simd`/
  `acc_init`/`finalize` (float/int split lives inside hooks via
  `comptime if`). nansum.mojo is now math-hooks-only. KNOWN COST: 1 acc =
  0.45 vs 4 accs = 0.28 ns/elem @100k (unroll_factor does NOT create
  independent FADD chains — measured identical with/without). Facade still
  beats numbagg (1M flat: 570 vs 4467 us). If the CodSpeed matrix flags
  nansum-class ops, restore x4 accumulators INSIDE reduce1d only (hooks
  unchanged).
- **BETTER (found later same day, bench_wide_acc_evl.mojo): single WIDE
  accumulator + evl-predicated vectorize.** `acc: SIMD[dtype, 8*W]` is
  legalized by LLVM into 8 registers = 8 independent FADD chains — one
  variable, full x4-acc speed (EVL8 = 0.276–0.279 vs S6 = 0.287–0.304
  @100k, ties at 10M). And the evl vectorize overload
  (`def step[width: Int](i: Int, evl: Int)`) EXISTS in pinned 1.0.0: one
  `vectorize[8*W](n, step)` call, no pre-alignment; full blocks get
  `evl == 8*W`, the single tail call gets `evl = remainder` and gathers
  per-lane (`acc[k] = fold_scalar(acc[k], map_scalar(x))` under
  `comptime for k / if k < evl`) — no OOB full-width load. Correct at
  n=8/13/40/100_003. This is the recommended helper shape: simplest AND
  fastest. NOT yet applied to core/reduce1d.mojo (user froze code changes
  mid-session) — apply on next session's go-ahead.

## 7. AGREED next architecture (user-approved direction, not yet written)

- `src/mojagg/core/tensor_view.mojo` — typed `LayoutTensor`/`RuntimeLayout` metadata:
  `InlineArray` sizes/strides (ELEMENT units), built ONCE per call from
  PythonObject attrs. Stack-only metadata.
- `src/mojagg/drivers/gufunc.mojo` — odometer over outer dims; per
  slice `stride==1 ? SIMD kernel : scalar strided`; serial incremental vs
  parallel (divmod fast-forward) chosen by `cfg.parallel_threshold`;
  `axis=None` collapses to one flat run; tuple axes merge when strides
  chain (`stride[i] == stride[i+1]*size[i+1]`) else inner odometer.
- Op structs expose hooks: `State`, `step` (scalar), `step_simd` (vector),
  `result`; shared `drive_contig[Op]` / `drive_strided[Op]` loops in core.
- Bindings: ONE generic `binding[dtype, Op](arr, axes, out)` instantiated
  per dtype at `PyInit` registration (`m.def_function[binding[DType.float64,
  ...]]("nansum_f64")`) — zero per-dtype code; dtype→symbol table lives in
  the Python facade. (Compile-risk point: op-as-type-param; fallback = two
  comptime fn params `kernel_contig`/`kernel_strided`.)
- Facade `_reduce.py`: keep promotion/axis-normalization/stride-sort
  (int-tuple math) + byteswap for non-native endianness; DELETE moveaxis,
  ascontiguousarray, and the per-row loop; call the binding once with
  (arr, axes, out).
- Then re-run parity tests AND `benchmarks/gufunc_driver/run.sh` through
  the real facade as acceptance.

## 8. numbagg parity semantics spec (extracted from source — the contract)

### nanfuncs (decorators: ndaggregate / ndreduce)
- `nansum`: skip NaN; all-NaN/empty → 0. f64→f64, f32→f32, i64→i64, i32→i32.
- `nanmean`: sum/count; count==0 → NaN. Floats only (ints promote to f64 in
  facade; f16→f32).
- `nanvar/nanstd(ddof=1)`: TWO-PASS (mean first — 3× faster than Welford and
  more stable); count<=ddof → NaN; var = ssq_dev/(count-ddof). Floats only.
- `nanmin/nanmax`: skip NaN; all-NaN → NaN; EMPTY RAISES ValueError
  ("zero-size array to reduction operation fmin/fmax which has no
  identity"). i32→i64, i64→i64 out (floats keep dtype).
- `nanargmin/nanargmax`: FIRST occurrence on ties (strict </>); all-NaN OR
  empty → ValueError("All-NaN slice encountered"); result int64.
- `nancount`(=`count`): count non-NaN → ALWAYS int64 out.
- `allnan`: empty → True; ints → False. `anynan`: empty → False; ints →
  False. Both bool out.
- `nanquantile/nanmedian`: f64 only; NOT streaming — NaN→max replace,
  np.partition selection + linear interpolation (numpy "linear" method);
  quantiles must be in [0,1]; NaN quantile → NaN out. Facade: scalar
  quantile squeezes; result axis moved FIRST (numpy convention).
- Axis semantics (ndaggregate): axis=None → all axes (result 0-d);
  int/tuple → those axes moved last (stride-DESCENDING sorted first:
  `_optimize_axis_order`, decorators.py:213-230) then reduced; output =
  remaining shape. Empty tuple axis → return input unchanged.
- Promotion (facade): f16→f32; bool/u8/i8/i16/u16→i32; u32→i64 (numbagg
  lacks u32 → overflow risk if unpromoted); big-endian → byteswap to native
  (ONLY allowed copy, documented).

### groupby (groupndreduce, grouped.py)
`group_nansum group_nanmean group_nanprod group_nancount group_nanmin
group_nanmax group_nanargmin group_nanargmax group_nanfirst group_nanlast
group_nanany group_nanall group_nanvar group_nanstd group_nansum_of_squares`
- **Labels < 0 are SKIPPED** (never an error).
- Labels must be int dtype; if `labels.dtype.max < values.size` numbagg
  upcasts labels (overflow guard).
- `num_labels = max(labels)+1` default. Output init per op (sum=0, prod=1,
  count=0, min/max/first/last track "seen").
- Axis modes: int → labels 1-D of len shape[axis] (moveaxis axis→-1);
  tuple → labels broadcast on those axes (moveaxis axes→trailing);
  None → labels shape == values shape. Out shape = outer + (num_labels,).
- group_nanmean: count==0 → NaN. group_nanvar/std(ddof=1): 3 accumulators
  (sum, sumsq, count); `denom = count-ddof <= 0 → NaN`;
  var = (sumsq - sum²/count)/denom.
- group_nanfirst: first non-NaN per group (seen bitmap); float groups with
  no data → NaN. group_nanlast: init NaN, overwrite on each non-NaN.
- group_nanargmin/max: best value + flat index; empty group → NaN.
- group_nanany/all: NaN treated as missing; any → 1 if any truthy valid;
  all → 0 if any falsy valid.
- supports_ints=False ops (nanmean/var/std): ints promote to f64 in facade.

### rolling (ndmove, moving.py) — float only
`move_sum move_mean move_std move_var move_cov move_corr`
- Single-pass running accumulator O(n); NaNs excluded from count/sums.
- `min_count`: None → =window; <0 → ValueError; clamped ≥1 (mean/sum/corr),
  ≥2 (std/var/cov); count<min_count → NaN. window must be 0<w<=shape[axis].
- First `window` elements = prefix-window incremental results.
- var = (sum_sq − sum²/count)/(count−1); std = sqrt(var).
- cov: pairwise-valid only; (prodsum − asum·bsum/count)/(count−1).
- corr: cov_ab/sqrt(var_a·var_b) from sums/sumsq/prodsum; variances ≤0 → NaN.
- axis: single int only (tuple of 1 ok).

### exp rolling (ndmoveexp, moving_exp.py) — float only
`move_exp_nansum move_exp_nanmean move_exp_nancount move_exp_nanvar
move_exp_nanstd move_exp_nancov move_exp_nancorr`
- alpha: scalar broadcast, or 1-D (len of core axis), or full-shape.
  min_weight default 0.
- Per step: `decay = 1 − alpha_i`; states *= decay; if valid: add current
  (count/sums += 1/x; weight += alpha_i).
- nancount: emit count if weight≥min_weight else NaN. nanmean: numer/denom.
  nansum: `zero_count` until first valid; NaN while zero_count.
- nanvar/std: sum_x, sum_x2, sum_weight (decayed count), sum_weight_2
  (decayed Σw²); `var_biased = sum_x2/sw − (sum_x/sw)²`;
  `bias = 1 − sw2/sw²`; emit `var_biased/bias` (or sqrt) if
  weight≥min_weight AND bias>0 else NaN.
- nancov: cov_biased = (sum_x1x2 − sum_x1·sum_x2/sw)/sw; same bias gate.
- nancorr: also sum_x1_2/sum_x2_2; cov/sqrt(var1·var2) with bias>0 and
  denominator>0 gates.

### matrix (ndmovematrix, moving_matrix.py + ndmatrix static)
`move_covmatrix move_corrmatrix move_exp_nancovmatrix move_exp_nancorrmatrix
nancovmatrix nancorrmatrix`
- Input (..., obs, vars)?? — MOVING matrix input is (n_obs, n_vars) →
  output (n_obs, n_vars, n_vars): one matrix per time step. STATIC
  nancovmatrix/nancorrmatrix input (..., vars, obs) → (..., vars, vars).
- **Pairwise accumulators per (i,j)**: only rows where BOTH series valid.
  f64 accumulators even for f32 input; per-variable shift for stability;
  clamp negative variances to 0; clip corr to [−1, 1].
- move_covmatrix: n≥min_count and n>1:
  cov = (prods/n − mean_i·mean_j) · n/(n−1) else NaN.
- move_corrmatrix: n ≥ max(min_count, 2); var>0 gates; NaN otherwise.
- min_count None → window; window must be 0<w≤shape[obs_axis].
- Exp matrix: decay all pairwise stats (squares by decay²); bias gate like
  1-D exp ops.

### fill (ndfill, funcs.py)
`ffill bfill` — (n),()->(n); limit=None → limit=shape[axis]; limit<0 →
ValueError. `lives_remaining` counts down ONLY on NaN; exhausted → current
resets to NaN (i.e., gap longer than limit stays NaN). bfill walks backward.
Numeric dtypes only (TypeError otherwise). limit is i64 for floats, i32 for
int32 arrays, i64 otherwise for ints.

## 9. Immediate next steps (in order)

1. Implement `core/tensor_view.mojo` + `drivers/gufunc.mojo` + hook loops;
   convert NanSum/AllNan/AnyNan to hook structs; generic binding; rewire
   facade. Acceptance: parity tests green + driver benchmark through the
   real facade reproduces the §6 numbers.
2. Remaining nanfuncs: nanmean, nanvar, nanstd, nanmin, nanmax, nancount
   (→i64), nanargmin/argmax (first-occurrence; errors), then
   nanmedian/nanquantile (selection, special case).
3. groupby family (`drivers/group_reduce.mojo`), rolling, exp, matrix, fill.
4. Benchmarks vs numbagg per op (pytest+codspeed harness); then the AWS
   full-matrix run for README numbers.
5. Repo manual steps pending: create GitHub remote+push, PyPI Trusted
   Publisher, CodSpeed token (see commit 29aa248 summary / RELEASING.md).

Session DB todos: nanfuncs-core (pending), nanfuncs-quantile, groupby,
rolling, rolling-exp, matrix, fill, benchmarks (all pending);
gufunc-driver-bench (done).
