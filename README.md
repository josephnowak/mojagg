# mojagg: numbagg-compatible numerical kernels in Mojo

NaN-aware reductions, grouped operations, rolling windows, and matrix statistics with a Python API and compiled Mojo kernels.

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0%2B-blue.svg)](https://numpy.org/)

---

## At a glance

`mojagg` is a high-performance numerical library that provides a drop-in, compiled alternative to [numbagg](https://github.com/numbagg/numbagg). It implements fast, NaN-aware reductions, grouped aggregations, rolling windows, and matrix statistics written in [Mojo](https://www.modular.com/mojo) and compiled ahead-of-time (AOT) into a native extension.

- **100% numbagg compatibility:** Identical function names, keyword signatures, and NaN/NaT semantics validated against numbagg's test suite.
- **Zero JIT compilation latency:** Kernels are pre-compiled into native machine code. Unlike Numba, there is no first-call compilation delay.
- **Python-first distribution:** Distributed as standard binary wheels (`pip install mojagg`). End users do not need Mojo or a local compiler toolchain installed.
- **Drop-in registration:** Seamlessly integrates with [xarray](https://github.com/pydata/xarray) and existing numbagg pipelines via `mojagg.register()` or `mojagg.patch()`.

---

## Why mojagg?

This project was born out of a desire to learn Mojo through a real-world, practical codebase. In my experience, the fastest way to truly understand a new programming language is to implement a production-relevant system with it.

I chose to replicate `numbagg` because it is an essential library in the scientific Python ecosystem (powering much of Xarray) and implements fundamental algorithms with clean Python/Numba syntax while competing with compiled code. That makes numbagg exceptionally maintainable—one of the most important qualities for any numerical library.

Beyond learning the syntax, I wanted to test the performance of Mojo with algorithms that do not require extreme low-level programming. I wanted to see if Mojo could be that sweet spot between Python-level readability and bare-metal performance, and whether it could beat Numba under those conditions.

I did not originally plan to replicate the entire library—I initially set out to implement only a handful of algorithms. But after discovering that Mojo could match and in several functions substantially outperform Numba, I decided to complete the implementation, achieve full compatibility, and make it available.

Mojo is also an emerging language, so implementations like these can be useful beyond this project: they provide a more diverse set of real-world numerical algorithms with which to evaluate the language, its compiler, and its runtime behavior. The results should be interpreted as one practical data point rather than a complete benchmark of Mojo.

---

## Why Mojo?

I find Mojo's proposal compelling: a language that remains syntactically close to Python while compiling directly to machine code, featuring first-class SIMD primitives, and offering a unified programming model across different hardware targets (CPUs and GPUs).

Coming from Python, I am not a fan of the steep learning curve and syntax of lower-level alternatives like C++ or Rust. Mojo provides a practical candidate for developing high-performance algorithms without sacrificing time learning an entirely different language paradigm.

Furthermore, as AI-assisted software engineering continues to advance, having a language that is clean, readable, and consistent across hardware targets makes it easier to migrate existing Python code and write new high-performance kernels from scratch. A readable, expressive language with compiled performance represents an exciting foundation for the future of numerical computing.

---

## Installation & Quick Start

Install `mojagg` via `pip`:

```bash
pip install mojagg
```

### Requirements
- Python `>= 3.10`
- NumPy `>= 2.0`
- Pre-compiled wheels include all native kernels; no Mojo installation is required for normal use.
- Binary wheels are currently available for Linux and Apple Silicon macOS. Windows users should install and run `mojagg` inside Ubuntu WSL.

### Quick Start

```python
import numpy as np
import mojagg

# 1. NaN-aware Reductions
values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
mojagg.nansum(values, axis=1)
# array([4., 9.])

# 2. Grouped Aggregations
labels = np.array([0, 1, 0])
mojagg.group_nanmean(values[0], labels, axis=0)
# array([2., nan])

# 3. Moving / Rolling Windows
mojagg.move_mean(values[0], window=2, min_count=1)
# array([1., 1., 3.])
```

Direct calls do not modify `numbagg` or any other library. For integration with downstream packages like `xarray`, see [Compatibility Registration](#compatibility-registration).

---

## Compatibility Registration

`mojagg` can act as a drop-in replacement for `numbagg` in libraries such as [xarray](https://github.com/pydata/xarray). By registering `mojagg`, any module importing or calling `numbagg` will automatically resolve to `mojagg`'s compiled kernels.

### Scoped Patching (Recommended)

Use the `patch()` context manager to temporarily redirect `numbagg` calls within a specific block:

```python
import mojagg
import xarray as xr

with mojagg.patch():
    # Inside this block, xarray and numbagg use compiled mojagg kernels
    # automatically restored upon exiting
    pass

assert not mojagg.is_registered()
```

### Global Registration

For process-wide registration across your entire application or interactive session:

```python
import mojagg

mojagg.register()

try:
    # All numbagg calls now dispatch to mojagg
    pass
finally:
    mojagg.unregister()
```

`is_registered()` returns `True` whenever `mojagg` is actively patching `numbagg`.

---

## Available Operations

All functions match the public signatures and return conventions of `numbagg`. The catalog is organized by operation family:

### Reductions
`allnan`, `anynan`, `count`, `nanargmax`, `nanargmin`, `nancount`, `nanmax`, `nanmean`, `nanmedian`, `nanmin`, `nanprod`, `nanquantile`, `nanstd`, `nansum`, `nanvar`

### Grouped Aggregations
`group_nanall`, `group_nanany`, `group_nanargmax`, `group_nanargmin`, `group_nancount`, `group_nanfirst`, `group_nanlast`, `group_nanmax`, `group_nanmean`, `group_nanmin`, `group_nanprod`, `group_nanstd`, `group_nansum`, `group_nansum_of_squares`, `group_nanvar`

### Rolling Windows
`move_corr`, `move_cov`, `move_mean`, `move_std`, `move_sum`, `move_var`

### Exponentially Weighted Windows
`move_exp_nancorr`, `move_exp_nancount`, `move_exp_nancov`, `move_exp_nanmean`, `move_exp_nanstd`, `move_exp_nansum`, `move_exp_nanvar`

### Matrix Statistics
`nancorrmatrix`, `nancovmatrix`, `move_corrmatrix`, `move_covmatrix`, `move_exp_nancorrmatrix`, `move_exp_nancovmatrix`

### Fill Operations
`bfill`, `ffill`

### Semantics & Contracts
- **Axes & Broadcasting:** Full support for integer axes, `axis=None`, and generalized broadcasting matching NumPy/numbagg rules.
- **Degrees of Freedom:** `nanvar` and `nanstd` follow the numbagg default of `ddof=1`.
- **Grouped Labels:** Grouped operations expect dense, non-negative integer labels.
- **Quantiles:** `nanquantile` supports both scalar quantiles and vector quantiles (where the quantile dimension appears as the leading axis of the output).
- **Dtype Preservation:** Matches numbagg's exact return types, including preserving input float types across predicate operations.

---

## Configuration & Tuning

`mojagg` provides flexible thread and threshold configuration via context managers, function calls, or environment variables:

```python
import mojagg

# Temporary per-call configuration
with mojagg.config(parallel_threshold=50_000, parallel_min_groups=64, threads=8):
    mojagg.group_nansum(values, labels)

# Global runtime configuration
mojagg.set_config(backend="cpu", threads=4)

# Current thread settings
cfg = mojagg.get_config()
```

### Environment Variables
- `MOJAGG_PARALLEL_THRESHOLD`: Minimum total elements required to trigger parallel dispatch (default: tuned per kernel).
- `MOJAGG_THREADS`: Number of worker threads for parallel execution.
- `MOJAGG_BACKEND`: Execution backend (`cpu`).

**Dispatch Rule:** Parallelization requires two thresholds to be met: sufficient outer slices and sufficient elements per input core, preventing multi-threading overhead on small workloads.

---

## Execution Model & Architecture

`mojagg` bridges Python NumPy arrays to compiled Mojo kernels using a generalized universal function (`guvectorize`) architecture:

```
Python Array (NumPy)
       │
       ▼
Python Facade (python/mojagg/)
  • Shape & dtype validation
  • Return layout allocation
       │
       ▼
Native Typed Binding (src/mojagg/python/)
  • Fixed-arity GUTensor descriptors
  • Symbolic core-dimension resolution
       │
       ▼
guvectorize Driver (src/mojagg/drivers/)
  • Outer-dimension broadcasting & slicing
  • Parallel work distribution (DispatchPolicy)
  • Scratch buffer management for non-contiguous slices
       │
       ▼
Compiled Mojo Kernel (src/mojagg/nanfuncs/, groupby/, moving/)
  • Explicit SIMD vectorization with masked NaN checks
  • Contiguous memory pointers with zero bounds checks
```

For an in-depth explanation of descriptor ownership, core axes, and memory layout, see the [guvectorize Driver Reference](docs/guvectorize.md).

---

## Benchmark Results & Analysis

Comprehensive benchmarks were executed on dedicated AWS instances across a wide variety of array shapes, dtypes, and memory layouts. The full interactive dashboard is available here:

👉 **[View the Complete Benchmark Results](docs/benchmarks/index.html)**

### Performance Overview

The results reveal clear trade-offs across different algorithmic patterns:

- **Where mojagg excels:**
  - **Large Reductions:** Algorithms like `nanstd`, `nanvar`, and `nansum` on medium-to-large inputs show significant speedups over numbagg thanks to efficient SIMD vectorization and accumulator unrolling.
  - **Matrix Statistics:** Multi-threaded kernels like `nancovmatrix` and `nancorrmatrix` achieve up to **10x speedups** along with dramatically lower memory consumption.
  - **Argmin / Argmax:** Kernels like `nanargmax` and `nanargmin` were manually restructured to eliminate redundant NaN comparison checks, outperforming numbagg substantially.

- **Where performance is comparable:**
  - **Rolling & Moving Windows:** Moving calculations (`move_mean`, `move_std`, etc.) show very similar performance between Mojo and Numba, serving as an effective direct comparison between the two compilers on identical algorithms.

- **Where numbagg leads:**
  - **Axis-0 Early-Exit Predicates:** For operations like `allnan` or `anynan` along `axis=0`, numbagg's loop structure can short-circuit earlier across outer dimensions, whereas mojagg processes through its standardized driver.
  - **Grouped Reductions with Scattered Labels:** Grouped operations where keys are not contiguous in memory present cache-locality challenges; sorting or indirect scatter operations are currently harder to vectorize effectively with SIMD.
  - **Multi-Quantile Selection:** `nanquantile` uses a custom multi-quantile partition algorithm. While it consumes up to **3x less peak memory** than numbagg, it is currently slower on certain shapes.

### Layout & Memory Considerations
To guarantee maximum execution speed without bounds checks or strided branching, mojagg's compiled kernels operate on contiguous memory. When an input core is non-contiguous, the driver allocates a thread-local scratch buffer to stage the contiguous slice. In contrast, Numba handles non-contiguous inputs through strided indexing. Depending on the input memory layout, this architectural choice represents a conscious trade-off between kernel simplicity/SIMD efficiency and scratch allocation.

---

## Pros & Cons

An honest assessment of `mojagg` compared to `numbagg`:

### Advantages
1. **Instant First Execution:** No JIT warm-up latency. Kernels are pre-compiled and run at full speed on the very first invocation.
2. **Superior Reduction & Matrix Performance:** Substantially faster on large-scale reductions and multi-threaded covariance/correlation matrix operations.
3. **Lower Memory Footprint:** More conservative memory usage across quantile and matrix calculations.
4. **Clean Parallelization:** Straightforward multi-threading via `DispatchPolicy` without unpredictable Numba parallelization heuristics.
5. **No Compiler Dependencies for Users:** Installs cleanly via `pip` on standard Linux/WSL environments.
6. **Fast Sorting Routines:** Efficient sorting and partition implementations in Mojo.

### Limitations
1. **Rank Limit (`MAX_RANK = 8`):** Arrays are currently limited to 8 dimensions (sufficient for >95% of scientific workloads, but less than NumPy's theoretical 64).
2. **Maintenance Complexity:** Writing generic, type-safe Mojo drivers with manual SIMD intrinsics is more complex than maintaining high-level Numba code.
3. **Driver Scratch Copies:** Non-contiguous slices along the reduction core require scratch staging.
4. **Preserved Numbagg Dtype Quirks:** Retains numbagg's convention of returning input dtypes for predicate reductions for 100% compatibility, adding complexity to the Python binding.
5. **No Native Windows Toolchain Yet:** Mojo compilation is currently Linux-oriented (Windows contributors develop via WSL).

---

## Personal Development Experience

Building `mojagg` provided valuable firsthand experience with Mojo in its current evolutionary stage:

- **Generic Programming & `guvectorize`:** The most challenging part of the project was building a `guvectorize` equivalent, which does not exist out of the box in Mojo. Mojo's support for variadic generic arguments, compile-time tuple manipulation, and shape inference is still maturing. Achieving clean generic dispatch required implementing explicit typed helpers for common signatures.
- **NumPy Views vs. Contiguous Strides:** Standard scientific Python relies heavily on strided views. Because Mojo's standard buffers favor contiguous memory, bridging the gap required building custom view abstractions (`NDView`) and scratch buffers to avoid unnecessary copies.
- **SIMD Vectorization:** Mojo's SIMD syntax is intuitive and expressive. Handling register tails cleanly when array lengths are not multiples of the vector register width was the primary implementation detail.
- **Compiler Maturity & JIT:** While compilation of dozens of specialized kernel variants takes noticeable time, the ability to iterate using the JIT was invaluable during kernel development.
- **AI-Assisted Development:** Rapid syntax changes across Mojo compiler versions often led AI models to suggest outdated syntax, necessitating strict automated parity tests and compiler verification.
- **Windows Workflow:** In the absence of a native Windows Mojo compiler, developing a unified PowerShell wrapper (`scripts/mojagg.ps1`) targeting Ubuntu WSL created a seamless local development workflow.

---

## Contributing: Implementing an Operation

Extending `mojagg` with a new operation follows a five-step path:

1. **Python Facade:** Add the public function in `python/mojagg/` with docstrings, argument validation, and return shape planning.
2. **Native Binding:** Expose the typed binding in `src/mojagg/python/` defining the `GUTensor` input/output contract.
3. **Mojo Kernel:** Implement the numerical algorithm in `src/mojagg/nanfuncs/`, `groupby/`, or `moving/` utilizing SIMD primitives.
4. **Parity Tests:** Add test cases under `tests/python/` verifying 100% numerical and exception parity against `numbagg`.
5. **Benchmark:** Add the function to the benchmark suite (`benchmarks/`) to validate performance across data sizes and layouts.

For detailed rules on memory ownership, core axes, and driver contracts, consult the [guvectorize Driver Reference](docs/guvectorize.md) and [`AGENTS.md`](AGENTS.md).

---

## Local Development (WSL)

Contributors working on Windows use the unified PowerShell wrapper, which transparently executes all commands inside Ubuntu WSL:

```powershell
# Verify environment and dependencies
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 doctor

# Build native extension
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 build

# Run Mojo unit tests
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-mojo

# Run Python parity tests vs numbagg
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-python

# Format and lint
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 format
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 lint

# Run local benchmark suite
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-public --profile quick
```

---

## Roadmap

Future development directions and exploration areas:

- **Nested Parallelism:** Explore inner-loop parallelization within individual cores while balancing outer-dimension slicing to avoid CPU thread oversubscription.
- **Simplified Generic Signatures:** Refactor driver and kernel signatures as Mojo's metaprogramming and variadic generic capabilities mature.
- **Additional Operations:** Implement algorithms not covered by numbagg (such as `rankdata` to remove the runtime dependency on SciPy).
- **Non-NaN Baseline Kernels:** Explore pure non-NaN kernel variants to evaluate how compiled Mojo kernels compare directly against baseline NumPy implementations.
- **Kernel Optimization:** Continue optimizing the existing algorithms by testing different SIMD widths and unroll factors, and by evaluating additional SIMD capabilities where they fit the workload.

---

## Maintenance Automation

The repository includes scheduled CI workflows to maintain stability across ecosystem updates:
- [`mojo-watch.yml`](.github/workflows/mojo-watch.yml): Tracks new Modular MAX releases and drafts toolchain updates.
- [`python-support-watch.yml`](.github/workflows/python-support-watch.yml): Monitors CPython releases and NumPy compatibility.
- [`pr-review.yml`](.github/workflows/pr-review.yml): Automated pull request checks for test parity and benchmark evidence.

---

## License & Acknowledgments

- **License:** Released under the [BSD-3-Clause License](LICENSE).
- **Acknowledgments:** Inspired by and API-compatible with [numbagg](https://github.com/numbagg/numbagg) (BSD-3). Group-label conventions follow [numpy-groupies](https://github.com/ml31415/numpy-groupies).
