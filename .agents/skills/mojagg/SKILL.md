---
name: mojagg
description: How to contribute a numbagg-compatible operation to mojagg using the current Mojo GUFuncKernel, GUTensor, native binding, Python facade, parity-test, and benchmark architecture. Read before adding or changing a kernel, driver, binding, public function, test, or benchmark.
---

# mojagg contribution and implementation guide

mojagg is a Python-first, AOT-compiled Mojo implementation of numbagg-style
NaN-aware operations. The Python package owns the public API, promotion rules,
axis normalization, output allocation, and error translation. The Mojo package
owns the numerical loop, typed memory access, SIMD, outer-slice traversal, and
parallel scheduling. The public contract is the behavior observed from Python:
function names and signatures, values, shapes, dtypes, NaN behavior, and
exception types must agree with numbagg wherever numbagg has the operation.

This file is the contribution guide. Read it before changing a kernel or
binding. For repository commands and the WSL environment, also follow
.agents/skills/mojagg-workflow/SKILL.md.

## Current source of truth

The current checkout has one generic gufunc driver, split into focused modules:

~~~text
Python call
  -> python/mojagg facade
  -> compiled PythonModuleBuilder binding
  -> GUTensor descriptors + AxisSpec
  -> guvectorize signature planning and outer-slice execution
  -> GUFuncKernel.__call__(one core tuple)
  -> caller-owned NumPy output
~~~

The driver files are:

| File | Responsibility |
|---|---|
| src/mojagg/drivers/guvectorize_spec.mojo | GUFuncKernel, CoreSpec, Dim, FixedDim, and AxisSpec |
| src/mojagg/drivers/gutensor.mojo | Borrowed GUTensor descriptors, pointer/span access, and strided-core copying |
| src/mojagg/drivers/guvectorize_layout.mojo | MAX_RANK, DimArray, and runtime OperandPlan metadata |
| src/mojagg/drivers/guvectorize_plan.mojo | Core-axis selection, symbolic dimension binding, and outer broadcasting |
| src/mojagg/drivers/guvectorize_execute.mojo | Outer iteration, worker-local scratch, operation copies, and dispatch |
| src/mojagg/drivers/guvectorize.mojo | Public re-export surface for the driver |

Read [docs/guvectorize.md](../../../docs/guvectorize.md) for the detailed
driver reference. It is the concrete explanation of gufunc terminology,
core/inner/outer dimensions, axis flattening, output placement, outer
broadcasting, stride handling, scratch ownership, scheduling, and binding
responsibilities. Keep this skill focused on contribution decisions and use
the reference when reasoning about an implementation detail.

The implemented operation families in this checkout are under
src/mojagg/nanfuncs, src/mojagg/groupby, and the two native binding modules
under src/mojagg/python. The Python facade also exposes static matrix and fill
operations. Older documents may mention reduce_axis, group_reduce,
rolling_axis, or other drivers that are not present here. Do not create or
reference those paths from a new contribution unless the files have actually
been added. Check the current tree before choosing an extension point; the
README's future rolling catalog is not an implementation contract.

## Development workflow

On Windows, every project command goes through the WSL wrapper. Do not invoke
the host python, pytest, mojo, pixi, or ruff for this repository.

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 doctor
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 build
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-mojo
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 test-python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 lint
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-quick
~~~

Run doctor before changing code. Use --env py310 through --env py314
when a locked Python version matters. Use the wrapper's files and search
commands for normal source discovery; the normal source boundary is
python/, src/, and tests/.

Use test-mojo for driver and kernel behavior, test-python for public parity,
and both for a Mojo/Python boundary change. Run lint for every Mojo or Python
change. A numerical or dispatch change also needs bench-quick; compare the
same cases with numbagg using bench-reference when the operation has a
numbagg equivalent. Dispatch thresholds are benchmark data, not tuning guesses.

## The Numba gufunc mental model

The closest Numba concept is a @guvectorize function. A Numba gufunc declares
core dimensions such as (n)->() and lets the gufunc machinery handle outer
broadcasting, axes, and repeated calls to the inner function. In mojagg:

| Numba concept | mojagg concept |
|---|---|
| Core dimension name n | Dim[0] in a CoreSpec |
| Fixed core dimension | FixedDim[size] |
| Gufunc argument | A typed GUTensor[dtype, output, core] |
| Gufunc signature | GUFuncKernel.Signature, a native Tuple of tensors |
| Inner gufunc body | GUFuncKernel.__call__ |
| Numba's outer iteration/broadcasting | GUVectorizePlan and guvectorize |
| Numba JIT specialization | Mojo compile-time DType, output flag, and core spec parameters |

The important difference is that GUFuncKernel is an operation object with a
fully static tuple type. There is no boxed runtime dtype dispatch or Python
callback for every slice. The native binding selects a compiled specialization
once, the driver prepares one logical core for each outer position, and the
operation processes that core with raw typed spans. Mojo builds the extension
ahead of time; it does not JIT-compile a Python gufunc at the first call.

## GUTensor: what it is and where it is used

GUTensor is a non-owning descriptor for one gufunc operand. It is metadata plus
an address, not an array container and not a memory owner. Its type is:

~~~mojo
GUTensor[element_dtype, output, core_spec]
~~~

The three compile-time parameters mean:

- element_dtype is the exact Mojo DType loaded from or written to memory.
- output is False for a read input and True for a writable output.
- core_spec declares the logical gufunc core dimensions for this operand.

The runtime fields are an integer address, a bound flag, rank, total active
length, shape, and element strides. shape and stride use elements, not
bytes. The binding computes them from NumPy's arr.shape and
arr.strides / arr.dtype.itemsize once at the Python boundary.

Useful constructors and methods are:

~~~mojo
# A bound borrowed view over caller-owned storage.
var input = GUTensor[DType.float64, False, CoreSpec[Dim[0]]].borrow(
    address,
    shape,
    stride,
    ndim,
)

# An unbound output template. build_signature fills its shape and stride.
var output_template = GUTensor[
    DType.float64,
    True,
    CoreSpec[],
].empty()

var values = input.read_span()
var destination = output.write_span()
~~~

read_span() and write_span() are the normal kernel interface. The output
method is compile-time restricted to output=True. Use unsafe_ptr() on a
span for a hot loop, and use the active span length; an operation should not
perform its own rank or axis traversal.

GUTensor is used in three places:

1. A native binding creates bound descriptors by borrowing NumPy addresses.
2. The planner reads descriptor shape and element-stride metadata to build an
   OperandPlan for each operand.
3. The executor copies descriptor values per worker, rebinds their address and
   active length for each outer slice, and calls the operation with that tuple.

The descriptor does not retain a Python owner. The binding must keep the input,
output, and any auxiliary NumPy arrays alive for the duration of the native
call. Do not store a PythonObject or an owning Python reference in a Mojo
kernel.

The boundary is zero-copy for ordinary inputs and outputs. A non-contiguous
selected read core is copied into worker-local scratch because the operation
contract is a contiguous span; this is an internal per-slice fallback, not a
full hidden ascontiguousarray of the user's input. Writable cores must already
be contiguous in the operation's logical core order, otherwise the planner
raises writable core must be contiguous.

## Core dimensions, signatures, and axes

CoreSpec is a compile-time description of the logical dimensions passed to an
operation. Dim[id] is a named symbolic extent. Equal IDs mean equal extents
when a signature is resolved. FixedDim[size] requires an exact extent.

~~~mojo
CoreSpec[]                    # scalar core, rank 0
CoreSpec[Dim[0]]              # one symbolic core dimension, often n
CoreSpec[Dim[0], Dim[1]]      # two ordered core dimensions
CoreSpec[Dim[0], Dim[0]]      # square output: both axes share one extent
CoreSpec[FixedDim[3]]          # exactly three elements
~~~

CoreSpec stores the dimension IDs and fixed flags in compile-time metadata.
It does not store a runtime shape. CoreBindings in the planner resolves the
runtime values. Symbol IDs are limited by MAX_RANK (currently 8), so use a
small stable ID set within a signature.

The complete kernel signature is a native tuple, in operand order:

~~~mojo
comptime Signature = Tuple[
    GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
    GUTensor[Self.dtype, True, CoreSpec[]],
]
~~~

guvectorize[Operation] has a compile-time assertion that the passed tensor
tuple is exactly Operation.Signature. A wrong operand order, dtype, output
flag, or core spec is a compile-time contract error. Keep the tuple in the same
order in the kernel, binding, and facade.

AxisSpec is the runtime counterpart to the core-axis part of the signature. It
contains a fixed-capacity DimArray of physical axis positions and an explicit
count. The input and output axis specs can differ, for example an input
reduction core has one or more axes while a scalar output has zero axes.

The Python facade normalizes user axes before crossing the boundary. For value
reductions, _resolve_axes sorts selected axes by descending byte stride so the
unit-stride axis is innermost. Arg reductions preserve the user's normalized
axis order because the flattened index is observable. Grouped axes preserve
their order. Negative axes are resolved in Python; the Mojo driver expects
normalized, unique, in-range positions.

If a one-dimensional operation receives several selected input axes, the
planner treats the selected core as one flattened Dim[0] extent. The core
order in AxisSpec determines the flattening order. This is how nanmean(a,
axis=(0, 2)) still reaches a one-dimensional NanMean kernel.

The planner removes selected core axes from each operand to obtain its outer
shape. Input outer shapes are right-aligned and broadcast like NumPy: each
dimension must match or be one. All writable outputs must have exactly the
resolved outer rank and shape. GUVectorizePlan records outer strides and
maps each flat outer index back to an address with an odometer.

build_signature is used when outputs start as unbound templates and their
shape follows from input core bindings and outer broadcasting. It fills output
metadata, after which the binding validates and binds the caller-allocated
NumPy output. build_signature_with_bindings is for a dimension supplied by a
separate runtime object. nanquantile uses it to seed the quantile output
dimension with the number of quantiles before binding the output.

## Choosing the operation shape

Use the smallest existing driver contract that expresses the public operation.
These are the signatures already used in the source:

| Operation shape | Signature pattern | Axis/output rule |
|---|---|---|
| Scalar reduction | input: CoreSpec[Dim[0]], output: CoreSpec[] | Python may select one or many axes; the driver supplies a flattened read span and one output value |
| Grouped reduction | values: CoreSpec[Dim[0]], labels: CoreSpec[Dim[0]], groups: CoreSpec[Dim[1]] | Values and labels are aligned; Python sizes and initializes dense group outputs |
| Grouped reduction with workspaces | Add writable CoreSpec[Dim[1]] tensors | Auxiliary outputs hold counts, best values, seen flags, or sums of squares |
| Same-shape transform/fill | Input and output both CoreSpec[Dim[0]] | Pass the same core axes to input and output; output core must be writable-contiguous |
| Static matrix | Input CoreSpec[Dim[0], Dim[1]], output CoreSpec[Dim[0], Dim[0]] | The binding selects the two trailing axes and the kernel computes a square matrix |
| Selection with external output length | Input CoreSpec[Dim[0]], output CoreSpec[Dim[1]] | Seed CoreBindings from the external length, as quantile does |

Do not force a scalar reduction into a new N-D driver. Put rank/axis/stride
logic in the existing guvectorize path and keep the operation one-core. If a
new family genuinely needs a different state machine or core shape, first
extend the shared driver with a tested contract and then add the operation.

## Building a new GUFuncKernel

Start with one Mojo file per public operation or tightly coupled variant under
the matching family directory. A scalar reduction skeleton looks like this:

~~~mojo
from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)


@always_inline
def reduce_core[dtype: DType](
    values: Span[Scalar[dtype], ImmUntrackedOrigin],
) -> Scalar[dtype]:
    comptime width = simd_width_of[dtype]() * 8
    var accumulator = SIMD[dtype, width](0)
    var pointer = values.unsafe_ptr()

    def step[vector_width: Int](i: Int, evl: Int) {
        imm pointer, mut accumulator
    }:
        if evl == width:
            var block = pointer.unsafe_load[width=width](i)
            comptime if dtype.is_floating_point():
                accumulator += isnan(block).select(
                    SIMD[dtype, width](0), block
                )
            else:
                accumulator += block
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var value = pointer[unsafe_offset=i + lane]
                    comptime if dtype.is_floating_point():
                        if not isnan(value):
                            accumulator[lane] += value
                    else:
                        accumulator[lane] += value

    vectorize[width](len(values), step)
    return accumulator.reduce_add()


@fieldwise_init
struct ExampleReduce[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[]],
    ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        output.write_span()[0] = reduce_core[Self.dtype](input.read_span())
~~~

The example shows the contract; the identity and finalization must be changed
for the operation's semantics. Follow the checked-in kernels for the actual
algorithm. NanSum is the simplest scalar reduction, NanMean demonstrates a
float64 sum plus a count while retaining the requested output dtype, NanVar
demonstrates a runtime ddof field, and NanQuantileKernel demonstrates the
non-streaming workspace exception.

Kernel rules:

- __call__ receives one prepared tuple for one outer slice. It must not read
  Python objects, inspect ndim, normalize axes, or choose workers.
- Read inputs with read_span() and outputs with write_span(). Use
  len(span) for the active core length.
- std.algorithm.vectorize owns the full-block and tail calls. The current
  code uses the (i, evl) form so a tail never performs an out-of-bounds wide
  load. Guard tail lanes with if lane < evl.
- Specialize integer and floating behavior with comptime if dtype.is_floating_point().
  Integer kernels must not execute a NaN branch.
- Use std.math.isnan for the current Mojo SIMD NaN path and verify masks with
  parity tests. Do not replace it with an unverified v != v shortcut.
- Accumulate in float64 only when the operation's reference semantics require
  it, such as nanmean, variance, and matrix statistics. This is internal
  accumulator precision; it does not authorize a hidden input cast.
- Keep group scatter stores scalar because each label selects an arbitrary
  destination. Vectorize the loads and arithmetic around the unavoidable
  scatter.
- Keep the operation's core loop independent of outer rank, axis order, and
  strides. A non-contiguous read has already been copied to a worker scratch
  span by the driver.

### Stateful kernels and worker copies

execute_serial_or_parallel copies the operation once per worker. A stateless
operation can use ImplicitlyCopyable. A runtime parameter such as ddof, limit,
or matrix dimensions is stored in the operation and copied by value.

If the operation owns workspace, implement the current Mojo copy constructor
and destructor pattern. Each worker copy must own independent storage:

~~~mojo
def __init__(out self, *, copy: Self):
    self.parameter = copy.parameter
    self.workspace = alloc(Layout[Scalar[Self.dtype]](count=0))
    self.capacity = 0

def __deinit__(deinit self):
    dealloc(self.workspace^)
~~~

NanQuantileKernel follows this pattern because sorting mutates a private
worker buffer. Ordinary streaming kernels should allocate nothing in their hot
loop. Allocate output, auxiliary arrays, and driver scratch before the loop;
the explicit quantile workspace is the special non-streaming case.

## Native Mojo bindings

The binding is the only layer that touches PythonObject during execution. It
validates the boundary, creates typed descriptors, selects the operation
specialization, and enters guvectorize once.

For a scalar reduction, add a typed binding in
src/mojagg/python/nanfuncs_native.mojo and reuse _apply_reduction:

~~~mojo
def myop_binding[dtype: DType](
    arr: PythonObject,
    axes: PythonObject,
    out_arr: PythonObject,
    cfg: PythonObject,
) raises -> PythonObject:
    return _apply_reduction[dtype, dtype, ExampleReduce[dtype]](
        arr,
        axes,
        out_arr,
        cfg,
        ExampleReduce[dtype](),
        "myop",
    )
~~~

_apply_reduction validates input and output dtype, converts NumPy shape and
byte strides to element metadata, creates an input CoreSpec[Dim[0]] tensor and
an unbound CoreSpec[] output template, calls build_signature, validates/binds
the Python output, and invokes guvectorize. If the result dtype differs, use
the separate out_dtype type parameter and the matching GUTensor output type.

For a runtime parameter, place it before cfg in the native function and call
the helper with that parameter. The Python _reduce.py helper will call the
entry point as (array, axes, out, parameter, cfg).

Register every supported specialization in PyInit_nanfuncs_native:

~~~mojo
m.def_function[myop_binding[DType.float64]]("myop_f64")
m.def_function[myop_binding[DType.float32]]("myop_f32")
~~~

Only the PyInit_* function is @export/abi("C"); registered functions
remain normal Mojo functions and may be raises. The module name, init symbol,
binding filename, and scripts/build_ext.py binding list must agree. Existing
nanfuncs and grouped operations belong in their existing native module; a new
family requires a new binding module, a build entry, a loader entry in
python/mojagg/_native.py, and a Python facade.

### Grouped binding shape

Grouped operations use src/mojagg/python/groupby_native.mojo and
_apply_group. The public facade has already allocated and initialized the
group result and any workspaces. The binding receives:

~~~text
(values, labels, normalized_axes, group_output, (auxiliary_tuple, config, ddof))
~~~

The common typed shape is:

~~~mojo
Tuple[
    GUTensor[value_dtype, False, CoreSpec[Dim[0]]],
    GUTensor[label_dtype, False, CoreSpec[Dim[0]]],
    GUTensor[output_dtype, True, CoreSpec[Dim[1]]],
    # optional writable auxiliary tensors, also CoreSpec[Dim[1]]
]
~~~

Set aux_count, auxiliary dtypes, and the operation type consistently in the
binding. Register all value/label pairs that the Python facade will expose.
The group kernels assume dense output labels, skip negative labels, and use the
output arrays as their initialized identities/workspaces.

## Python facade plumbing

The facade is not boilerplate: it is where numbagg compatibility belongs. A
new operation is incomplete until a Python user can import and call it without
knowing the native symbol name.

### Scalar nanfuncs

For a regular reduction:

1. Add the kernel import and typed binding/registrations in
   src/mojagg/python/nanfuncs_native.mojo.
2. Add a dtype-to-native table in python/mojagg/nanfuncs.py.
3. Build the public function with reduce_op in that file. Use
   reduce_op_with_ddof for the standard runtime ddof shape, or write an
   explicit wrapper when the public signature has different parameters.
4. Choose the promotion function, result dtype, empty-input error, invalid
   output error, and axis-order behavior from the numbagg reference.
5. Export the function from python/mojagg/__init__.py and __all__; add it
   to python/mojagg/compat.py if it is part of the drop-in registration list.

The regular table pattern is:

~~~python
_myop_kernels = {
    np.dtype(np.float64): _native.myop_f64,
    np.dtype(np.float32): _native.myop_f32,
}

myop = reduce_op(
    "myop",
    _myop_kernels,
    _promote_nanmean_like,
    out_dtype=np.float64,
)
~~~

_reduce_axis turns an input into an ndarray, performs only the documented
promotion or endian normalization, normalizes axes, allocates the output,
resolves config once, and makes one native call. Do not add a Python loop over
outer slices, moveaxis, or ascontiguousarray to make a kernel easier.

### Grouped operations

python/mojagg/groupby.py owns label validation, axis/label shape rules,
num_labels, dense output initialization, auxiliary array allocation, and
visible promotions. To add a grouped operation:

1. Add its name to _GROUP_KERNELS construction and decide whether it belongs
   in _GROUP_FLOAT64_PROMOTIONS or the boolean-supported set.
2. Add its native typed binding and every registration in
   groupby_native.mojo.
3. Initialize the result identity and any auxiliary arrays in _group_reduce.
   Empty groups and all-NaN groups must have the same result as numbagg.
4. Add the public wrapper, __all__, package export, and compatibility entry.

Grouped axis behavior is part of the API:

- axis=int expects labels shaped like the selected value axis.
- axis=tuple expects labels shaped like the selected axes and broadcasts
  them over the remaining value axes with a zero-copy view.
- axis=None requires labels and values to have the same shape and reduces all
  axes.
- The result shape is the unselected outer shape followed by num_labels.
- If num_labels is omitted, it is one greater than the maximum non-negative
  label; if supplied it must not be smaller than that requirement.
- Negative labels are skipped. They are never an error and never index from
  the end.

### Special families

nanquantile/nanmedian are selection operations, not streaming reductions.
The facade converts numeric input to the f64 selection dtype, binds the
quantile array, seeds the output dimension through CoreBindings, and the
kernel uses worker-private sortable storage. Preserve scalar quantile squeeze
and the public leading quantile axis.

Static matrix operations use python/mojagg/matrix.py and the matrix bindings.
They accept (..., vars, obs), use pairwise-valid observations, and have
float32/float64 native specializations after explicit facade promotion. Do not
replace pairwise accumulation with listwise masking.

Fill operations use a same-core input/output signature. The Python facade
handles limit and integer fast paths; the Mojo kernel walks each prepared core.

## Dtype and result contracts

The native registry is explicit. Add only the combinations justified by the
reference behavior; do not make the kernel accept arbitrary dtypes and cast
inside its loop.

| Family | Native combinations in the current checkout | Result notes |
|---|---|---|
| Basic nanfuncs | f64, f32, i64, i32 where the operation supports them | nancount/count return int64; predicates return bool; arg reductions return int64 |
| nanmean, nanvar, nanstd | f64, f32 native | The facade visibly promotes integer/small inputs; internal mean/variance accumulation may be f64 |
| nanmin/nanmax | f64, f32, i64, i32 | i32 results are widened to int64 to match the public contract |
| Grouped operations | value f64/f32/i64/i32 with label i64/i32 | var/std are float-only after facade promotion; dense group output and auxiliaries are allocated in Python |
| Static matrices | f64, f32 | Numeric integer inputs are explicitly promoted by matrix.py |
| Fill | f64, f32, i64, i32 | Integer inputs do not contain NaNs and use a Python copy path |
| Quantile/median | f64 selection kernel | Numeric inputs are explicitly converted to f64 for selection |

The exact public promotion rules live in python/mojagg/_reduce.py,
groupby.py, matrix.py, and fill.py. Unsupported dtype errors must list
the supported dtypes. Big-endian normalization and documented public promotion
are the only input copies currently allowed at the facade. A float32 result
may still be produced from a float64 accumulator; that is not the same as
silently changing the input dtype in Mojo.

## Numerical semantics to preserve

Read the relevant numbagg source and tests before choosing an identity or
algorithm. The common contracts in this checkout include:

- Skip NaN values per operation. nansum uses zero, nanprod uses one,
  nanmean uses sum/count, and nancount returns an int64 count.
- nanmean, nanvar, and nanstd return NaN when there are no valid values;
  variance and standard deviation return NaN when count - ddof <= 0.
- nanmin/nanmax return NaN for all-NaN floating slices and raise the
  numbagg-compatible empty reduction error for an empty selected axis.
- nanargmin/nanargmax return the first valid occurrence on ties. The
  facade translates an invalid negative native result into the public
  all-NaN/empty exception.
- allnan has the empty identity True; anynan has the empty identity
  False. Integer inputs cannot be NaN.
- Group labels below zero are ignored. Group output identity, unseen-group
  behavior, first/last tie behavior, pairwise NaN handling, and ddof boundaries
  must match numbagg.
- The facade deliberately handles the odd axis cases. In particular, inspect
  _resolve_axes before changing axis behavior: the current implementation
  normalizes negative axes, validates duplicates, sorts value reductions by
  stride, and treats an empty axis tuple as a flattening reduction.

Do not rely on NumPy alone for a parity claim. Compare values, shapes, dtypes,
and exception types directly with numbagg. Operations without a numbagg
equivalent must say so in the test and benchmark, then use the specified NumPy
contract.

## Testing a new operation

Write the public parity test before or alongside the kernel. At minimum cover:

~~~text
empty selected core and empty outer dimensions
all-NaN and mixed NaN/finite slices
SIMD tails around several widths (including n=0, 1, width-1, width, width+1)
contiguous, reversed, strided, transposed, and broadcast views
axis=None, one axis, negative axis, tuples in both orders, and duplicate errors
f32/f64 plus every supported integer/bool promotion
result shape, result dtype, and exact exception type/message class
~~~

For grouped operations also cover negative labels, repeated labels, empty
groups, num_labels, label dtype, one-axis labels, multi-axis broadcast,
boolean support, and ddof boundaries. For matrices cover pairwise missing
observations, diagonal/off-diagonal behavior, constant variables, and integer
promotion. For selection operations cover scalar/vector quantiles and NaN
quantiles.

Use the existing structure:

- tests/mojo/test_gufunc.mojo tests signatures, multi-input/output tuples,
  strided input scratch, writable-core rejection, outer order, and dispatch
  thresholds.
- tests/python/ is the executable public contract and contains focused
  internal tests plus the upstream numbagg snapshot.
- tests/python/conftest.py registers mojagg into numbagg for upstream tests.
  When a test needs the unpatched reference, use the existing unregister/
  restore pattern rather than comparing mojagg to itself.

Meaningful Mojo driver tests should be added when the planner or operation
contract changes. A Python test that merely repeats a scalar arithmetic line
does not prove a kernel; test layouts, boundaries, promotions, or parity.

## Performance and memory rules

These rules apply to ordinary kernels and drivers:

- Validate dtype, rank, shape, and strides once at the Python/native boundary.
  The kernel then works only with typed spans and pointers.
- Do not allocate, grow collections, print, call Python, read environment
  variables, or read mutable global configuration in a hot kernel loop.
- Allocate outputs, group workspaces, bitmaps, and driver scratch up front.
  The quantile workspace is an intentional worker-private exception because
  selection needs mutable storage.
- Use explicit SIMD and vectorize; reduce SIMD lanes once at the end where
  the algorithm permits. Check masked NaN and tail behavior, especially for
  float32.
- Parallelism is over the driver outer-slice domain. DispatchPolicy requires
  both enough outer groups and enough input-core work, then caps workers at the
  outer count. A new threshold or policy branch requires a benchmark result.
- Keep operation copies independent across workers. Group scatter writes must
  not introduce races; if a future parallel group implementation needs partial
  accumulators, allocate and merge them explicitly.
- Store raw addresses as Int in long-lived descriptor structs when needed;
  current Mojo does not allow the relevant mutable pointer origins in struct
  fields. Create typed pointers in accessors or local scopes.
- Use current Mojo syntax: def, comptime, out self, deinit self, explicit
  numeric conversions, Self.<parameter> inside generic structs, std. imports,
  unsafe_offset=, and runtime closures. Do not generate fn, alias,
  @parameter, old __copyinit__/__moveinit__ forms, or @__parameter.
- out is reserved in Mojo. Use out_arr or another name for a Python binding
  parameter; do not declare a Mojo argument named out.

Every optimization must be visible in the appropriate benchmark. A fast
kernel that changes parity, copies a whole input, or makes dispatch thresholds
less predictable is not an acceptable contribution.

## Complete new-operation checklist

Before calling an operation complete, verify each applicable item:

1. Read the numbagg implementation/tests and write down values, shapes, dtypes,
   promotion, empty/all-NaN, axis, and exception semantics.
2. Choose the existing guvectorize signature and assign core symbols. Decide
   whether the result shape comes from build_signature, direct bound output,
   or build_signature_with_bindings.
3. Add one Mojo operation file. Implement GUFuncKernel, exact Signature,
   __call__, valid identities, tails, NaN masks, and any explicit workspace
   copy/destructor logic.
4. Add a native typed binding, validate the NumPy boundary, and register every
   supported specialization with PythonModuleBuilder.
5. Add the dtype table and public wrapper in the correct Python facade. Handle
   promotion, output allocation/initialization, axis normalization, kwargs,
   errors, __init__.py, __all__, and compatibility registration.
6. Add parity and boundary tests, including a Mojo driver test when the shared
   contract is involved. Compare directly with numbagg wherever possible.
7. Add the corresponding benchmark case and reference comparison. Do not alter
   dispatch thresholds without data.
8. Build and run the relevant wrapper commands: build, test-mojo,
   test-python, lint, and bench-quick as applicable.
9. Inspect the diff and git status; leave unrelated user changes untouched.

The contribution is complete only when the public Python path, compiled native
path, tests, and documented performance contract all agree.

## Common failure modes

- Using the old driver names. The current implementation is the split
  guvectorize driver. Search for the file before importing a driver.
- A signature that almost matches. Tuple order, dtype, output flag, and
  CoreSpec are all part of the compile-time type. Copy the same tuple into
  the binding call and kernel.
- Treating GUTensor as an owning array. It only borrows an address and
  metadata. Keep NumPy objects alive in the binding and free only allocations
  owned by a Mojo operation.
- Doing axis work in __call__. The operation sees one core. Put axis
  normalization in the facade and layout work in the driver.
- Writing a non-contiguous output core. The planner rejects it. Allocate a
  C-contiguous output whose selected output axes are unit-stride in logical
  core order.
- Forgetting worker copies. A mutable operation field or workspace is
  copied once per worker. Define copy/deinit behavior and never share mutable
  scratch accidentally.
- Registering only one dtype. A Python dtype table is not enough; the
  corresponding compiled m.def_function specialization must exist.
- Comparing against a patched numbagg. conftest.py registers mojagg;
  unregister before obtaining an independent reference.
- Calling host tools or broadening the API. Use the WSL wrapper and keep a
  new operation within the requested family and current public contract.
