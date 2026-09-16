# The `guvectorize` driver

This document is the implementation reference for the generic Mojo
`guvectorize` driver. It describes the contract that a new `GUFuncKernel`, a
native Python binding, or a public facade must follow. The code is split across
the following modules:

| Module | Role |
| --- | --- |
| [`guvectorize_spec.mojo`](../src/mojagg/drivers/guvectorize_spec.mojo) | Compile-time core dimensions, runtime axis lists, and the kernel trait |
| [`gutensor.mojo`](../src/mojagg/drivers/gutensor.mojo) | Borrowed tensor descriptors, pointer/span access, and input scratch copies |
| [`guvectorize_layout.mojo`](../src/mojagg/drivers/guvectorize_layout.mojo) | Fixed-capacity dimension arrays and per-operand plans |
| [`guvectorize_plan.mojo`](../src/mojagg/drivers/guvectorize_plan.mojo) | Signature resolution, core binding, and outer broadcasting |
| [`guvectorize_execute.mojo`](../src/mojagg/drivers/guvectorize_execute.mojo) | Outer iteration, scratch allocation, worker copies, and dispatch |
| [`guvectorize.mojo`](../src/mojagg/drivers/guvectorize.mojo) | Public re-exports used by kernels and bindings |

The driver is a compiled execution layer, not a Python callback mechanism. The
Python facade validates public arguments, promotes inputs, and normalizes
axes. A typed Mojo binding borrows the inputs, builds the signature, allocates
NumPy outputs from the resolved metadata, binds their addresses, and calls the
driver. The driver then invokes a compiled operation once for every outer
slice:

```text
Python function
    │  promotion and axis normalization
    ▼
typed PythonModuleBuilder binding
    │  dtype checks, signature planning, NumPy allocation, address/shape/stride conversion
    ▼
GUTensor descriptors + AxisSpec
    │
    ├─ build_signature_plan: resolve symbolic core sizes, output shape, and layout plan
    ├─ materialize_outputs: allocate NumPy outputs and bind their addresses in place
    └─ guvectorize: execute the existing plan over every outer slice
           │
           └─ GUFuncKernel.__call__(one prepared core tuple)
```

`MAX_RANK` is currently `8`. Runtime dimension metadata is held in inline
fixed arrays, so the driver does not allocate a dynamic shape or stride object
for every call.

## The gufunc mental model

A generalized universal function separates each operand into **core
dimensions** and **outer dimensions**. The core is the part consumed by one
invocation of the numerical operation. The outer dimensions select how many
times that operation is invoked and how operands are broadcast against one
another.

For example, the Numba spelling of a scalar reduction is conceptually:

```python
@guvectorize([(float64[:], float64[:])], "(n)->()")
def sum_core(values, result): ...
```

`n` is a core dimension. The gufunc machinery calls `sum_core` once for each
outer position, and the core call sees one `values` vector and one scalar
`result`. In mojagg, the equivalent operation is a `GUFuncKernel` whose static
signature contains `GUTensor[..., CoreSpec[Dim[0]]]` and
`GUTensor[..., CoreSpec[]]`:

| Gufunc idea | mojagg representation |
| --- | --- |
| Symbolic core name `n` | `Dim[0]` inside a `CoreSpec` |
| Fixed core extent | `FixedDim[size]` |
| Gufunc operand | `GUTensor[element_dtype, output, core_spec]` |
| Gufunc signature | `GUFuncKernel.Signature`, a native `Tuple` of tensors |
| Inner gufunc body | `GUFuncKernel.__call__` |
| Outer iteration and broadcasting | `GUVectorizePlan` plus `guvectorize` |
| Runtime dtype specialization | Mojo compile-time `DType` parameters and native registrations |

The resemblance to Numba is about the data model and call structure. Mojo
does not JIT-compile a Python function at the first call, and it does not
box a dtype or call a Python function for every slice. The binding selects an
AOT-compiled specialization once. The operation receives a native tuple of
typed descriptors and works on raw spans.

The word **inner** in this repository means the work inside one prepared core,
usually the loop in `__call__`. The generic driver does not expose an
`inner_axes` object to the kernel. It converts the selected physical axes into
one contiguous logical span whenever the operation needs one. The word
**outer** means the remaining dimensions and the loop over their broadcast
domain.

## Core dimensions and signatures

`CoreSpec` is compile-time metadata. It declares the ordered dimensions that an
operand has when the kernel is called; it does not contain a runtime shape.

```mojo
CoreSpec[]                    # scalar core; rank 0
CoreSpec[Dim[0]]              # one symbolic dimension
CoreSpec[Dim[0], Dim[1]]      # two ordered symbolic dimensions
CoreSpec[Dim[0], Dim[0]]      # two equal dimensions; a square result
CoreSpec[FixedDim[3]]         # one dimension that must be exactly 3
```

`Dim[id]` is a symbolic extent. Every use of the same `id` in a complete
signature must resolve to the same runtime value. `Dim[0]` and `Dim[1]` are
different dimensions and may have different values. `FixedDim[3]` does not
bind a value from an input; it rejects any runtime core extent other than
three. Symbol IDs are indexes into the fixed `CoreBindings` array and must be
between `0` and `MAX_RANK - 1`.

The complete signature is a single native tuple. Its order, dtype, writable
flag, and `CoreSpec` are all part of the compile-time type:

```mojo
comptime Signature = Tuple[
    GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
    GUTensor[Self.dtype, True, CoreSpec[]],
]
```

This is a `(n)->()` reduction. The first tensor is a read input with a
symbolic one-dimensional core. The second tensor is a writable scalar output.
There is no separate runtime signature string. Both `build_signature_plan` and
`guvectorize` contain a compile-time assertion that the tuple passed by the
binding exactly equals `Operation.Signature`. Swapping two operands or using
the wrong output dtype is therefore a contract error at compile time.

Common signatures in the repository are:

| Operation family | Native signature shape | Meaning |
| --- | --- | --- |
| Scalar reduction | `(n) -> ()` | One selected input core produces one scalar |
| Same-shape transform/fill | `(n) -> (n)` | Read and write cores have the same logical length |
| Grouped reduction | `(n), (n) -> (g)` | Values and labels align; output has one slot per dense group |
| Grouped reduction with state | `(n), (n) -> (g), (g), ...` | Extra writable cores hold counts, best values, or other state |
| Matrix statistic | `(v, o) -> (v, v)` | A variables-by-observations core produces a square core |
| Quantile selection | `(n) -> (q)` | `q` comes from a separate quantile array |

The kernel still receives spans. A two-dimensional core has a flattened length
of `v * o`; the matrix operation keeps `v` and `o` in its operation state so it
can interpret that row-major logical span. The driver uses the two dimensions
to validate and resolve the shape, while the operation owns the numerical
interpretation.

## `GUTensor`: descriptor, not container

`GUTensor[element_dtype, output, core_spec]` is a non-owning descriptor for one
operand. It is deliberately smaller than an ndarray object and does not retain
a Python owner. Its runtime metadata is:

| Field | Meaning |
| --- | --- |
| `address` | Raw base address in the borrowed array or current core |
| `bound` | Whether an address has been installed |
| `shape` | Full tensor shape, in elements |
| `stride` | Full tensor strides, in elements rather than bytes |
| `ndim` | Full tensor rank |
| `length` | Active span length; full element count initially, core length during execution |

The compile-time parameters mean:

* `element_dtype` is the exact dtype loaded from or written to memory.
* `output` is `False` for a read-only input and `True` for a writable output.
* `core_spec` declares the logical dimensions of this operand's core.

The native binding converts a NumPy array once:

```text
shape[axis]  = arr.shape[axis]
stride[axis] = arr.strides[axis] // arr.dtype.itemsize
address      = arr.ctypes.data
```

The divisibility check is performed at the boundary. A negative NumPy stride
remains negative. A zero stride, such as a broadcast view, remains zero. The
Mojo driver therefore sees the actual view layout and does not silently turn
the input into an owning contiguous array.

`borrow(address, shape, stride, ndim)` creates a bound descriptor over caller
owned storage. `empty()` creates an unbound output template. `unbound_contiguous`
creates metadata for a planned output using row-major element strides. The
binding keeps the Python arrays alive for the synchronous native call; the
descriptor itself must never store a `PythonObject` or an owning Python
reference.

`GUTensor` is also `Defaultable`, so `empty_signature[Operation]()` can create
the operation's complete unbound descriptor tuple. Input descriptors are bound
before planning. The planner fills output shape and stride metadata in that
same tuple, and `materialize_outputs` later installs each NumPy address in
place.

Kernels use only these accessors:

```mojo
var values = input.read_span()
var destination = output.write_span()
```

`write_span` and `write_ptr` contain a compile-time assertion that the tensor
is an output. `read_span` and `write_span` use the descriptor's current
`length`, so a kernel must loop over `len(span)` and must not use the original
rank or full array length. The executor copies descriptors once per worker,
then changes only their address and active length for each outer slice.

## Core axes, logical order, and flattening

`AxisSpec` is the runtime counterpart of the core-axis part of a signature. It
contains an inline array of physical axis positions and an explicit `count`.
`input_axes` selects core axes in every read input. `output_axes` selects core
axes in every writable output. They may have different counts and positions.

Physical axis positions are zero based and refer to the original tensor. The
native driver expects them to be normalized, unique, and in range. Python
facades resolve negative axes and reject duplicates before crossing the
boundary.

For an operand with physical shape `s`, element strides `t`, selected axes
`a[0] ... a[k-1]`, and remaining axes `o[0] ... o[m-1]`:

* the logical core shape is `(s[a[0]], ..., s[a[k-1]])`;
* the outer shape is `(s[o[0]], ..., s[o[m-1]])`, preserving physical order;
* a flat outer coordinate has base offset
  `sum(coordinate[j] * t[o[j]])`;
* a logical core coordinate has offset
  `sum(core_coordinate[j] * t[a[j]])`;
* flattening uses the last selected axis as the fastest changing coordinate.

The executor's `outer_offset` uses an odometer over the right-aligned common
outer shape. A strided input scratch copy uses the same last-axis-fastest
odometer over `core_shape` and `core_stride`. This makes the order explicit
even when selected axes are not adjacent in the source array.

### One-dimensional reduction over several axes

Most reduction kernels are intentionally one dimensional:

```mojo
GUTensor[Self.dtype, False, CoreSpec[Dim[0]]]
```

When a facade selects several physical axes, `build_signature_plan` has a special
case: if the operand's core spec has rank one but `input_axes.count` is greater
than one, it binds `Dim[0]` to the product of the selected extents. The kernel
therefore receives one flattened core without needing N-D axis logic.

For `a.shape == (2, 3, 4)` and `input_axes == (1, 2)`:

```text
logical core shape = (3, 4)
core length        = 12
outer shape        = (2,)
outer count        = 2

flat index 0       = a[outer, 0, 0]
flat index 1       = a[outer, 0, 1]
...
flat index 11      = a[outer, 2, 3]
```

The native kernel sees a length-12 span. It does not know that the source was
three by four, and it does not need to. For a reduction this flattening is
usually all that is required. For an operation whose shape matters, use a
multi-dimensional `CoreSpec` and keep the resolved extents in the operation
state, as the matrix kernels do.

Axis order is observable for index-producing operations and for any operation
whose traversal order matters. `_resolve_axes` in the regular value-reduction
facade sorts selected axes by descending byte stride so the smallest stride is
last and the normal C-order core is fastest along its last logical coordinate.
`nanargmin` and `nanargmax` preserve normalized user order because their flat
index is part of the public result. Grouped operations also preserve axis
order. A kernel must not reorder axes in `__call__`.

### Examples of selected axes

For a C-contiguous `(2, 3, 4)` array:

| Selected axes | Outer shape | Logical core shape | Core contiguous? |
| --- | --- | --- | --- |
| `(2,)` | `(2, 3)` | `(4,)` | Yes |
| `(1, 2)` | `(2,)` | `(3, 4)` | Yes |
| `(0, 1)` | `(4,)` | `(2, 3)` | No; the unselected axis is interleaved |
| `(0, 2)` | `(3,)` | `(2, 4)` | No; the unselected axis is interleaved |

The table assumes the axis order shown. Python's stride sorting produces the
contiguous `(1, 2)` order for a normal C-order value reduction. A noncontiguous
selection is still valid for a read input; it takes the scratch path described
below.

## Signature resolution

There are two stages. They should not be confused:

1. `build_signature_plan` or `build_signature_plan_with_bindings` resolves
   symbolic core dimensions, mutates output metadata, and returns the layout
   plan.
2. `materialize_outputs` allocates each output from that metadata and binds the
   NumPy address directly in the same signature tuple.
3. `guvectorize` consumes the bound signature and the existing plan.

### `build_signature_plan`

The normal flow is:

```text
input descriptor(s), already bound
output descriptor(s), unbound templates
             │
             ├─ plan bound inputs and read core extents
             ├─ bind every Dim[id] in CoreBindings
             ├─ check repeated symbols and FixedDim values
             ├─ resolve the input outer broadcast shape
             └─ create contiguous output metadata
```

For each bound input, the planner records the selected core shape and binds
the corresponding `CoreSpec` dimensions. If the same symbolic ID occurs in two
inputs, a different extent raises `named core dimensions must have equal
extents`. A fixed extent mismatch raises `input extent does not match fixed
core dimension`.

The output template must be unbound. Its rank is

```text
resolved common outer rank + output CoreSpec rank
```

The output axes tell the planner where each output core dimension is placed.
Every other output axis receives one dimension from the common outer shape.
For a symbolic output dimension, `resolve_core_dimensions` requires that an
input has already bound its ID. A symbol that exists only on an output cannot
be inferred and raises `output named dimension has no input extent`.

`build_signature_plan` uses input shapes to determine the outer domain. It does not
allow core dimensions to broadcast: core extents are either fixed, symbolically
equal, or, for the one-dimensional reduction case, multiplied into one
flattened extent.

### `build_signature_plan_with_bindings`

Some output dimensions come from a separate runtime object rather than an
input tensor. Quantile is the current example. The quantile array length is
bound before planning:

```text
quantiles.shape == (q,)
CoreBindings[1] = q
output CoreSpec[Dim[1]]
```

The planner then uses that externally supplied value to create an output shape
`outer_shape + (q,)` when `output_axes` places the quantile core last. Use this
entry point only when the value has been validated independently; it is not a
way to bypass core-shape equality checks.

## Outer dimensions and broadcasting

After core axes are removed, every input has an outer shape. The driver
broadcasts these input outer shapes with NumPy's right-aligned rule:

* dimensions are aligned from the trailing side;
* two dimensions are compatible when they are equal or one of them is `1`;
* incompatible dimensions raise `input outer shapes cannot broadcast`;
* a missing leading dimension behaves like a broadcast dimension;
* the common outer shape is the elementwise maximum compatible extent.

Only outer dimensions broadcast. A core dimension of size `1` is still a core
dimension and must satisfy its signature; it is not promoted to a larger core.

For example, select the last axis from two inputs:

```text
x.shape = (5, 1, 3), x core axis = 2  -> x outer shape = (5, 1)
y.shape = (1, 7, 3), y core axis = 2  -> y outer shape = (1, 7)

common outer shape = (5, 7)
core shape         = (3,) for both operands
outer invocations  = 35
```

On an `x` slice whose broadcast coordinate is in the second dimension, the
planner gives `x` an outer stride of zero. The executor reuses the same source
core for all seven positions. The analogous first dimension of `y` also gets
stride zero. This is metadata-level broadcasting; it does not materialize a
`(5, 7, 3)` input.

Different outer ranks work the same way. A vector with shape `(3,)` whose only
axis is the core has outer shape `()`. It can be paired with an input of shape
`(4, 3)` whose last axis is the same core; the vector is reused for all four
outer calls.

Outputs are different. Input shapes define the common domain, and every output
must match that domain exactly in its outer axes. `GUVectorizePlan.build`
rejects an output with a missing outer rank, a singleton that would need to be
expanded, or an extent different from the common shape. Output broadcasting
would create ambiguous write ownership, so outputs are never implicitly
duplicated.

The group facade uses this same idea explicitly before entering Mojo. Public
labels may be shaped like only the selected value axes. `groupby.py` reshapes
them with singleton outer dimensions and calls `np.broadcast_to`, producing a
zero-copy view with the same full rank and shape as the values. The native
group binding can then enforce equal value/label shapes while the generic
driver handles their outer slices.

## Output axes and result layout

`input_axes` and `output_axes` are independent because an operation can change
the core rank or place a result dimension elsewhere.

### Scalar reduction

For `(n)->()`, the input axes contain the selected value axes and the output
axis spec is empty. If the input shape is `(2, 3, 4)` and axes `(1, 2)`, the
planned output shape is `(2,)`.

### Same-shape transform

For `(n)->(n)`, both specs normally select the same physical axis positions.
The output core must be writable-contiguous in the logical order. `ffill` and
`bfill` use this form; when several physical axes are selected, the rank-one
kernel receives their flattened span while the planned output retains those
physical core dimensions at the end. The Python facade moves those output axes
back to their public positions.

### Grouped reduction

The values and labels use `CoreSpec[Dim[0]]` and share the flattened input
length. The dense group output uses `CoreSpec[Dim[1]]`. The binding puts the
group core at the last output axis, so a values outer shape `(2, 4)` and `g ==
3` produce an output shape `(2, 4, 3)`. Each outer call receives a values span,
a labels span, and one writable span of length three. Auxiliary arrays use the
same group core and are written in the same call.

### Matrix operation

The matrix binding accepts any two input axes as `(variables, observations)`
and always places the square output core at the end of the execution result.
Unselected dimensions retain their original order. The operation receives a
`(variables * observations)` span and writes a `(variables * variables)` span.
The `NanCorrOp`/`NanCovOp` operation stores the two input extents so it can map
the flat span into variable rows. The repeated `Dim[0]` in the output
signature enforces the square result.

### Quantile output

The native output is planned with the quantile core last. The Python facade
calls `np.moveaxis(out, -1, 0)` afterward because the public API exposes a
leading quantile axis. This presentation step is outside the generic driver;
the driver itself always follows the `output_axes` metadata it receives.

## Layout, contiguity, and scratch buffers

The operation contract is a contiguous logical core span. The input array as a
whole does not need to be C-contiguous, and its outer dimensions may have
arbitrary strides. The planner checks only the selected core axes.

For a logical core shape `c[0] ... c[k-1]`, the core is considered contiguous
when, from the last core axis to the first, the expected stride is:

```text
expected[last] = 1
expected[previous] = c[last]
expected[previous-1] = c[previous] * c[last]
...
```

The implementation skips stride equality for a core dimension whose extent is
`1`, because no movement along that dimension is observable. A zero-length
core also has no elements to copy or write.

### Read inputs

If a selected read core is contiguous in that logical order, the executor
points the local descriptor directly at the source address. If it is strided,
reversed, transposed, or broadcast within the core, the executor copies only
that core into a worker-local scratch span. The copy uses the selected axis
order and last-axis-fastest odometer, so the kernel sees the same logical order
for every source layout.

There is no full-array `ascontiguousarray` and no per-slice heap allocation.
Scratch is allocated once per native call, with one aligned block per worker.
The block contains a region for each noncontiguous read input. The offset of
each region is aligned upward to 64 bytes. The same worker reuses its regions
for every outer slice it processes.

### Writable outputs

The driver cannot scatter a normal contiguous span back into an arbitrary
strided output core, so a writable selected core must already satisfy the
logical contiguity rule. Otherwise planning raises `writable core must be
contiguous`. There is no output scratch and no copy-back step.

The generic Mojo driver permits arbitrary outer strides as long as the selected
output core is writable-contiguous and the output outer shape matches the
common domain. The binding-level materialization helper allocates each writable
operand from the planned shape and binds its NumPy address before
`guvectorize` runs. It also keeps the Python owners alive for the synchronous
call. Direct Mojo tests can exercise the lower-level outer-stride contract.

The descriptor stores strides in elements. All byte-to-element conversion is
performed in the binding, and the executor multiplies an outer element offset
by `size_of[Scalar[dtype]]()` only when forming the slice address.

## Execution and worker scheduling

`build_signature_plan` performs the shape and layout checks before output
materialization:

1. Assert that the tensor tuple equals `Operation.Signature`.
2. Build an `OperandPlan` for every tensor.
3. Resolve and validate the common input outer shape.
4. Reject invalid output ranks, shapes, or writable core strides.

`guvectorize` then consumes that plan:

1. Assert that the tensor tuple equals `Operation.Signature`.
2. Return immediately when `outer_count == 0`.
3. Compute the largest read core length.
4. Choose one worker or a parallel worker count from `DispatchPolicy`.
5. Allocate one worker-local scratch block if any read core needs copying.
6. Iterate the flat outer domain and call the operation once per position.
7. Free the scratch block after all slices complete.

The executor does not split one core across workers. Parallelism is over outer
invocations. A single enormous core therefore remains one operation call on
one worker unless the caller has multiple outer slices.

The effective worker rule is:

```text
requested = configured workers, or 16 when workers <= 0
serial if outer_count <= 1 or requested <= 1
serial if outer_count < max(parallel_min_groups, 1)
serial if largest_read_core < max(parallel_threshold, 0)
parallel workers = min(requested, outer_count)
```

The threshold comparisons are inclusive at the boundary: exactly
`parallel_min_groups` outer slices and exactly `parallel_threshold` elements
pass their respective gates. A new threshold or dispatch branch requires
benchmark evidence.

In the serial path the executor copies the operation once and uses worker id
zero. In the parallel path it divides the flat outer range into contiguous
ceil-divided chunks and copies the operation once per worker. Each worker also
has its own tensor descriptors and scratch region. This matters for stateful
operations: mutable operation fields and scratch must not be shared between
workers.

`NanQuantileKernel` is the current explicit workspace example. Its copy
constructor creates an independent sortable buffer for every worker. Ordinary
streaming kernels should not allocate in their hot loop. Group scatter kernels
write only their own outer output slice; the generic driver does not parallelize
different group labels inside a slice.

## Native Python binding responsibilities

The native binding is the boundary between Python objects and the typed driver.
It is responsible for:

1. Checking the exact input and output dtypes selected by the compiled
   specialization.
2. Reading `ndim`, shape, byte strides, itemsize, and `ctypes.data` once.
3. Converting byte strides to element strides.
4. Building `GUTensor` descriptors with the exact compile-time core specs.
5. Turning the normalized Python axis tuple into `AxisSpec`.
6. Calling `build_signature_plan` when output shape is derived from inputs, or
   `build_signature_plan_with_bindings` when an external dimension is supplied.
7. Calling `materialize_outputs` to allocate outputs and bind their addresses.
8. Constructing the dispatch policy from `MojaggConfig`.
9. Entering `guvectorize` exactly once.

For a scalar reduction, `_apply_reduction` follows this shape:

```text
arr, axes
  │
  ├─ input: GUTensor[value_dtype, False, CoreSpec[Dim[0]]]
  ├─ output template: GUTensor[out_dtype, True, CoreSpec[]].empty()
  ├─ build_signature_plan[Operation](signature, axes, empty output axes)
  ├─ materialize_outputs(signature): allocate and bind NumPy outputs in place
  └─ guvectorize(operation, signature, plan, policy)
```

Grouped bindings use the tuple `(values, labels, group_output, auxiliaries...)`.
The facade has already validated labels, computed `num_labels`, and broadcast
labels to the values shape. The binding seeds the group dimension, plans and
materializes the result and workspaces, initializes them, then gives values and
labels `CoreSpec[Dim[0]]`, every group result and workspace `CoreSpec[Dim[1]]`,
and puts the output core at the last output axis.

The binding must keep all Python arrays alive for the duration of the call.
Because the call is synchronous, local `PythonObject` arguments and auxiliary
objects are sufficient. Never store a Python object in a long-lived kernel or
descriptor.

The public facade remains responsible for behavior that belongs to the Python
API: dtype promotion, endian normalization, axis normalization,
quantile-axis presentation, public exceptions, and compatibility with numbagg.
The typed binding owns output shape resolution and NumPy allocation after the
signature has been planned. The driver must not grow Python-facing branches to
handle those concerns.

## Public axis and promotion behavior

The regular reduction facade turns `axis` into a tuple before entering Mojo:

* `axis=None` selects every physical axis;
* an integer becomes a one-element tuple;
* a tuple is normalized for negative positions and duplicate/bounds errors;
* the current `axis=()` compatibility behavior treats the reduction as a
  flattening reduction over every axis;
* value reductions sort by descending byte stride;
* arg reductions preserve the normalized tuple order.

Zero-dimensional public inputs are reshaped to a one-element view so the
reduction binding can pass a normal one-axis core. This is a view and does not
copy the value.

The facade may make an explicit copy for documented dtype promotion or
big-endian normalization. It does not add `moveaxis`, a full
`ascontiguousarray`, or a Python loop over outer slices. All ordinary input
layout handling after that point comes from the descriptors and the driver.

Grouped operations have additional public rules before the native call:

* labels must be native-endian `int32` or `int64`;
* negative labels are ignored by group kernels;
* `num_labels` determines the dense output width and must cover the largest
  nonnegative label;
* `axis=None` requires values and labels to have the same shape;
* labels shaped like only the selected axes are reshaped and broadcast as a
  zero-copy view over the remaining value axes.

These rules explain why a native binding should not be called directly from a
new Python wrapper without reusing the existing facade patterns.

## Writing a new `GUFuncKernel`

Start by choosing the smallest existing core contract. A normal scalar
reduction should use `(n)->()`, a same-shape operation `(n)->(n)`, and a
grouped operation the existing `(n),(n)->(g)` arrangement. Do not put rank or
axis traversal into a kernel just because a public function accepts N-D input.
The driver already turns that N-D call into prepared core spans.

A minimal reduction operation has this structure:

```mojo
@fieldwise_init
struct ExampleReduce[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0]]],
        GUTensor[Self.dtype, True, CoreSpec[]],
    ]

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        var values = input.read_span()
        var destination = output.write_span()
        # Compute one result from the prepared values span.
        destination[0] = ...
```

The checked-in [`nansum.mojo`](../src/mojagg/nanfuncs/nansum.mojo) is the
complete SIMD example. Its operation does not inspect `ndim`, axes, shape,
strides, Python objects, or worker count. It uses `len(values)` for the active
core and handles SIMD tails with the `(i, evl)` vectorize callback. Floating
and integer behavior is separated with `comptime if`, so integer
specializations contain no NaN branch.

Kernel requirements:

* Declare the exact `Signature`, including tuple order and output flags.
* Read and write only through `read_span()` and `write_span()` or their pointer
  accessors.
* Treat the input span as already ordered and contiguous.
* Handle `len(span) == 0` according to the operation's identity or error
  semantics; never assume element zero exists for a read input.
* Keep output writes inside the current core. The executor supplies a distinct
  output slice for each outer position.
* Use explicit SIMD and masked tails where the operation permits it.
* Keep stateful fields copyable per worker. A kernel-owned allocation requires
  the current copy-constructor and `__deinit__` pattern.
* Do not allocate, call Python, normalize axes, or read mutable configuration
  in the core loop.

For a multi-output kernel, put every input and output in one tuple and use the
same tuple in the binding and `Signature`. `tests/mojo/test_gufunc.mojo` has a
heterogeneous two-input/two-output operation and is the reference for this
case.

## Choosing a driver shape

Use these patterns before inventing a new driver:

| Need | Use |
| --- | --- |
| N-D reduction to a scalar per outer position | `CoreSpec[Dim[0]] -> CoreSpec[]`; flatten selected axes when needed |
| Elementwise or stateful scan along an axis | matching `CoreSpec[Dim[0]]` input/output |
| Values grouped by aligned labels | two matching `Dim[0]` inputs and `Dim[1]` output |
| Additional per-group state | more writable `CoreSpec[Dim[1]]` tensors |
| Fixed-size mathematical core | `FixedDim[size]` or repeated symbolic dimensions |
| Matrix statistic | two input core symbols and a square output core |
| Output length from a separate array | `build_signature_plan_with_bindings` |

A new driver is justified only when the shared contract cannot express the
operation's state or core layout. First add a tested shared contract if that
is the case; do not move axis or broadcast logic into every new kernel.

## Testing the contract

The native driver tests are in
[`tests/mojo/test_gufunc.mojo`](../tests/mojo/test_gufunc.mojo). They cover:

* a scalar reduction and SIMD-width input;
* heterogeneous dtypes and multiple inputs/outputs;
* a stride-two read core copied into scratch;
* rejection of a noncontiguous writable core;
* all pairs of selected axes in a three-dimensional array and their outer
  ordering;
* the independent outer-count and inner-core parallel thresholds;
* one operation copy per selected worker.

A new or changed kernel should add public parity tests under `tests/python/`
against an unpatched numbagg reference wherever an equivalent exists. Include
empty and zero-sized cores, empty outer dimensions, all-NaN and mixed values,
SIMD tails around several widths, reversed and strided views, transposed views,
broadcast views, `axis=None`, negative and tuple axes, result shapes, result
dtypes, and exception types. Grouped operations also need negative labels,
empty groups, label broadcasting, repeated labels, and `ddof` boundaries.

The driver behavior is complete only when the public facade, native tuple,
signature planner, operation, tests, and benchmark all describe the same
contract.

## Debugging checklist

When a new operation fails, inspect the boundary in this order:

1. Is the Python facade passing a normalized, unique, in-range axis tuple?
2. Do the input and output dtypes exactly match the registered specialization?
3. Does the tuple order exactly match `Operation.Signature`?
4. Does every `Dim[id]` bind to the expected extent, including repeated IDs?
5. Are the selected core axes in the intended logical order?
6. Does the input outer shape broadcast to the expected common domain?
7. Does the output have exactly that outer shape and a contiguous selected core?
8. Is a read core being copied to scratch because its selected strides are not
   row-major in logical core order?
9. Is the operation reading `len(span)` rather than full tensor metadata?
10. Does a stateful operation have independent worker copies?
11. Is a parallel result race caused by shared auxiliary or output storage?
12. Do direct parity tests compare against numbagg before any test fixture has
    registered mojagg as the reference implementation?

Most failures are caused by one of these boundaries being crossed in the wrong
layer. Keep public semantics in Python, layout planning in the driver, and
numeric work in the kernel.
