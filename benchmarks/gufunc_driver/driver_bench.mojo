"""Prototype gufunc-driver variants for the driver-design benchmark.

NOT production code. Measures four ways to implement the numba.guvectorize
outer loop in Mojo, all wrapping the same masked-SIMD sum kernel:

1. rows_f64        — flat per-row binding (current mojagg approach: one FFI
                     call per outer slice, driven by a Python for-loop).
2. contig_f64      — single FFI call; Mojo loops outer rows of a C-contiguous
                     2-D array, SIMD kernel per row.
3. contig_par_f64  — (2) + std.algorithm.parallelize over rows.
4. strided_f64     — single FFI call; fully general N-D odometer over
                     (shape, strides), no copies; contiguous fast path when
                     the reduced axis has stride 1.
5. view_f64        — NuMojo-style: per outer slice, construct a heap-backed
                     view struct (shape/strides Lists) and reduce through it
                     (lower-bound emulation of NDArray slice construction).
"""

from max.algorithm import parallelize
from std.collections import List, Span
from std.math import isnan
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime ITEMSIZE = 8  # float64


# --- kernels ---------------------------------------------------------------


@always_inline
def _sum_row_contig(
    ptr: Pointer[mut=True, Float64, MutAnyOrigin], n: Int
) -> Float64:
    """Masked SIMD sum over a contiguous row (same pattern as nansum_core)."""
    var acc = SIMD[DType.float64, W](0.0)
    var zero = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var v = ptr.unsafe_load[width=W](i)
        acc += v.eq(v).select(v, zero)
        i += W
    var total = acc.reduce_add()
    while i < n:
        var v = ptr[unsafe_offset=i]
        if v == v:
            total += v
        i += 1
    return total


@always_inline
def _sum_row_strided(
    ptr: Pointer[mut=True, Float64, MutAnyOrigin], n: Int, stride: Int
) -> Float64:
    """Scalar strided walk with per-element NaN check."""
    var total = Float64(0.0)
    var off = 0
    for _ in range(n):
        var v = ptr[unsafe_offset=off]
        if not isnan(v):
            total += v
        off += stride
    return total


@always_inline
def _sum_row(
    ptr: Pointer[mut=True, Float64, MutAnyOrigin], n: Int, stride: Int
) -> Float64:
    # Branch once per row, never per element (SKILL.md §3.6).
    if stride == 1:
        return _sum_row_contig(ptr, n)
    return _sum_row_strided(ptr, n, stride)


# --- validation helpers -----------------------------------------------------


def _data_ptr(arr: PythonObject, op: String) raises -> Pointer[
    mut=True, Float64, MutAnyOrigin
]:
    var got = String(py=arr.dtype.name)
    if got != "float64":
        raise Error(op + ": expected dtype float64, got " + got)
    return Pointer[mut=True, Float64, MutAnyOrigin](
        unsafe_from_address=Int(py=arr.ctypes.data)
    )


def _validated_2d_contig(
    arr: PythonObject, op: String
) raises -> Tuple[Pointer[mut=True, Float64, MutAnyOrigin], Int, Int]:
    if Int(py=arr.ndim) != 2:
        raise Error(op + ": expected a 2-D C-contiguous float64 array")
    if not Bool(py=arr.flags.c_contiguous):
        raise Error(op + ": array must be C-contiguous")
    return (
        _data_ptr(arr, op),
        Int(py=arr.shape[0]),
        Int(py=arr.shape[1]),
    )


# --- variant 1: per-row flat binding (current mojagg approach) --------------


def rows_f64(arr: PythonObject) raises -> PythonObject:
    """Sum one 1-D contiguous row; the Python caller loops rows."""
    if Int(py=arr.ndim) != 1:
        raise Error("rows_f64: expected a 1-D contiguous array")
    if not Bool(py=arr.flags.c_contiguous):
        raise Error("rows_f64: array must be C-contiguous")
    var data = _data_ptr(arr, "rows_f64")
    return PythonObject(_sum_row_contig(data, Int(py=arr.size)))


# --- variant 2: single-call contiguous driver --------------------------------


def contig_f64(arr: PythonObject, out_arr: PythonObject) raises -> PythonObject:
    """Reduce axis -1 of a C-contiguous 2-D array into out[outer]."""
    var t = _validated_2d_contig(arr, "contig_f64")
    var data = t[0]
    var outer = t[1]
    var n = t[2]
    var out_ptr = _data_ptr(out_arr, "contig_f64.out")
    for o in range(outer):
        out_ptr[unsafe_offset=o] = _sum_row_contig(data.unsafe_offset(o * n), n)
    return out_arr


# --- variant 3: contiguous driver + parallelize over rows --------------------


def contig_par_f64(
    arr: PythonObject, out_arr: PythonObject
) raises -> PythonObject:
    """Variant 2 with the outer loop parallelized across threads."""
    var t = _validated_2d_contig(arr, "contig_par_f64")
    var data = t[0]
    var outer = t[1]
    var n = t[2]
    var out_ptr = _data_ptr(out_arr, "contig_par_f64.out")

    @parameter
    def row_body(o: Int):
        out_ptr[unsafe_offset=o] = _sum_row_contig(data.unsafe_offset(o * n), n)

    parallelize[row_body](outer)
    return out_arr


# --- variant 4: fully general strided odometer driver (any axis, no copy) ----


def strided_f64(
    arr: PythonObject, out_arr: PythonObject, axis_obj: PythonObject
) raises -> PythonObject:
    """Reduce `axis` of an N-D array into out; strides handled, never copies."""
    var ndim = Int(py=arr.ndim)
    var axis = Int(py=axis_obj)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise Error("strided_f64: axis out of range")
    var data = _data_ptr(arr, "strided_f64")
    var out_ptr = _data_ptr(out_arr, "strided_f64.out")

    var n = Int(py=arr.shape[axis])
    var inner_stride = Int(py=arr.strides[axis]) // ITEMSIZE

    var osizes = List[Int]()
    var ostrides = List[Int]()
    var outer_count = 1
    for d in range(ndim):
        if d == axis:
            continue
        var sz = Int(py=arr.shape[d])
        osizes.append(sz)
        ostrides.append(Int(py=arr.strides[d]) // ITEMSIZE)
        outer_count *= sz

    # Odometer over outer dims: counters[] tick like a positional numeral,
    # base offset updated incrementally — O(1) amortized per slice.
    var k = len(osizes)
    var counters = List[Int](length=k, fill=0)
    var base = 0
    for o in range(outer_count):
        out_ptr[unsafe_offset=o] = _sum_row(data.unsafe_offset(base), n, inner_stride)
        var d = k - 1
        while d >= 0:
            counters[d] += 1
            base += ostrides[d]
            if counters[d] < osizes[d]:
                break
            counters[d] = 0
            base -= osizes[d] * ostrides[d]
            d -= 1
    return out_arr


# --- variant 4b: strided odometer driver + parallelize -----------------------


def strided_par_f64(
    arr: PythonObject, out_arr: PythonObject, axis_obj: PythonObject
) raises -> PythonObject:
    """Variant 4 with the outer loop parallelized.

    Each worker takes a contiguous chunk of the flat outer index space and
    fast-forwards the odometer to its chunk start via divmod decomposition.
    """
    var ndim = Int(py=arr.ndim)
    var axis = Int(py=axis_obj)
    if axis < 0:
        axis += ndim
    var data = _data_ptr(arr, "strided_par_f64")
    var out_ptr = _data_ptr(out_arr, "strided_par_f64.out")

    var n = Int(py=arr.shape[axis])
    var inner_stride = Int(py=arr.strides[axis]) // ITEMSIZE

    var osizes = List[Int]()
    var ostrides = List[Int]()
    var outer_count = 1
    for d in range(ndim):
        if d == axis:
            continue
        var sz = Int(py=arr.shape[d])
        osizes.append(sz)
        ostrides.append(Int(py=arr.strides[d]) // ITEMSIZE)
        outer_count *= sz

    var k = len(osizes)
    var num_workers = min(16, outer_count)
    var chunk = (outer_count + num_workers - 1) // num_workers

    @parameter
    def worker(w: Int):
        var start = w * chunk
        var end = min(start + chunk, outer_count)
        if start >= end:
            return
        # Fast-forward: decompose flat index into per-dim counters + offset.
        var counters = List[Int](length=k, fill=0)
        var base = 0
        var rem = start
        for d in range(k - 1, -1, -1):
            var q = rem % osizes[d]
            counters[d] = q
            base += q * ostrides[d]
            rem //= osizes[d]
        for o in range(start, end):
            out_ptr[unsafe_offset=o] = _sum_row(
                data.unsafe_offset(base), n, inner_stride
            )
            var d = k - 1
            while d >= 0:
                counters[d] += 1
                base += ostrides[d]
                if counters[d] < osizes[d]:
                    break
                counters[d] = 0
                base -= osizes[d] * ostrides[d]
                d -= 1

    parallelize[worker](num_workers)
    return out_arr


# --- variant 5: NuMojo-style per-slice view construction ----------------------

struct _RowView[o: Origin[mut=True]]:
    """Lower-bound emulation of an NDArray slice view: heap-backed metadata."""

    var data: Pointer[mut=True, Float64, Self.o]
    var shape: List[Int]
    var strides: List[Int]

    def __init__(
        out self,
        data: Pointer[mut=True, Float64, Self.o],
        var shape: List[Int],
        var strides: List[Int],
    ):
        self.data = data
        self.shape = shape^
        self.strides = strides^


def view_f64(arr: PythonObject, out_arr: PythonObject) raises -> PythonObject:
    """Like variant 2, but each row goes through a freshly constructed view."""
    var t = _validated_2d_contig(arr, "view_f64")
    var data = t[0]
    var outer = t[1]
    var n = t[2]
    var out_ptr = _data_ptr(out_arr, "view_f64.out")
    for o in range(outer):
        var view = _RowView(
            data.unsafe_offset(o * n),
            List[Int](length=1, fill=n),
            List[Int](length=1, fill=1),
        )
        var total = Float64(0.0)
        var stride = view.strides[0]
        var off = 0
        for _ in range(view.shape[0]):
            var v = view.data[unsafe_offset=off]
            if not isnan(v):
                total += v
            off += stride
        out_ptr[unsafe_offset=o] = total
    return out_arr


# --- dispatch-cost probes (isolate parallelize pool overhead) -----------------


def noop_par_fine(iters_obj: PythonObject) raises -> PythonObject:
    """parallelize over N no-op tasks — per-task + fixed dispatch cost."""

    var iters = Int(py=iters_obj)

    @parameter
    def body(i: Int):
        pass

    parallelize[body](iters)
    return iters_obj


def noop_par_chunked(iters_obj: PythonObject) raises -> PythonObject:
    """parallelize over 16 no-op tasks — near-pure fixed dispatch cost."""

    @parameter
    def body(i: Int):
        pass

    parallelize[body](16)
    return iters_obj


def noop_serial(iters_obj: PythonObject) raises -> PythonObject:
    """Serial baseline loop (no pool involvement)."""
    var iters = Int(py=iters_obj)
    var acc = 0
    for i in range(iters):
        acc += i
    return PythonObject(acc)


@export
def PyInit_gufunc_bench() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("gufunc_bench")
        m.def_function[rows_f64]("rows_f64")
        m.def_function[contig_f64]("contig_f64")
        m.def_function[contig_par_f64]("contig_par_f64")
        m.def_function[strided_f64]("strided_f64")
        m.def_function[strided_par_f64]("strided_par_f64")
        m.def_function[view_f64]("view_f64")
        m.def_function[noop_par_fine]("noop_par_fine")
        m.def_function[noop_par_chunked]("noop_par_chunked")
        m.def_function[noop_serial]("noop_serial")
        return m.finalize()
    except e:
        abort(String("failed to create gufunc_bench module: ", e))
