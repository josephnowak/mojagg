"""Python bindings for the nanfuncs family — THIN layer.

The binding exposes only flat, contiguous 1-D reductions. All axis handling
(moveaxis, per-slice looping, output assembly) lives in the Python facade
(`python/mojagg/nanfuncs.py`), which calls `_nansum_flat` per slice. This keeps
Mojo kernels pure and the N-D/axis logic in one testable place.

Zero-copy contract (SKILL.md §3.1): we read arr.ctypes.data directly and never
allocate or copy; unsupported dtype / non-contiguous input raises here.
"""

from std.collections import Span
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from mojagg.nanfuncs.nansum import nansum_f64, nansum_f32, sum_i64, sum_i32


@always_inline
def _validated_span[
    dtype: DType
](arr: PythonObject, op: String) raises -> Span[Scalar[dtype], MutAnyOrigin]:
    """Validate a contiguous 1-D numpy array and return a borrowed Span.

    Raises unless: 1-D, C-contiguous, and dtype matches the requested `dtype`.
    """
    if Int(py=arr.ndim) != 1:
        raise Error(op + ": expected a 1-D contiguous array")
    if not Bool(py=arr.flags.c_contiguous):
        raise Error(op + ": array must be C-contiguous")
    var want = "float64" if dtype == DType.float64 else "float32"
    var got = String(py=arr.dtype.name)
    if got != want:
        raise Error(op + ": expected dtype " + want + ", got " + got)
    var n = Int(py=arr.size)
    var ptr = Pointer[mut=True, Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(py=arr.ctypes.data)
    )
    return Span[Scalar[dtype], MutAnyOrigin](unsafe_ptr=ptr, length=n)


def nansum_f64_flat(arr: PythonObject) raises -> PythonObject:
    """Sum non-NaN elements of a 1-D float64 array (scalar out)."""
    return PythonObject(nansum_f64(_validated_span[DType.float64](arr, "nansum")))


def nansum_f32_flat(arr: PythonObject) raises -> PythonObject:
    """Sum non-NaN elements of a 1-D float32 array (scalar out)."""
    return PythonObject(nansum_f32(_validated_span[DType.float32](arr, "nansum")))


def _validated_int_span[
    dtype: DType
](arr: PythonObject, op: String) raises -> Span[Scalar[dtype], MutAnyOrigin]:
    if Int(py=arr.ndim) != 1:
        raise Error(op + ": expected a 1-D contiguous array")
    if not Bool(py=arr.flags.c_contiguous):
        raise Error(op + ": array must be C-contiguous")
    var want = "int64" if dtype == DType.int64 else "int32"
    var got = String(py=arr.dtype.name)
    if got != want:
        raise Error(op + ": expected dtype " + want + ", got " + got)
    var n = Int(py=arr.size)
    var ptr = Pointer[mut=True, Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(py=arr.ctypes.data)
    )
    return Span[Scalar[dtype], MutAnyOrigin](unsafe_ptr=ptr, length=n)


def sum_i64_flat(arr: PythonObject) raises -> PythonObject:
    """Sum of a 1-D int64 array (no NaN concept)."""
    return PythonObject(sum_i64(_validated_int_span[DType.int64](arr, "sum")))


def sum_i32_flat(arr: PythonObject) raises -> PythonObject:
    """Sum of a 1-D int32 array (no NaN concept)."""
    return PythonObject(sum_i32(_validated_int_span[DType.int32](arr, "sum")))


@export
def PyInit_nanfuncs_native() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("nanfuncs_native")
        m.def_function[nansum_f64_flat]("nansum_f64_flat")
        m.def_function[nansum_f32_flat]("nansum_f32_flat")
        m.def_function[sum_i64_flat]("sum_i64_flat")
        m.def_function[sum_i32_flat]("sum_i32_flat")
        return m.finalize()
    except e:
        abort(String("failed to create nanfuncs_native module: ", e))
