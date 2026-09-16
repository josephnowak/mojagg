"""Shared helpers for the native Python bindings.

The bindings all cross the same boundary: read normalized axes, borrow a
NumPy buffer as a typed ``GUTensor``, validate the registered dtype, and turn
the Python configuration into a dispatch policy. Keeping those conversions in
one module makes the operation bindings describe only their signatures.
"""

from std.python import PythonObject

from mojagg.core.numeric import _dtype_name
from mojagg.drivers.guvectorize import (
    AxisSpec,
    CoreSpecProtocol,
    DimArray,
    DispatchPolicy,
    GUTensor,
    MAX_RANK,
)


def dispatch_policy_from_py(cfg: PythonObject) raises -> DispatchPolicy:
    """Read a ``MojaggConfig`` or a legacy scalar threshold."""

    try:
        var threshold = Int(py=cfg.parallel_threshold)
        var workers = Int(py=cfg.threads)
        var groups = Int(py=cfg.parallel_min_groups)
        return DispatchPolicy(workers, threshold, groups)
    except:
        return DispatchPolicy(0, Int(py=cfg), 1)


def axes_from_py(axes: PythonObject, op: String) raises -> AxisSpec:
    """Read the normalized Python axes tuple into the native axis contract."""

    var count = Int(py=axes.__len__())
    if count < 1 or count > MAX_RANK:
        raise Error(
            op
            + ": expected 1.."
            + String(MAX_RANK)
            + " axes, got "
            + String(count)
        )

    var values = DimArray(fill=0)
    for i in range(count):
        values[i] = Int(py=axes[i])
    return AxisSpec(values^, count)


def matrix_axes_from_py(
    axes: PythonObject, ndim: Int, op: String
) raises -> AxisSpec:
    """Read and validate the two selected matrix axes."""

    if ndim < 2:
        raise Error(op + " requires at least two dimensions")
    var parsed = axes_from_py(axes, op)
    if parsed.count != 2:
        raise Error(op + ": expected exactly two axes")
    for i in range(2):
        var axis = parsed[i]
        if axis < 0 or axis >= ndim:
            raise Error(op + ": matrix axis is out of bounds")
        for j in range(i):
            if axis == parsed[j]:
                raise Error(op + ": matrix axes must be unique")
    return parsed^


def validate_dtype[dtype: DType](arr: PythonObject, op: String) raises:
    """Require the exact dtype registered for the native specialization."""

    var expected = _dtype_name[dtype]()
    var actual = String(py=arr.dtype.name)
    if expected != actual:
        raise Error(op + ": expected dtype " + expected + ", got " + actual)


def borrow_numpy_tensor[
    dtype: DType,
    writable: Bool,
    core: CoreSpecProtocol,
](arr: PythonObject) raises -> GUTensor[dtype, writable, core]:
    """Borrow NumPy shape, element strides, and address without copying."""

    var rank = Int(py=arr.ndim)
    if rank < 0 or rank > MAX_RANK:
        raise Error("tensor rank exceeds guvectorize capacity")

    var shape = DimArray(fill=1)
    var stride = DimArray(fill=0)
    var itemsize = Int(py=arr.dtype.itemsize)
    if itemsize <= 0:
        raise Error("tensor itemsize must be positive")
    for axis in range(rank):
        shape[axis] = Int(py=arr.shape[axis])
        var byte_stride = Int(py=arr.strides[axis])
        if byte_stride % itemsize != 0:
            raise Error("array stride is not divisible by its itemsize")
        stride[axis] = byte_stride // itemsize

    return GUTensor[dtype, writable, core].borrow(
        Int(py=arr.ctypes.data),
        shape,
        stride,
        rank,
    )
