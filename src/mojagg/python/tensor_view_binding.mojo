"""NumPy boundary constructors for ``mojagg.core.tensor_view``.

The tensor-view core intentionally has no Python dependency so the same typed
runtime layouts can be reused by a future device binding.  These two helpers
perform the one boundary-only task that depends on NumPy: converting byte
strides to signed element strides and borrowing the array address.
"""

from std.python import PythonObject

from mojagg.core.tensor_view import (
    DimArray,
    MAX_RANK,
    ReadTensor,
    TensorArg,
    WriteTensor,
)


def read_tensor_from_numpy[
    dtype: DType
](arr: PythonObject) raises -> ReadTensor[dtype]:
    """Borrow a typed read tensor from a NumPy array without copying data."""
    var rank = Int(py=arr.ndim)
    if rank < 0 or rank > MAX_RANK:
        raise Error("tensor rank exceeds MAX_RANK=" + String(MAX_RANK))
    var shape = DimArray(fill=1)
    var strides = DimArray(fill=0)
    var itemsize = Int(py=arr.dtype.itemsize)
    if itemsize <= 0:
        raise Error("tensor itemsize must be positive")
    for d in range(rank):
        shape[d] = Int(py=arr.shape[d])
        var byte_stride = Int(py=arr.strides[d])
        if byte_stride % itemsize != 0:
            raise Error("array stride is not divisible by its itemsize")
        strides[d] = byte_stride // itemsize
    var layout = TensorArg[dtype, False].runtime_layout(shape, strides)
    var pointer = Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin](
        unsafe_from_address=Int(py=arr.ctypes.data)
    )
    return ReadTensor[dtype](pointer, layout)


def write_tensor_from_numpy[
    dtype: DType
](arr: PythonObject) raises -> WriteTensor[dtype]:
    """Borrow a typed writable tensor from a caller-owned NumPy output."""
    var rank = Int(py=arr.ndim)
    if rank < 0 or rank > MAX_RANK:
        raise Error("tensor rank exceeds MAX_RANK=" + String(MAX_RANK))
    var shape = DimArray(fill=1)
    var strides = DimArray(fill=0)
    var itemsize = Int(py=arr.dtype.itemsize)
    if itemsize <= 0:
        raise Error("tensor itemsize must be positive")
    for d in range(rank):
        shape[d] = Int(py=arr.shape[d])
        var byte_stride = Int(py=arr.strides[d])
        if byte_stride % itemsize != 0:
            raise Error("array stride is not divisible by its itemsize")
        strides[d] = byte_stride // itemsize
    var layout = TensorArg[dtype, True].runtime_layout(shape, strides)
    var pointer = Pointer[mut=True, Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=Int(py=arr.ctypes.data)
    )
    return WriteTensor[dtype](pointer, layout)
