"""NumPy allocation and binding for planned GUFunc signatures.

The generic guvectorize planner only produces borrowed tensor metadata.  This
module is the Python boundary that turns unbound output descriptors into
NumPy-owned storage and installs each address before execution.
"""

from std.python import Python, PythonObject

from mojagg.drivers.guvectorize import (
    AnyGUTensor,
    CoreSpecProtocol,
    GUTensor,
    GUFuncKernel,
)


def _numpy_dtype[dtype: DType](numpy: PythonObject) raises -> PythonObject:
    """Map a native output type to its matching NumPy scalar type."""

    comptime if dtype == DType.bool:
        return numpy.bool_
    elif dtype == DType.float32:
        return numpy.float32
    elif dtype == DType.float64:
        return numpy.float64
    elif dtype == DType.int32:
        return numpy.int32
    elif dtype == DType.int64:
        return numpy.int64
    else:
        raise Error("unsupported NumPy output dtype")


def _allocate_output[
    dtype: DType,
    core: CoreSpecProtocol,
](planned: GUTensor[dtype, True, core]) raises -> PythonObject:
    """Allocate one NumPy array using the planned output shape."""

    var numpy = Python.import_module("numpy")
    var shape = Python.list()
    for axis in range(planned.ndim):
        shape.append(planned.shape[axis])
    if planned.ndim == 0:
        return numpy.empty(Python.tuple(), dtype=_numpy_dtype[dtype](numpy))
    return numpy.empty(shape, dtype=_numpy_dtype[dtype](numpy))


def materialize_outputs[
    Operation: GUFuncKernel,
    *Args: AnyGUTensor,
](mut signature: Tuple[*Args]) raises -> List[PythonObject]:
    """Allocate and bind every output in an already planned signature.

    The descriptor tuple is mutated in place. Each returned Python owner keeps
    the corresponding NumPy allocation alive while the native operation runs;
    no tensor or descriptor copy is needed.
    """

    comptime assert (
        Operation.Signature == Tuple[*Args]
    ), "signature does not match the operation signature"

    var outputs = List[PythonObject]()

    comptime for i in range(len(Args)):
        comptime if Args[i].is_output:
            var output = rebind[
                GUTensor[
                    Args[i].dtype,
                    True,
                    Args[i].core_spec,
                ]
            ](signature[i])
            var output_arr = _allocate_output[
                Args[i].dtype,
                Args[i].core_spec,
            ](output)
            signature[i].bind_address(
                Int(py=output_arr.ctypes.data),
                output.length,
            )
            outputs.append(output_arr)

    return outputs^
