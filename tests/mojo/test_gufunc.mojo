"""Behavioral tests for the native-tuple GUFunc driver."""

from std.atomic import Atomic
from std.memory import alloc, dealloc
from std.memory.alloc import Layout
from std.testing import assert_equal, assert_raises

from mojagg.core.tensor_view import DimArray, TensorArg, DType
from mojagg.drivers.gufunc import DispatchPolicy, GUFunc, GUFuncOperation
from mojagg.nanfuncs.nansum import NanSum


struct MixedTupleTransform(GUFuncOperation, ImplicitlyCopyable):
    """Exercise a heterogeneous two-input/two-output native tuple.

    The second input deliberately has a stride-two core.  The operation still
    receives two one-dimensional contiguous read spans because the GUFunc
    materializes that input into its worker scratch area before ``apply``.
    """

    comptime Tensors = Tuple[
        TensorArg[DType.float32, False],
        TensorArg[DType.float64, False],
        TensorArg[DType.float64, True],
        TensorArg[DType.float32, True],
    ]

    def __init__(out self):
        pass

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var left_arg = tensors[0].copy()
        var right_arg = tensors[1].copy()
        var first_arg = tensors[2].copy()
        var second_arg = tensors[3].copy()
        var left = left_arg.read_span()
        var right = right_arg.read_span()
        var first = first_arg.write_span()
        var second = second_arg.write_span()
        for i in range(len(left)):
            var value = (
                Float64(left.unsafe_ptr()[unsafe_offset=i])
                + right.unsafe_ptr()[unsafe_offset=i]
            )
            first.unsafe_ptr()[unsafe_offset=i] = value
            second.unsafe_ptr()[unsafe_offset=i] = Float32(value)


struct CopyTupleOperation(GUFuncOperation, ImplicitlyCopyable):
    """Copy one logical core into a writable output core."""

    comptime Tensors = Tuple[
        TensorArg[DType.float64, False],
        TensorArg[DType.float64, True],
    ]

    def __init__(out self):
        pass

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var source_arg = tensors[0].copy()
        var destination_arg = tensors[1].copy()
        var source = source_arg.read_span()
        var destination = destination_arg.write_span()
        for i in range(len(source)):
            destination.unsafe_ptr()[unsafe_offset=i] = source.unsafe_ptr()[
                unsafe_offset=i
            ]


struct ParallelProbeOperation(Copyable, GUFuncOperation):
    """Record how many worker operation copies the GUFunc creates."""

    comptime Tensors = Tuple[
        TensorArg[DType.float32, False],
        TensorArg[DType.float32, True],
    ]

    var copy_counter_address: Int

    def __init__(out self, copy_counter_address: Int):
        self.copy_counter_address = copy_counter_address

    def __init__(out self, *, copy: Self):
        self.copy_counter_address = copy.copy_counter_address
        var counter = Pointer[
            mut=True,
            Int32,
            MutAnyOrigin,
        ](unsafe_from_address=self.copy_counter_address)
        _ = Atomic[DType.int32].fetch_add(counter, 1)

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var source_arg = tensors[0].copy()
        var destination_arg = tensors[1].copy()
        var source = source_arg.read_span()
        var destination = destination_arg.write_span()
        if len(source) > 0:
            destination.unsafe_ptr()[unsafe_offset=0] = source.unsafe_ptr()[
                unsafe_offset=0
            ]


def test_native_tuple_nansum() raises:
    var source = alloc(Layout[Float64](count=4))
    var destination = alloc(Layout[Float64](count=1))
    var source_ptr = source.unsafe_ptr()
    for i in range(4):
        source_ptr[unsafe_offset=i] = Float64(i + 1)

    var shape = DimArray(fill=1)
    var stride = DimArray(fill=1)
    shape[0] = 4
    var input_layout = TensorArg[DType.float64, False].runtime_layout(
        shape, stride
    )
    var output_layout = TensorArg[DType.float64, True].runtime_layout(
        DimArray(fill=1), DimArray(fill=1)
    )
    var tensors = Tuple(
        TensorArg[DType.float64, False](input_layout, 1, Int(source_ptr)),
        TensorArg[DType.float64, True](
            output_layout, 0, Int(destination.unsafe_ptr())
        ),
    )
    var axes = DimArray(fill=0)
    var driver = GUFunc[NanSum[DType.float64]](NanSum[DType.float64]())
    try:
        driver.execute(tensors, axes, 1, axes, 0, DispatchPolicy(1, 0))
    except e:
        dealloc(source^)
        dealloc(destination^)
        raise e

    var result = destination.unsafe_ptr()[unsafe_offset=0]
    dealloc(source^)
    dealloc(destination^)
    assert_equal(result, 10.0)


def test_heterogeneous_tuple_and_input_scratch() raises:
    """Verify mixed dtypes, multiple outputs, and one copied read core."""

    var left_storage = alloc(Layout[Float32](count=6))
    var right_storage = alloc(Layout[Float64](count=12))
    var first_storage = alloc(Layout[Float64](count=6))
    var second_storage = alloc(Layout[Float32](count=6))

    for i in range(6):
        left_storage.unsafe_ptr()[unsafe_offset=i] = Float32(i + 1)
        first_storage.unsafe_ptr()[unsafe_offset=i] = 0.0
        second_storage.unsafe_ptr()[unsafe_offset=i] = 0.0
    for i in range(12):
        right_storage.unsafe_ptr()[unsafe_offset=i] = -1.0
    for row in range(2):
        for col in range(3):
            right_storage.unsafe_ptr()[
                unsafe_offset=row * 6 + col * 2
            ] = Float64(10 * row + col)

    var left_shape = DimArray(fill=1)
    left_shape[0] = 2
    left_shape[1] = 3
    var left_stride = DimArray(fill=0)
    left_stride[0] = 3
    left_stride[1] = 1

    var right_stride = DimArray(fill=0)
    right_stride[0] = 6
    right_stride[1] = 2

    var output_shape = left_shape.copy()
    var output_stride = left_stride.copy()

    var tensors = Tuple(
        TensorArg[DType.float32, False](
            TensorArg[DType.float32, False].runtime_layout(
                left_shape, left_stride
            ),
            2,
            Int(left_storage.unsafe_ptr()),
        ),
        TensorArg[DType.float64, False](
            TensorArg[DType.float64, False].runtime_layout(
                left_shape, right_stride
            ),
            2,
            Int(right_storage.unsafe_ptr()),
        ),
        TensorArg[DType.float64, True](
            TensorArg[DType.float64, True].runtime_layout(
                output_shape, output_stride
            ),
            2,
            Int(first_storage.unsafe_ptr()),
        ),
        TensorArg[DType.float32, True](
            TensorArg[DType.float32, True].runtime_layout(
                output_shape, output_stride
            ),
            2,
            Int(second_storage.unsafe_ptr()),
        ),
    )
    var axes = DimArray(fill=0)
    axes[0] = 1
    var driver = GUFunc[MixedTupleTransform](MixedTupleTransform())
    try:
        driver.execute(tensors, axes, 1, axes, 1, DispatchPolicy(1, 0))
    except e:
        dealloc(left_storage^)
        dealloc(right_storage^)
        dealloc(first_storage^)
        dealloc(second_storage^)
        raise e

    var ok = True
    for row in range(2):
        for col in range(3):
            var index = row * 3 + col
            var expected = Float64(row * 10 + col) + Float64(index + 1)
            if first_storage.unsafe_ptr()[unsafe_offset=index] != expected:
                ok = False
            if second_storage.unsafe_ptr()[unsafe_offset=index] != Float32(
                expected
            ):
                ok = False

    dealloc(left_storage^)
    dealloc(right_storage^)
    dealloc(first_storage^)
    dealloc(second_storage^)
    assert_equal(ok, True)


def run_noncontiguous_writable_core() raises:
    """A writable core must be unit-stride in the operation's iteration order.
    """

    var source = alloc(Layout[Float64](count=6))
    var destination = alloc(Layout[Float64](count=6))
    for i in range(6):
        source.unsafe_ptr()[unsafe_offset=i] = Float64(i + 1)
        destination.unsafe_ptr()[unsafe_offset=i] = -1.0

    var input_shape = DimArray(fill=1)
    input_shape[0] = 2
    input_shape[1] = 3
    var input_stride = DimArray(fill=0)
    input_stride[0] = 3
    input_stride[1] = 1

    # The output is globally C-contiguous with shape (3, 2), but selecting
    # axis 0 as its core makes the logical core stride two.
    var output_shape = DimArray(fill=1)
    output_shape[0] = 3
    output_shape[1] = 2
    var output_stride = DimArray(fill=0)
    output_stride[0] = 2
    output_stride[1] = 1

    var tensors = Tuple(
        TensorArg[DType.float64, False](
            TensorArg[DType.float64, False].runtime_layout(
                input_shape, input_stride
            ),
            2,
            Int(source.unsafe_ptr()),
        ),
        TensorArg[DType.float64, True](
            TensorArg[DType.float64, True].runtime_layout(
                output_shape, output_stride
            ),
            2,
            Int(destination.unsafe_ptr()),
        ),
    )
    var input_axes = DimArray(fill=0)
    input_axes[0] = 1
    var output_axes = DimArray(fill=0)
    output_axes[0] = 0
    var driver = GUFunc[CopyTupleOperation](CopyTupleOperation())

    try:
        driver.execute(
            tensors,
            input_axes,
            1,
            output_axes,
            1,
            DispatchPolicy(1, 1_000_000),
        )
    except e:
        dealloc(source^)
        dealloc(destination^)
        raise e

    dealloc(source^)
    dealloc(destination^)


def test_rejects_noncontiguous_writable_core() raises:
    with assert_raises():
        run_noncontiguous_writable_core()


def test_three_dimensional_outer_iterations() raises:
    """All two-axis selections preserve outer order and core order."""

    var source = alloc(Layout[Float64](count=24))
    var destination = alloc(Layout[Float64](count=24))
    for i in range(24):
        source.unsafe_ptr()[unsafe_offset=i] = Float64(i + 1)

    var shape = DimArray(fill=1)
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    var stride = DimArray(fill=0)
    stride[0] = 12
    stride[1] = 4
    stride[2] = 1

    # Include both contiguous and non-contiguous core selections, and both
    # orders for every pair of axes.
    var first_axes = [0, 0, 1, 1, 2, 2]
    var second_axes = [1, 2, 0, 2, 0, 1]
    var driver = GUFunc[CopyTupleOperation](CopyTupleOperation())
    var all_ok = True
    try:
        for case_index in range(6):
            var first_axis = first_axes[case_index]
            var second_axis = second_axes[case_index]
            var outer_axis = 0
            for axis in range(3):
                if axis != first_axis and axis != second_axis:
                    outer_axis = axis

            var core_length = shape[first_axis] * shape[second_axis]
            var output_shape = DimArray(fill=1)
            output_shape[0] = shape[outer_axis]
            output_shape[1] = core_length
            var output_stride = DimArray(fill=0)
            output_stride[0] = core_length
            output_stride[1] = 1

            for i in range(24):
                destination.unsafe_ptr()[unsafe_offset=i] = -1.0

            var input_axes = DimArray(fill=0)
            input_axes[0] = first_axis
            input_axes[1] = second_axis
            var output_axes = DimArray(fill=0)
            output_axes[0] = 1
            var tensors = Tuple(
                TensorArg[DType.float64, False](
                    TensorArg[DType.float64, False].runtime_layout(
                        shape, stride
                    ),
                    3,
                    Int(source.unsafe_ptr()),
                ),
                TensorArg[DType.float64, True](
                    TensorArg[DType.float64, True].runtime_layout(
                        output_shape, output_stride
                    ),
                    2,
                    Int(destination.unsafe_ptr()),
                ),
            )

            driver.execute(
                tensors,
                input_axes,
                2,
                output_axes,
                1,
                DispatchPolicy(1, 1_000_000),
            )

            for outer in range(shape[outer_axis]):
                for flat_core in range(core_length):
                    var remaining = flat_core
                    var second_coordinate = remaining % shape[second_axis]
                    remaining //= shape[second_axis]
                    var first_coordinate = remaining % shape[first_axis]
                    var source_offset = (
                        outer * stride[outer_axis]
                        + first_coordinate * stride[first_axis]
                        + second_coordinate * stride[second_axis]
                    )
                    var output_offset = outer * core_length + flat_core
                    if (
                        destination.unsafe_ptr()[unsafe_offset=output_offset]
                        != source.unsafe_ptr()[unsafe_offset=source_offset]
                    ):
                        all_ok = False
    except e:
        dealloc(source^)
        dealloc(destination^)
        raise e

    dealloc(source^)
    dealloc(destination^)
    assert_equal(all_ok, True)


def run_parallel_threshold_case(core_length: Int, expected_copies: Int) raises:
    var outer_count = 4
    var element_count = outer_count * core_length
    var source = alloc(Layout[Float32](count=element_count))
    var destination = alloc(Layout[Float32](count=element_count))
    var copy_counter = alloc(Layout[Int32](count=1))
    copy_counter.unsafe_ptr()[unsafe_offset=0] = 0

    for outer in range(outer_count):
        source.unsafe_ptr()[unsafe_offset=outer * core_length] = Float32(outer)

    var shape = DimArray(fill=1)
    shape[0] = outer_count
    shape[1] = core_length
    var stride = DimArray(fill=0)
    stride[0] = core_length
    stride[1] = 1
    var tensors = Tuple(
        TensorArg[DType.float32, False](
            TensorArg[DType.float32, False].runtime_layout(shape, stride),
            2,
            Int(source.unsafe_ptr()),
        ),
        TensorArg[DType.float32, True](
            TensorArg[DType.float32, True].runtime_layout(shape, stride),
            2,
            Int(destination.unsafe_ptr()),
        ),
    )
    var input_axes = DimArray(fill=0)
    input_axes[0] = 1
    var output_axes = DimArray(fill=0)
    output_axes[0] = 1
    var driver = GUFunc[ParallelProbeOperation](
        ParallelProbeOperation(Int(copy_counter.unsafe_ptr()))
    )
    # GUFunc construction also copies the operation.  Count only copies made
    # by execute_serial_or_parallel below.
    copy_counter.unsafe_ptr()[unsafe_offset=0] = 0

    try:
        driver.execute(
            tensors,
            input_axes,
            1,
            output_axes,
            1,
            DispatchPolicy(4, 2_000_000),
        )
    except e:
        dealloc(source^)
        dealloc(destination^)
        dealloc(copy_counter^)
        raise e

    var observed_copies = Int(copy_counter.unsafe_ptr()[unsafe_offset=0])
    dealloc(source^)
    dealloc(destination^)
    dealloc(copy_counter^)
    assert_equal(observed_copies, expected_copies)


def test_parallel_threshold_uses_two_million_element_cutoff() raises:
    """Execution stays serial below the cutoff and parallelizes at it."""

    var policy = DispatchPolicy(4, 2_000_000)
    assert_equal(policy.effective_workers(4, 1), 1)
    assert_equal(policy.effective_workers(4, 499_999), 1)
    assert_equal(policy.effective_workers(4, 500_000), 4)
    assert_equal(policy.effective_workers(4, 500_001), 4)
    assert_equal(policy.effective_workers(1, 2_000_000), 1)
    run_parallel_threshold_case(499_999, 1)
    run_parallel_threshold_case(500_000, 4)
    run_parallel_threshold_case(500_001, 4)


def main() raises:
    test_native_tuple_nansum()
    test_heterogeneous_tuple_and_input_scratch()
    test_rejects_noncontiguous_writable_core()
    test_three_dimensional_outer_iterations()
    test_parallel_threshold_uses_two_million_element_cutoff()
