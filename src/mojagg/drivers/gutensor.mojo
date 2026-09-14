"""Non-owning tensor descriptors used by the ``guvectorize`` driver.

``GUTensor`` stores an address and layout metadata; it never owns the NumPy
allocation behind that address. A complete tensor descriptor is created at the
Python boundary. During execution the driver copies the descriptor per worker
and temporarily rebinds its address and ``length`` to one outer slice. The
kernel consequently sees a typed contiguous span for its logical core.

The descriptor's shape and stride arrays describe the original full tensor.
Strides are measured in elements. ``copy_core`` is the only place where a
non-contiguous selected read core is materialized into worker-local storage.
Writable cores are validated by the planner and never use this copy path.
"""

from std.collections import Span

from mojagg.drivers.guvectorize_layout import DimArray, OperandPlan
from mojagg.drivers.guvectorize_spec import CoreSpecProtocol


@always_inline
def contiguous_strides(shape: DimArray, ndim: Int) -> DimArray:
    var stride = DimArray(fill=0)
    var step = 1
    for axis in range(ndim - 1, -1, -1):
        stride[axis] = step
        step *= shape[axis]
    return stride^


@always_inline
def tensor_element_count(shape: DimArray, ndim: Int) -> Int:
    var total = 1
    for axis in range(ndim):
        total *= shape[axis]
    return total


trait AnyGUTensor(Copyable & Deinitable):
    """Common descriptor contract used by the generic planner."""

    comptime dtype: DType
    comptime is_output: Bool
    comptime core_spec: CoreSpecProtocol

    def bind_address(mut self, address: Int, length: Int):
        ...

    def is_bound(self) -> Bool:
        ...

    def set_unbound_layout(mut self, shape: DimArray, ndim: Int):
        ...

    def copy_core(self, plan: OperandPlan, outer_offset: Int, destination: Int):
        ...


struct GUTensor[
    element_dtype: DType,
    output: Bool,
    core: CoreSpecProtocol,
](AnyGUTensor, ImplicitlyCopyable):
    """A non-owning descriptor for one gufunc operand.

    ``element_dtype``, ``output``, and ``core`` are compile-time properties.
    ``address`` is a raw borrowed address; the descriptor does not keep a
    Python owner alive. The binding must keep every participating Python array
    alive until the synchronous native call returns.
    """

    comptime dtype = Self.element_dtype
    comptime is_output = Self.output
    comptime core_spec = Self.core

    # Full-tensor metadata. ``length`` becomes the active core length after
    # execute_range rebinds this descriptor for one outer position.
    var address: Int
    var bound: Bool
    var shape: DimArray
    var stride: DimArray
    var ndim: Int
    var length: Int

    def __init__(
        out self,
        address: Int,
        shape: DimArray,
        stride: DimArray,
        ndim: Int,
    ):
        self.address = address
        self.bound = address != 0
        self.shape = shape.copy()
        self.stride = stride.copy()
        self.ndim = ndim
        self.length = tensor_element_count(shape, ndim)

    @staticmethod
    def unbound(
        shape: DimArray,
        stride: DimArray,
        ndim: Int,
    ) -> Self:
        """Create a metadata-only descriptor for a planned output."""

        return Self(0, shape, stride, ndim)

    @staticmethod
    def empty() -> Self:
        """Create an unbound descriptor used as an output template."""

        return Self.unbound(DimArray(fill=0), DimArray(fill=0), 0)

    @staticmethod
    def borrow(
        address: Int,
        shape: DimArray,
        stride: DimArray,
        ndim: Int,
    ) -> Self:
        """Create a bound descriptor over caller-owned storage."""

        var tensor = Self.unbound(shape, stride, ndim)
        tensor.bind_address(address, tensor.length)
        return tensor^

    @staticmethod
    def unbound_contiguous(shape: DimArray, ndim: Int) -> Self:
        """Create an unbound output using row-major element strides."""

        var stride = contiguous_strides(shape, ndim)
        return Self.unbound(shape, stride^, ndim)

    def __init__(out self, *, copy: Self):
        self.address = copy.address
        self.bound = copy.bound
        self.shape = copy.shape.copy()
        self.stride = copy.stride.copy()
        self.ndim = copy.ndim
        self.length = copy.length

    @always_inline
    def bind_address(mut self, address: Int, length: Int):
        """Bind this borrowed view to a complete tensor or one core slice."""

        self.address = address
        self.length = length
        self.bound = True

    @always_inline
    def is_bound(self) -> Bool:
        return self.bound

    @always_inline
    def set_unbound_layout(mut self, shape: DimArray, ndim: Int):
        """Set planned output metadata while retaining an unbound address."""

        self.address = 0
        self.bound = False
        self.shape = shape.copy()
        self.ndim = ndim
        self.stride = contiguous_strides(shape, ndim)
        self.length = tensor_element_count(shape, ndim)

    @always_inline
    def read_ptr(
        self,
    ) -> Pointer[mut=False, Scalar[Self.element_dtype], ImmUntrackedOrigin,]:
        return Pointer[
            mut=False,
            Scalar[Self.element_dtype],
            ImmUntrackedOrigin,
        ](unsafe_from_address=self.address)

    @always_inline
    def write_ptr(
        mut self,
    ) -> Pointer[mut=True, Scalar[Self.element_dtype], MutUntrackedOrigin,]:
        comptime assert Self.output, "write_ptr requires an output tensor"
        return Pointer[
            mut=True,
            Scalar[Self.element_dtype],
            MutUntrackedOrigin,
        ](unsafe_from_address=self.address)

    @always_inline
    def read_span(
        self,
    ) -> Span[Scalar[Self.element_dtype], ImmUntrackedOrigin]:
        return Span[
            Scalar[Self.element_dtype],
            ImmUntrackedOrigin,
        ](unsafe_ptr=self.read_ptr(), length=self.length)

    @always_inline
    def write_span(
        mut self,
    ) -> Span[Scalar[Self.element_dtype], MutUntrackedOrigin]:
        comptime assert Self.output, "write_span requires an output tensor"
        return Span[
            Scalar[Self.element_dtype],
            MutUntrackedOrigin,
        ](unsafe_ptr=self.write_ptr(), length=self.length)

    @always_inline
    def copy_core(
        self,
        plan: OperandPlan,
        outer_offset: Int,
        destination: Int,
    ):
        """Copy one logical read core in AxisSpec order into ``destination``.

        ``outer_offset`` already points at the selected outer coordinate. The
        odometer below advances the last logical core axis fastest, matching
        the order expected by a contiguous kernel span. The method is a no-op
        for outputs because writable cores must already be contiguous.
        """
        comptime if Self.output:
            return
        else:
            if plan.core_length <= 0:
                return

            var source = Pointer[
                mut=False,
                Scalar[Self.element_dtype],
                ImmUntrackedOrigin,
            ](unsafe_from_address=plan.base_address)
            var target = Pointer[
                mut=True,
                Scalar[Self.element_dtype],
                MutUntrackedOrigin,
            ](unsafe_from_address=destination)

            if plan.core_rank == 0:
                target[unsafe_offset=0] = source[unsafe_offset=outer_offset]
                return

            var coordinates = DimArray(fill=0)
            var source_offset = outer_offset
            for flat in range(plan.core_length):
                target[unsafe_offset=flat] = source[unsafe_offset=source_offset]

                var axis = plan.core_rank - 1
                while axis >= 0:
                    coordinates[axis] += 1
                    source_offset += plan.core_stride[axis]
                    if coordinates[axis] < plan.core_shape[axis]:
                        break
                    source_offset -= (
                        plan.core_shape[axis] * plan.core_stride[axis]
                    )
                    coordinates[axis] = 0
                    axis -= 1
