"""Typed runtime tensor arguments for the generic GUFunc boundary.

``TensorArg`` is the only tensor descriptor exposed to an operation.  Its
second compile-time parameter, ``writable``, describes the role of the
descriptor in the operation tuple:

* ``TensorArg[T, False]`` is a read-only view.  The GUFunc may rebind it to a
  worker scratch buffer when the selected core is strided.
* ``TensorArg[T, True]`` is a writable view.  Its selected core is validated
  as contiguous and is always rebound directly to caller-owned output
  storage.  No output scratch or copy-back path exists.

Operations use Mojo's native heterogeneous ``Tuple`` directly.  There are no
read-tuple, write-tuple, or tuple-view traits: each tuple element carries its
capabilities in its ``TensorArg`` type.  Compile-time tuple indexing therefore
specializes to the concrete dtype and mutability without boxing or runtime
dispatch.

Shapes and strides are runtime metadata held by ``RuntimeLayout``.  The fixed
``MAX_RANK`` capacity is only storage for metadata; tensor extents and slice
lengths remain dynamic.  Addresses are borrowed from the Python owner, so the
views do not own or copy tensor data.
"""

from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from std.collections import InlineArray, Span
from std.memory import unsafe_memcpy
from std.sys import size_of

comptime MAX_RANK = 8
comptime DimArray = InlineArray[Int, MAX_RANK]


def _dtype_name[dtype: DType]() -> String:
    """Return the NumPy spelling of a dtype supported by mojagg."""

    comptime if dtype == DType.float64:
        return "float64"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unsupported"


comptime UnknownDims = IntTuple(
    UNKNOWN_VALUE,
    UNKNOWN_VALUE,
    UNKNOWN_VALUE,
    UNKNOWN_VALUE,
    UNKNOWN_VALUE,
    UNKNOWN_VALUE,
    UNKNOWN_VALUE,
    UNKNOWN_VALUE,
)
comptime DynamicLayout = Layout(UnknownDims, UnknownDims)
comptime DynamicRuntimeLayout = RuntimeLayout[
    DynamicLayout,
    element_type=DType.int64,
    linear_idx_type=DType.int64,
]

# Storage aliases are intentionally separate from TensorArg.  They are typed
# LayoutTensors used by the Python boundary while constructing a borrowed
# argument.  TensorArg itself stores only runtime metadata and raw addresses,
# which keeps its writable/read-only specializations identical in layout.
comptime ReadTensor[dtype: DType] = LayoutTensor[
    mut=False,
    dtype,
    DynamicLayout,
    ImmUntrackedOrigin,
]
comptime WriteTensor[dtype: DType] = LayoutTensor[
    mut=True,
    dtype,
    DynamicLayout,
    MutUntrackedOrigin,
]


@fieldwise_init
struct OperandPlan(Copyable):
    """Runtime address and axis metadata for one tensor operand."""

    var rank: Int
    var core_rank: Int
    var outer_rank: Int
    var core_length: Int
    var outer_count: Int
    var base_address: Int
    var shape: DimArray
    var stride: DimArray
    var core_shape: DimArray
    var core_stride: DimArray
    var outer_shape: DimArray
    var outer_stride: DimArray
    var core_contiguous: Bool

    @staticmethod
    def empty() -> Self:
        """Construct the zero metadata value used to initialize plan arrays."""

        return Self(
            0,
            0,
            0,
            0,
            0,
            0,
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            True,
        )


trait TensorArgProtocol(Copyable & Deinitable):
    """Common compile-time contract for native Tuple elements.

    This is an operand contract, not a tuple wrapper.  It lets generic GUFunc
    helpers specialize native ``Tuple[*Args]`` packs while each concrete
    element remains ``TensorArg[dtype, writable]``.
    """

    comptime dtype: DType

    def rank(self) -> Int:
        ...

    def shape_at(self, axis: Int) -> Int:
        ...

    def stride_at(self, axis: Int) -> Int:
        ...

    def base_address(self) -> Int:
        ...

    def is_writable(self) -> Bool:
        ...

    def set_slice(mut self, address: Int, length: Int):
        ...

    def copy_core(self, plan: OperandPlan, outer_offset: Int, destination: Int):
        ...

    def item_size(self) -> Int:
        ...


struct TensorArg[
    element_dtype: DType,
    writable: Bool,
](ImplicitlyCopyable, TensorArgProtocol):
    """One typed tensor argument, readable in every mode and writable by flag.

    ``TensorArg[T, False]`` exposes an immutable ``read_span``.  The
    ``TensorArg[T, True]`` specialization additionally permits ``write_span``
    and tells the planner that the selected core must be directly contiguous.
    The original allocation is borrowed; copies of this struct copy only
    metadata and addresses.
    """

    comptime dtype = Self.element_dtype

    var layout: DynamicRuntimeLayout
    var base: Int
    var ndim: Int
    var active_length: Int
    var active: Int

    def __init__(
        out self,
        layout: DynamicRuntimeLayout,
        ndim: Int,
        base: Int,
    ):
        self.layout = layout
        self.base = base
        self.ndim = ndim
        self.active_length = 0
        self.active = base

    def __init__(out self, *, copy: Self):
        self.layout = copy.layout
        self.base = copy.base
        self.ndim = copy.ndim
        self.active_length = copy.active_length
        self.active = copy.active

    @staticmethod
    def runtime_layout(
        shape: DimArray, strides: DimArray
    ) -> DynamicRuntimeLayout:
        """Construct runtime layout metadata from signed element strides."""

        var runtime_shape = DynamicRuntimeLayout.ShapeType()
        var runtime_stride = DynamicRuntimeLayout.StrideType()
        for i in range(MAX_RANK):
            runtime_shape.value[i] = shape[i]
            runtime_stride.value[i] = strides[i]
        return DynamicRuntimeLayout(runtime_shape, runtime_stride)

    @staticmethod
    def from_tensor(
        layout: DynamicRuntimeLayout,
        ndim: Int,
        base: Int,
    ) -> Self:
        """Construct a borrowed argument from runtime layout metadata."""

        return Self(layout, ndim, base)

    @staticmethod
    def from_read_tensor(
        tensor: ReadTensor[Self.element_dtype],
        ndim: Int,
        base: Int,
    ) -> Self:
        """Construct a read argument from a borrowed read LayoutTensor."""

        comptime assert not Self.writable, "read tensor requires writable=False"
        return Self(
            rebind[DynamicRuntimeLayout](tensor.runtime_layout),
            ndim,
            base,
        )

    @staticmethod
    def from_write_tensor(
        tensor: WriteTensor[Self.element_dtype],
        ndim: Int,
        base: Int,
    ) -> Self:
        """Construct a writable argument from a borrowed write LayoutTensor."""

        comptime assert Self.writable, "write tensor requires writable=True"
        return Self(
            rebind[DynamicRuntimeLayout](tensor.runtime_layout),
            ndim,
            base,
        )

    @always_inline
    def rank(self) -> Int:
        return self.ndim

    @always_inline
    def is_writable(self) -> Bool:
        return Self.writable

    @always_inline
    def item_size(self) -> Int:
        return size_of[Scalar[Self.element_dtype]]()

    @always_inline
    def shape_at(self, axis: Int) -> Int:
        return self.layout.shape.value[axis]

    @always_inline
    def stride_at(self, axis: Int) -> Int:
        return self.layout.stride.value[axis]

    @always_inline
    def base_address(self) -> Int:
        return self.base

    @always_inline
    def active_address(self) -> Int:
        return self.active

    @always_inline
    def set_slice(mut self, address: Int, length: Int):
        """Bind this descriptor to one prepared outer slice."""

        self.active_length = length
        self.active = address

    @always_inline
    def copy_contiguous_run(
        self,
        source: Pointer[
            mut=False,
            Scalar[Self.element_dtype],
            ImmUntrackedOrigin,
        ],
        target: Pointer[
            mut=True,
            Scalar[Self.element_dtype],
            MutUntrackedOrigin,
        ],
        source_offset: Int,
        target_offset: Int,
        length: Int,
    ):
        """Copy a non-overlapping unit-stride run with ``memcpy``."""

        unsafe_memcpy(
            dest=target.unsafe_offset(target_offset),
            src=source.unsafe_offset(source_offset),
            count=length,
        )

    @always_inline
    def copy_core(self, plan: OperandPlan, outer_offset: Int, destination: Int):
        """Materialize one strided read core into caller-provided scratch."""

        comptime if Self.writable:
            return
        else:
            var source = Pointer[
                mut=False,
                Scalar[Self.element_dtype],
                ImmUntrackedOrigin,
            ](unsafe_from_address=self.base)
            var target = Pointer[
                mut=True,
                Scalar[Self.element_dtype],
                MutUntrackedOrigin,
            ](unsafe_from_address=destination)

            if plan.core_rank == 0:
                target[unsafe_offset=0] = source[unsafe_offset=outer_offset]
                return

            if plan.core_stride[plan.core_rank - 1] == 1:
                var run_length = plan.core_shape[plan.core_rank - 1]
                if run_length > 0:
                    var run_count = plan.core_length // run_length
                    var coordinates = DimArray(fill=0)
                    var offset = outer_offset
                    var copied = 0
                    for _ in range(run_count):
                        self.copy_contiguous_run(
                            source,
                            target,
                            offset,
                            copied,
                            run_length,
                        )
                        copied += run_length
                        var d = plan.core_rank - 2
                        while d >= 0:
                            coordinates[d] += 1
                            offset += plan.core_stride[d]
                            if coordinates[d] < plan.core_shape[d]:
                                break
                            offset -= plan.core_shape[d] * plan.core_stride[d]
                            coordinates[d] = 0
                            d -= 1
                    return

            var coordinates = DimArray(fill=0)
            var offset = outer_offset
            for flat in range(plan.core_length):
                target[unsafe_offset=flat] = source[unsafe_offset=offset]
                var d = plan.core_rank - 1
                while d >= 0:
                    coordinates[d] += 1
                    offset += plan.core_stride[d]
                    if coordinates[d] < plan.core_shape[d]:
                        break
                    offset -= plan.core_shape[d] * plan.core_stride[d]
                    coordinates[d] = 0
                    d -= 1

    @always_inline
    def read_span(
        self,
    ) -> Span[Scalar[Self.element_dtype], ImmUntrackedOrigin]:
        """Return the active slice as an immutable span in every mode."""

        var pointer = Pointer[
            mut=False,
            Scalar[Self.element_dtype],
            ImmUntrackedOrigin,
        ](unsafe_from_address=self.active)
        return Span[
            Scalar[Self.element_dtype],
            ImmUntrackedOrigin,
        ](unsafe_ptr=pointer, length=self.active_length)

    @always_inline
    def write_span(
        mut self,
    ) -> Span[Scalar[Self.element_dtype], MutUntrackedOrigin]:
        """Return a mutable active span; valid only for writable arguments."""

        comptime assert Self.writable, "write_span requires writable=True"
        var pointer = Pointer[
            mut=True,
            Scalar[Self.element_dtype],
            MutUntrackedOrigin,
        ](unsafe_from_address=self.active)
        return Span[
            Scalar[Self.element_dtype],
            MutUntrackedOrigin,
        ](unsafe_ptr=pointer, length=self.active_length)


# This is only a spelling aid for generic helpers.  It is Mojo's native
# heterogeneous tuple itself: there is no wrapper object, heap allocation, or
# runtime tuple protocol.  Operation declarations normally use ``Tuple[...]``
# directly so their complete fixed-arity contract stays visible at the call
# site.
comptime TensorTuple[*Args: TensorArgProtocol] = Tuple[*Args]
