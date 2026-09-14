"""Compile-time signatures and runtime axis contracts for ``guvectorize``.

The driver follows the gufunc split between a logical core and the remaining
outer dimensions. ``CoreSpec`` describes the core dimensions at compile time;
``AxisSpec`` selects the corresponding physical axes at runtime. The executor
removes those axes from each operand, broadcasts the remaining input shapes,
and calls ``GUFuncKernel.__call__`` once for every outer position.

The operation receives a native tuple rather than named gufunc arguments. The
tuple's dtype, read/write flags, and core specs are all part of its compile-time
type, so the binding and the kernel must use exactly the same tuple order.
"""

from std.collections import InlineArray

from mojagg.drivers.guvectorize_layout import DimArray, MAX_RANK


comptime BoolArray = InlineArray[Bool, MAX_RANK]


trait CoreDim(Copyable & Deinitable):
    """One compile-time gufunc core dimension.

    ``value`` is either a symbolic ID or a fixed extent. ``is_fixed`` tells
    signature resolution whether the runtime core shape must bind the value or
    compare against it.
    """

    comptime value: Int
    comptime is_fixed: Bool


struct FixedDim[size: Int](CoreDim, ImplicitlyCopyable):
    """A core dimension whose extent is fixed by the signature."""

    comptime value = Self.size
    comptime is_fixed = True


struct Dim[symbol_id: Int](CoreDim, ImplicitlyCopyable):
    """A named core dimension shared by equal symbol IDs."""

    comptime value = Self.symbol_id
    comptime is_fixed = False


trait CoreSpecProtocol:
    """Associated metadata exposed by a core-shape specification."""

    comptime rank: Int

    @staticmethod
    def values() -> DimArray:
        ...

    @staticmethod
    def fixed() -> BoolArray:
        ...


struct CoreSpec[*Dims: CoreDim](CoreSpecProtocol):
    """An ordered, statically constrained gufunc core shape.

    ``CoreSpec[]`` is a scalar core. Equal ``Dim`` IDs express equal runtime
    extents, while ``FixedDim`` expresses an exact extent. The runtime shape is
    resolved later by ``build_signature``.
    """

    comptime rank = len(Self.Dims)
    comptime dimensions = Tuple[*Self.Dims]

    @staticmethod
    def values() -> DimArray:
        var result = DimArray(fill=0)
        comptime for i in range(len(Self.Dims)):
            result[i] = Self.Dims[i].value
        return result^

    @staticmethod
    def fixed() -> BoolArray:
        var result = BoolArray(fill=False)
        comptime for i in range(len(Self.Dims)):
            result[i] = Self.Dims[i].is_fixed
        return result^


@fieldwise_init
struct AxisSpec(Copyable):
    """Runtime physical positions of an operand's logical core axes.

    ``values[0:count]`` is ordered logical core order. It is not necessarily
    sorted physical-axis order: that order determines how a selected N-D core
    is flattened before the kernel sees its span. ``values`` has fixed capacity
    ``MAX_RANK`` and ``count`` is the only active length.
    """

    var values: DimArray
    var count: Int

    @staticmethod
    def empty() -> Self:
        return Self(DimArray(fill=0), 0)

    @always_inline
    def __getitem__(self, index: Int) -> Int:
        return self.values[index]

    @always_inline
    def __setitem__(mut self, index: Int, value: Int):
        self.values[index] = value


trait GUFuncKernel(Copyable & Deinitable):
    """Operation contract for one prepared core tuple.

    ``__call__`` is the inner gufunc body. It is called once per outer slice
    with descriptors whose active lengths are the selected core lengths. Rank,
    axis normalization, outer broadcasting, source strides, scratch copies,
    and worker scheduling have already been handled by the driver.

    The trait cannot declare a parameterized variadic argument in the current
    Mojo version, so each operation supplies a concrete native tuple through
    its ``Signature`` associated constant.
    """

    comptime Signature: AnyType

    # A parameterized variadic trait would express direct named arguments, but
    # Mojo 1.0 does not permit parameters on trait declarations.  The concrete
    # signature remains fully static and the operation receives one native tuple.
    def __call__(mut self, tensors: Self.Signature):
        ...
