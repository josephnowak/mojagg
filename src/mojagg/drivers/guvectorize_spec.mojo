"""Compile-time signatures and runtime axis contracts for guvectorize."""

from std.collections import InlineArray

from mojagg.drivers.guvectorize_layout import DimArray, MAX_RANK


comptime BoolArray = InlineArray[Bool, MAX_RANK]


trait CoreDim(Copyable & Deinitable):
    """One compile-time gufunc core dimension."""

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
    """A statically constrained heterogeneous tuple of core dimensions."""

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
    """Runtime core-axis positions with one explicit logical count."""

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
    """Operation contract for a combined input/output tensor signature."""

    comptime Signature: AnyType

    # A parameterized variadic trait would express direct named arguments, but
    # Mojo 1.0 does not permit parameters on trait declarations.  The concrete
    # signature remains fully static and the operation receives one native tuple.
    def __call__(mut self, tensors: Self.Signature):
        ...
