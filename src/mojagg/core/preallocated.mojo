"""Reusable worker-local allocations for stateful kernels."""

from std.memory import Allocation, alloc, dealloc
from std.memory.alloc import Layout
from std.sys.info import simd_width_of


struct Preallocated[dtype: DType](ImplicitlyCopyable):
    """Own and reset a resizable typed buffer for one operation copy."""

    var storage: Allocation[Scalar[Self.dtype]]
    var capacity: Int

    def __init__(out self):
        self.storage = alloc(Layout[Scalar[Self.dtype]](count=0))
        self.capacity = 0

    def __init__(out self, *, copy: Self):
        self.storage = alloc(Layout[Scalar[Self.dtype]](count=0))
        self.capacity = 0

    def __deinit__(deinit self):
        dealloc(self.storage^)

    @always_inline
    def get_ptr(
        mut self,
        size: Int,
        identity: Scalar[Self.dtype],
    ) -> Pointer[mut=True, Scalar[Self.dtype], MutUntrackedOrigin]:
        if size != self.capacity:
            dealloc(self.storage^)
            self.storage = alloc(Layout[Scalar[Self.dtype]](count=size))
            self.capacity = size

        var pointer = Pointer[
            mut=True,
            Scalar[Self.dtype],
            MutUntrackedOrigin,
        ](unsafe_from_address=Int(self.storage.unsafe_ptr()))
        comptime width = simd_width_of[Self.dtype]() * 4
        var identity_block = SIMD[Self.dtype, width](identity)
        var i = 0
        while i + width <= size:
            pointer.unsafe_store[width=width](i, identity_block)
            i += width
        while i < size:
            pointer[unsafe_offset=i] = identity
            i += 1
        return pointer
