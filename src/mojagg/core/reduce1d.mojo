"""Reduction contracts and shared, layout-specialized 1-D scanners.

State is a mergeable partial, not necessarily the public result. Acc is
contiguous working storage; strided scans only carry State. finalize runs
once per output slice, after the axis driver has merged every partial.

SHORT_CIRCUIT specializes both scanners and the axis driver.
Enabling it requires overriding its corresponding hook.
"""

from std.algorithm import vectorize
from std.collections import Span
from std.sys.info import simd_width_of


struct ReductionWidth[dtype: DType, chains: Int = 0]:
    """Generic dtype-aware SIMD width and accumulator chain determination.

    Computes native lanes and scales unrolled chains depending on the dtype's
    bitwidth and hardware vector capacity so all operations can reuse a
    unified policy.
    """

    comptime native_lanes = simd_width_of[Self.dtype]()

    # Default optimal chains:
    # 64-bit dtypes (float64, int64): 4 chains (e.g. 4x4 = 16 lanes on AVX2)
    # 32-bit dtypes (float32, int32): 2 chains (e.g. 2x8 = 16 lanes on AVX2)
    # smaller dtypes (int16, int8, bool): 1 chain (16-32 native lanes)
    comptime is_64bit = (
        Self.dtype == DType.float64
        or Self.dtype == DType.int64
        or Self.dtype == DType.uint64
    )
    comptime is_32bit = (
        Self.dtype == DType.float32
        or Self.dtype == DType.int32
        or Self.dtype == DType.uint32
    )
    comptime default_chains = 4 if Self.is_64bit else (
        2 if Self.is_32bit else 1
    )

    comptime effective_chains = (
        Self.chains if Self.chains > 0 else Self.default_chains
    )
    comptime block_lanes = Self.native_lanes * Self.effective_chains


trait Reduction1D(Copyable & Deinitable):
    comptime value_dtype: DType
    comptime out_dtype: DType
    comptime State: Copyable & Deinitable
    comptime SHORT_CIRCUIT: Bool = False

    @staticmethod
    def identity() -> Self.State:
        ...

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        ...

    @staticmethod
    def combine_at(
        acc: Self.State, partial: Self.State, offset: Int
    ) -> Self.State:
        return Self.combine(acc, partial)

    def finalize(self, state: Self.State) -> Scalar[Self.out_dtype]:
        ...

    @staticmethod
    def contig(data: Span[Scalar[Self.value_dtype], _]) -> Self.State:
        ...

    @staticmethod
    def strided(
        ptr: Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin],
        n: Int,
        stride: Int,
    ) -> Self.State:
        ...

    @staticmethod
    def is_terminal(state: Self.State) -> Bool:
        comptime assert not Self.SHORT_CIRCUIT, "override is_terminal"
        return False


trait NaNReduction1D(Reduction1D):
    comptime Acc: Copyable & Deinitable
    comptime block_lanes: Int

    @staticmethod
    def acc_init() -> Self.Acc:
        ...

    @staticmethod
    def step_scalar(
        state: Self.State, value: Scalar[Self.value_dtype]
    ) -> Self.State:
        ...

    @staticmethod
    def step_simd(
        mut acc: Self.Acc,
        values: SIMD[Self.value_dtype, Self.block_lanes],
    ):
        ...

    @staticmethod
    def step_tail[
        lane: Int
    ](mut acc: Self.Acc, value: Scalar[Self.value_dtype]):
        ...

    @staticmethod
    def collapse(acc: Self.Acc) -> Self.State:
        ...

    @staticmethod
    def contig(data: Span[Scalar[Self.value_dtype], _]) -> Self.State:
        return scan_contig[Self](data)

    @staticmethod
    def strided(
        ptr: Pointer[mut=True, Scalar[Self.value_dtype], MutAnyOrigin],
        n: Int,
        stride: Int,
    ) -> Self.State:
        return scan_strided[Self](ptr, n, stride)


@always_inline
def scan_contig[
    Op: NaNReduction1D
](data: Span[Scalar[Op.value_dtype], _],) -> Op.State:
    comptime L = Op.block_lanes
    comptime assert L > 0 and (L & (L - 1)) == 0, "power-of-two block required"
    var n = len(data)
    var ptr = data.unsafe_ptr()
    var acc = Op.acc_init()

    comptime if Op.SHORT_CIRCUIT:
        # vectorize callbacks cannot terminate the enclosing traversal.
        var end = n - n % L
        var i = 0
        while i < end:
            Op.step_simd(acc, ptr.unsafe_load[width=L](i))
            if Op.is_terminal(Op.collapse(acc)):
                return Op.collapse(acc)
            i += L
        var state = Op.collapse(acc)
        while i < n:
            state = Op.step_scalar(state, ptr[unsafe_offset=i])
            if Op.is_terminal(state):
                return state^
            i += 1
        return state^
    else:

        def step[width: Int](i: Int, evl: Int) {imm ptr, mut acc}:
            # Fixed L avoids the V1 closure's unbound-width type issue.
            if evl == L:
                Op.step_simd(acc, ptr.unsafe_load[width=L](i))
            else:
                comptime for k in range(L):
                    if k < evl:
                        Op.step_tail[k](acc, ptr[unsafe_offset=i + k])

        vectorize[L](n, step)
        return Op.collapse(acc)


@always_inline
def scan_strided[
    Op: NaNReduction1D
](
    ptr: Pointer[mut=True, Scalar[Op.value_dtype], MutAnyOrigin],
    n: Int,
    stride: Int,
) -> Op.State:
    var state = Op.identity()
    var off = 0
    for _ in range(n):
        state = Op.step_scalar(state, ptr[unsafe_offset=off])
        comptime if Op.SHORT_CIRCUIT:
            if Op.is_terminal(state):
                return state^
        off += stride
    return state^
