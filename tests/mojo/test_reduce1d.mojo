"""Instrument traversal counts, not just results, across driver scenarios.

Run: mojo run -I src tests/mojo/test_reduce1d.mojo
"""

from std.math import isnan
from std.python import Python, PythonObject
from std.testing import assert_equal, assert_true

from mojagg.core.ndview import DimArray, NDView
from mojagg.core.numeric import nan_or_zero
from mojagg.core.reduce1d import NaNReduction1D
from mojagg.drivers.gufunc import apply_gufunc
from mojagg.nanfuncs.nanmean import NanMean


@fieldwise_init
struct Observations(Copyable):
    var all_nan: Bool
    var reads: Int


struct RecordingAllNan(NaNReduction1D):
    comptime value_dtype = DType.float64
    comptime out_dtype = DType.int64
    comptime State = Observations
    comptime Acc = Observations
    comptime block_lanes = 4
    comptime SHORT_CIRCUIT = True

    @staticmethod
    def identity() -> Self.State:
        return Observations(True, 0)

    @staticmethod
    def acc_init() -> Self.Acc:
        return Self.identity()

    @staticmethod
    def step_scalar(state: Self.State, value: Float64) -> Self.State:
        return Observations(state.all_nan and isnan(value), state.reads + 1)

    @staticmethod
    def step_simd(mut acc: Self.Acc, values: SIMD[DType.float64, 4]):
        acc.all_nan = acc.all_nan and Bool(isnan(values).reduce_and())
        acc.reads += 4

    @staticmethod
    def step_tail[lane: Int](mut acc: Self.Acc, value: Float64):
        acc = Self.step_scalar(acc, value)

    @staticmethod
    def collapse(acc: Self.Acc) -> Self.State:
        return acc.copy()

    @staticmethod
    def is_terminal(state: Self.State) -> Bool:
        return not state.all_nan

    @staticmethod
    def combine(acc: Self.State, partial: Self.State) -> Self.State:
        return Observations(
            acc.all_nan and partial.all_nan, acc.reads + partial.reads
        )

    def __init__(out self):
        pass

    def finalize(self, state: Self.State) -> Int64:
        return Int64(state.reads)


def run[
    Op: NaNReduction1D
](
    op: Op, a: PythonObject, axes: DimArray, k: Int, threshold: Int
) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var result = np.full(3, -1, dtype=np.int64)
    var view = NDView[Op.value_dtype].from_numpy(a, "test")
    apply_gufunc[Op](op, view, axes, k, Int(py=result.ctypes.data), threshold)
    return result


def test_mean_state[Op: NaNReduction1D](op: Op) raises:
    assert_true(isnan(op.finalize(Op.identity())))
    var acc = Op.acc_init()
    Op.step_tail[0](acc, Scalar[Op.value_dtype](3))
    Op.step_tail[0](acc, nan_or_zero[Op.value_dtype]())
    var values = SIMD[Op.value_dtype, Op.block_lanes](2)
    values[0] = nan_or_zero[Op.value_dtype]()
    Op.step_simd(acc, values)
    var state = Op.collapse(acc)
    var extra = Op.step_scalar(Op.identity(), Scalar[Op.value_dtype](10))
    var merged = Op.combine(state, extra)
    assert_equal(
        op.finalize(merged),
        Scalar[Op.out_dtype](
            Float64(2 * Op.block_lanes + 11) / Float64(Op.block_lanes + 1)
        ),
    )

    # Synthesize >2**24 observations without a large allocation: float32
    # counts would lose the final increment and change the rounded mean.
    state = Op.step_scalar(Op.identity(), Scalar[Op.value_dtype](2))
    for _ in range(24):
        state = Op.combine(state, state)
    extra = Op.step_scalar(Op.identity(), Scalar[Op.value_dtype](6))
    merged = Op.combine(state, extra)
    assert_equal(
        op.finalize(merged),
        Scalar[Op.out_dtype](Float64(33_554_438) / Float64(16_777_217)),
    )


def main() raises:
    test_mean_state(NanMean[DType.float32]())
    test_mean_state(NanMean[DType.float64]())
    var np = Python.import_module("numpy")
    var axes = DimArray(fill=0)
    axes[0] = 0
    axes[1] = 2
    for parallel in range(2):
        var threshold = 1 << 60
        if parallel:
            threshold = 0
        var a = np.full(Python.tuple(6, 3, 8), np.nan)
        a[0, 0, 0] = 0.0
        a[5, 1, 6] = np.inf
        var result = run(RecordingAllNan(), a, axes, 2, threshold)
        assert_equal(Int(py=result[0]), 4)
        assert_equal(Int(py=result[1]), 48)
        assert_equal(Int(py=result[2]), 48)

        var view = Python.evaluate("lambda a: a[:, :, ::2]")(a)
        result = run(RecordingAllNan(), view, axes, 2, threshold)
        assert_equal(Int(py=result[0]), 1)
        assert_equal(Int(py=result[1]), 24)
        assert_equal(Int(py=result[2]), 24)

    axes[0] = 1
    var rows = np.full(Python.tuple(3, 65), np.nan)
    rows[0, 0] = 0.0
    rows[1, 64] = np.inf
    var result = run(RecordingAllNan(), rows, axes, 1, 1 << 60)
    assert_equal(Int(py=result[0]), 4)
    assert_equal(Int(py=result[1]), 65)
    assert_equal(Int(py=result[2]), 65)
