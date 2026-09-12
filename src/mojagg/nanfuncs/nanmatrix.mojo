"""NaN-aware covariance and correlation matrices."""

from std.algorithm import vectorize
from std.collections import Span
from std.math import isnan, sqrt
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero
from mojagg.core.tensor_view import TensorArg
from mojagg.drivers.gufunc import GUFuncOperation


struct PairwiseAcc(Copyable):
    var count: Float64
    var sum_x: Float64
    var sum_y: Float64
    var sum_xx: Float64
    var sum_yy: Float64
    var sum_xy: Float64

    @always_inline
    def __init__(out self):
        self.count = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xx = 0.0
        self.sum_yy = 0.0
        self.sum_xy = 0.0

    @always_inline
    def __copyinit__(out self, other: Self):
        self.count = other.count
        self.sum_x = other.sum_x
        self.sum_y = other.sum_y
        self.sum_xx = other.sum_xx
        self.sum_yy = other.sum_yy
        self.sum_xy = other.sum_xy


trait MatrixPairwiseOp:
    comptime out_dtype: DType

    @staticmethod
    def finalize(acc: PairwiseAcc, is_diag: Bool) -> Scalar[Self.out_dtype]:
        ...


@always_inline
def _accumulate_pair_simd[
    dtype: DType
](
    p_i: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    p_j: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    n_obs: Int,
    shift_i: Float64,
    shift_j: Float64,
    is_diag: Bool,
) -> PairwiseAcc:
    """Accumulate one pair with float64 SIMD lanes and pairwise NaN masks."""

    comptime width = simd_width_of[dtype]() * 8
    var count = SIMD[DType.float64, width](0.0)
    var sum_x = SIMD[DType.float64, width](0.0)
    var sum_y = SIMD[DType.float64, width](0.0)
    var sum_xx = SIMD[DType.float64, width](0.0)
    var sum_yy = SIMD[DType.float64, width](0.0)
    var sum_xy = SIMD[DType.float64, width](0.0)
    var zero = SIMD[DType.float64, width](0.0)
    var one = SIMD[DType.float64, width](1.0)
    var shift_x = SIMD[DType.float64, width](shift_i)
    var shift_y = SIMD[DType.float64, width](shift_j)

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {
        imm p_i,
        imm p_j,
        mut count,
        mut sum_x,
        mut sum_y,
        mut sum_xx,
        mut sum_yy,
        mut sum_xy,
        imm zero,
        imm one,
        imm shift_x,
        imm shift_y,
        imm is_diag,
        imm shift_i,
        imm shift_j,
    }:
        if evl == width:
            var raw_x = p_i.unsafe_load[width=width](i)
            var x = raw_x.cast[DType.float64]()
            var missing_x = isnan(raw_x)
            if is_diag:
                var dx = x - shift_x
                var dx2 = dx * dx
                count += missing_x.select(zero, one)
                sum_x += missing_x.select(zero, dx)
                sum_xx += missing_x.select(zero, dx2)
            else:
                var raw_y = p_j.unsafe_load[width=width](i)
                var y = raw_y.cast[DType.float64]()
                var missing = missing_x | isnan(raw_y)
                var dx = x - shift_x
                var dy = y - shift_y
                count += missing.select(zero, one)
                sum_x += missing.select(zero, dx)
                sum_y += missing.select(zero, dy)
                sum_xx += missing.select(zero, dx * dx)
                sum_yy += missing.select(zero, dy * dy)
                sum_xy += missing.select(zero, dx * dy)
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var raw_x = p_i[unsafe_offset=i + lane]
                    var x = Float64(raw_x)
                    if is_diag:
                        if not isnan(raw_x):
                            var dx = x - shift_i
                            count[lane] += 1.0
                            sum_x[lane] += dx
                            sum_xx[lane] += dx * dx
                    else:
                        var raw_y = p_j[unsafe_offset=i + lane]
                        if not isnan(raw_x) and not isnan(raw_y):
                            var y = Float64(raw_y)
                            var dx = x - shift_i
                            var dy = y - shift_j
                            count[lane] += 1.0
                            sum_x[lane] += dx
                            sum_y[lane] += dy
                            sum_xx[lane] += dx * dx
                            sum_yy[lane] += dy * dy
                            sum_xy[lane] += dx * dy

    vectorize[width](n_obs, step)

    var acc = PairwiseAcc()
    acc.count = count.reduce_add()
    acc.sum_x = sum_x.reduce_add()
    acc.sum_xx = sum_xx.reduce_add()
    if is_diag:
        acc.sum_y = acc.sum_x
        acc.sum_yy = acc.sum_xx
        acc.sum_xy = acc.sum_xx
    else:
        acc.sum_y = sum_y.reduce_add()
        acc.sum_yy = sum_yy.reduce_add()
        acc.sum_xy = sum_xy.reduce_add()
    return acc^


@always_inline
def _nanmatrix_2d_contiguous[
    Op: MatrixPairwiseOp
](
    source: Span[Scalar[Op.out_dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[Op.out_dtype], MutUntrackedOrigin],
    n_vars: Int,
    n_obs: Int,
):
    """Compute one matrix core received from the tuple based GUFunc."""

    var nan_value = nan_or_zero[Op.out_dtype]()
    var n2 = n_vars * n_vars
    if n_vars == 0:
        return
    if n_obs == 0:
        for i in range(n2):
            destination[i] = nan_value
        return

    var source_ptr = source.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()

    # Keep the per-variable shifts in the output diagonal until all off-diagonal
    # pairs have consumed them.  A shift is an input value, so storing it in the
    # input dtype preserves it exactly for both float32 and float64 matrices.
    for i in range(n_vars):
        var shift = Float64(0.0)
        var row_offset = i * n_obs
        for k in range(n_obs):
            var value = source_ptr[unsafe_offset=row_offset + k]
            if not isnan(value):
                shift = Float64(value)
                break
        destination_ptr[unsafe_offset=i * n_vars + i] = Scalar[Op.out_dtype](
            shift
        )

    for i in range(n_vars):
        var p_i = source_ptr.unsafe_offset(i * n_obs)
        var shift_i = Float64(destination_ptr[unsafe_offset=i * n_vars + i])
        var row_offset = i * n_vars
        for j in range(i + 1, n_vars):
            var p_j = source_ptr.unsafe_offset(j * n_obs)
            var shift_j = Float64(destination_ptr[unsafe_offset=j * n_vars + j])
            var acc = _accumulate_pair_simd[Op.out_dtype](
                p_i, p_j, n_obs, shift_i, shift_j, False
            )
            var value = Op.finalize(acc, False)
            destination_ptr[unsafe_offset=row_offset + j] = value
            destination_ptr[unsafe_offset=j * n_vars + i] = value

    for i in range(n_vars):
        var p_i = source_ptr.unsafe_offset(i * n_obs)
        var shift_i = Float64(destination_ptr[unsafe_offset=i * n_vars + i])
        var acc = _accumulate_pair_simd[Op.out_dtype](
            p_i, p_i, n_obs, shift_i, shift_i, True
        )
        destination_ptr[unsafe_offset=i * n_vars + i] = Op.finalize(acc, True)


struct NanCovOp[dtype: DType](
    GUFuncOperation, ImplicitlyCopyable, MatrixPairwiseOp
):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime Tensors = Tuple[
        TensorArg[Self.value_dtype, False],
        TensorArg[Self.out_dtype, True],
    ]

    var n_vars: Int
    var n_obs: Int

    def __init__(out self, n_vars: Int, n_obs: Int):
        self.n_vars = n_vars
        self.n_obs = n_obs

    @always_inline
    @staticmethod
    def finalize(acc: PairwiseAcc, is_diag: Bool) -> Scalar[Self.out_dtype]:
        if acc.count <= 1.0:
            return nan_or_zero[Self.out_dtype]()
        var mean_i = acc.sum_x / acc.count
        var mean_j = acc.sum_y / acc.count
        var covariance = (acc.sum_xy / acc.count) - (mean_i * mean_j)
        var result = covariance * acc.count / (acc.count - 1.0)
        if is_diag and result < 0.0:
            result = 0.0
        return Scalar[Self.out_dtype](result)

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        _nanmatrix_2d_contiguous[Self](
            input.read_span(), output.write_span(), self.n_vars, self.n_obs
        )


struct NanCorrOp[dtype: DType](
    GUFuncOperation, ImplicitlyCopyable, MatrixPairwiseOp
):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime Tensors = Tuple[
        TensorArg[Self.value_dtype, False],
        TensorArg[Self.out_dtype, True],
    ]

    var n_vars: Int
    var n_obs: Int

    def __init__(out self, n_vars: Int, n_obs: Int):
        self.n_vars = n_vars
        self.n_obs = n_obs

    @always_inline
    @staticmethod
    def finalize(acc: PairwiseAcc, is_diag: Bool) -> Scalar[Self.out_dtype]:
        if acc.count <= 1.0:
            return nan_or_zero[Self.out_dtype]()
        if is_diag:
            var mean = acc.sum_x / acc.count
            var variance = (acc.sum_xx / acc.count) - (mean * mean)
            if variance <= 0.0:
                return nan_or_zero[Self.out_dtype]()
            return Scalar[Self.out_dtype](1.0)

        var mean_i = acc.sum_x / acc.count
        var mean_j = acc.sum_y / acc.count
        var covariance = (acc.sum_xy / acc.count) - (mean_i * mean_j)
        var variance_i = (acc.sum_xx / acc.count) - (mean_i * mean_i)
        var variance_j = (acc.sum_yy / acc.count) - (mean_j * mean_j)
        if variance_i <= 0.0 or variance_j <= 0.0:
            return nan_or_zero[Self.out_dtype]()
        var correlation = covariance / sqrt(variance_i * variance_j)
        if correlation > 1.0:
            correlation = 1.0
        elif correlation < -1.0:
            correlation = -1.0
        return Scalar[Self.out_dtype](correlation)

    @always_inline
    def apply(mut self, tensors: Self.Tensors):
        var input = tensors[0].copy()
        var output = tensors[1].copy()
        _nanmatrix_2d_contiguous[Self](
            input.read_span(), output.write_span(), self.n_vars, self.n_obs
        )
