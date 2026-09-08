"""Covariance and correlation matrix kernels."""

from std.math import isnan, sqrt
from std.memory import alloc, dealloc, Layout
from std.sys.info import simd_width_of
from max.algorithm import parallelize

from mojagg.core.numeric import nan_or_zero


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


struct NanCovOp[dtype: DType](MatrixPairwiseOp):
    comptime out_dtype = Self.dtype

    @always_inline
    @staticmethod
    def finalize(acc: PairwiseAcc, is_diag: Bool) -> Scalar[Self.out_dtype]:
        if acc.count <= 1.0:
            return nan_or_zero[Self.out_dtype]()
        var mean_i = acc.sum_x / acc.count
        var mean_j = acc.sum_y / acc.count
        var cov = (acc.sum_xy / acc.count) - (mean_i * mean_j)
        var cov_unbiased = cov * acc.count / (acc.count - 1.0)
        if is_diag and cov_unbiased < 0.0:
            cov_unbiased = 0.0
        return Scalar[Self.out_dtype](cov_unbiased)


struct NanCorrOp[dtype: DType](MatrixPairwiseOp):
    comptime out_dtype = Self.dtype

    @always_inline
    @staticmethod
    def finalize(acc: PairwiseAcc, is_diag: Bool) -> Scalar[Self.out_dtype]:
        if acc.count <= 1.0:
            return nan_or_zero[Self.out_dtype]()
        if is_diag:
            var mean_i = acc.sum_x / acc.count
            var var_i = (acc.sum_xx / acc.count) - (mean_i * mean_i)
            if var_i <= 0.0:
                return nan_or_zero[Self.out_dtype]()
            return Scalar[Self.out_dtype](1.0)
        var mean_i = acc.sum_x / acc.count
        var mean_j = acc.sum_y / acc.count
        var cov = (acc.sum_xy / acc.count) - (mean_i * mean_j)
        var var_i = (acc.sum_xx / acc.count) - (mean_i * mean_i)
        var var_j = (acc.sum_yy / acc.count) - (mean_j * mean_j)
        if var_i <= 0.0 or var_j <= 0.0:
            return nan_or_zero[Self.out_dtype]()
        var denom = sqrt(var_i * var_j)
        var corr = cov / denom
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0
        return Scalar[Self.out_dtype](corr)


@always_inline
def _accumulate_pair_simd[
    dtype: DType
](
    p_i: Pointer[mut=True, Scalar[dtype], MutAnyOrigin],
    p_j: Pointer[mut=True, Scalar[dtype], MutAnyOrigin],
    n_obs: Int,
    shift_i: Float64,
    shift_j: Float64,
    is_diag: Bool,
) -> PairwiseAcc:
    var acc = PairwiseAcc()
    comptime W = simd_width_of[dtype]()

    var vec_count = SIMD[dtype, W](0.0)
    var vec_sx = SIMD[dtype, W](0.0)
    var vec_sy = SIMD[dtype, W](0.0)
    var vec_sxx = SIMD[dtype, W](0.0)
    var vec_syy = SIMD[dtype, W](0.0)
    var vec_sxy = SIMD[dtype, W](0.0)

    var k = 0
    var zero = SIMD[dtype, W](0.0)
    var one = SIMD[dtype, W](1.0)
    var s_i = SIMD[dtype, W](Scalar[dtype](shift_i))
    var s_j = SIMD[dtype, W](Scalar[dtype](shift_j))

    if is_diag:
        while k + W <= n_obs:
            var vx = p_i.unsafe_load[width=W](k)
            var mask = ~isnan(vx)
            var dx = mask.select(vx - s_i, zero)
            vec_count += mask.select(one, zero)
            vec_sx += dx
            vec_sxx += dx * dx
            k += W

        acc.count = Float64(vec_count.reduce_add())
        acc.sum_x = Float64(vec_sx.reduce_add())
        acc.sum_xx = Float64(vec_sxx.reduce_add())
        acc.sum_y = acc.sum_x
        acc.sum_yy = acc.sum_xx
        acc.sum_xy = acc.sum_xx

        while k < n_obs:
            var rx = Float64(p_i[unsafe_offset=k])
            if not isnan(rx):
                var dx = rx - shift_i
                acc.count += 1.0
                acc.sum_x += dx
                acc.sum_y += dx
                acc.sum_xx += dx * dx
                acc.sum_yy += dx * dx
                acc.sum_xy += dx * dx
            k += 1
    else:
        while k + W <= n_obs:
            var vx = p_i.unsafe_load[width=W](k)
            var vy = p_j.unsafe_load[width=W](k)
            var mask = (~isnan(vx)) & (~isnan(vy))
            var dx = mask.select(vx - s_i, zero)
            var dy = mask.select(vy - s_j, zero)
            vec_count += mask.select(one, zero)
            vec_sx += dx
            vec_sy += dy
            vec_sxx += dx * dx
            vec_syy += dy * dy
            vec_sxy += dx * dy
            k += W

        acc.count = Float64(vec_count.reduce_add())
        acc.sum_x = Float64(vec_sx.reduce_add())
        acc.sum_y = Float64(vec_sy.reduce_add())
        acc.sum_xx = Float64(vec_sxx.reduce_add())
        acc.sum_yy = Float64(vec_syy.reduce_add())
        acc.sum_xy = Float64(vec_sxy.reduce_add())

        while k < n_obs:
            var rx = Float64(p_i[unsafe_offset=k])
            var ry = Float64(p_j[unsafe_offset=k])
            if not isnan(rx) and not isnan(ry):
                var dx = rx - shift_i
                var dy = ry - shift_j
                acc.count += 1.0
                acc.sum_x += dx
                acc.sum_y += dy
                acc.sum_xx += dx * dx
                acc.sum_yy += dy * dy
                acc.sum_xy += dx * dy
            k += 1

    return acc^


@always_inline
def _nanmatrix_2d_serial[
    Op: MatrixPairwiseOp
](
    src: Pointer[mut=True, Scalar[Op.out_dtype], MutAnyOrigin],
    dst: Pointer[mut=True, Scalar[Op.out_dtype], MutAnyOrigin],
    n_vars: Int,
    n_obs: Int,
):
    var nan_val = nan_or_zero[Op.out_dtype]()
    var n2 = n_vars * n_vars

    if n_vars == 0:
        return

    if n_obs == 0:
        for i in range(n2):
            dst[unsafe_offset=i] = nan_val
        return

    # 1. Shift per variable (first valid observation as float64, or 0.0)
    var alloc_shift = alloc(Layout[Float64](count=n_vars))
    var shift = alloc_shift.unsafe_ptr()
    for i in range(n_vars):
        var s = Float64(0.0)
        var row_offset = i * n_obs
        for k in range(n_obs):
            var v = src[unsafe_offset=row_offset + k]
            if not isnan(v):
                s = Float64(v)
                break
        shift[unsafe_offset=i] = s

    for i in range(n_vars):
        var p_i = src.unsafe_offset(i * n_obs)
        var s_i = shift[unsafe_offset=i]
        var row_idx = i * n_vars
        for j in range(i, n_vars):
            var p_j = src.unsafe_offset(j * n_obs)
            var s_j = shift[unsafe_offset=j]
            var acc = _accumulate_pair_simd[Op.out_dtype](
                p_i, p_j, n_obs, s_i, s_j, i == j
            )
            var val = Op.finalize(acc, i == j)
            dst[unsafe_offset=row_idx + j] = val
            dst[unsafe_offset=j * n_vars + i] = val

    dealloc(alloc_shift^)


def matrix_batch[
    Op: MatrixPairwiseOp
](
    src_addr: Int,
    dst_addr: Int,
    batch: Int,
    n_vars: Int,
    n_obs: Int,
    threshold: Int,
    workers: Int,
):
    var src = Pointer[mut=True, Scalar[Op.out_dtype], MutAnyOrigin](
        unsafe_from_address=src_addr
    )
    var dst = Pointer[mut=True, Scalar[Op.out_dtype], MutAnyOrigin](
        unsafe_from_address=dst_addr
    )
    var in_slice_size = n_vars * n_obs
    var out_slice_size = n_vars * n_vars
    var total_pairs = n_vars * (n_vars + 1) // 2
    var total_elements = batch * total_pairs * n_obs

    var effective_workers = workers if workers > 0 else 16

    if batch > 1:
        if total_elements >= threshold and effective_workers > 1:
            var num_workers = min(effective_workers, batch)
            var chunk = (batch + num_workers - 1) // num_workers

            def batch_worker(
                w: Int,
            ) {
                imm src_addr,
                imm dst_addr,
                imm chunk,
                imm batch,
                imm n_vars,
                imm n_obs,
                imm in_slice_size,
                imm out_slice_size,
            }:
                var p_src = Pointer[
                    mut=True, Scalar[Op.out_dtype], MutAnyOrigin
                ](unsafe_from_address=src_addr)
                var p_dst = Pointer[
                    mut=True, Scalar[Op.out_dtype], MutAnyOrigin
                ](unsafe_from_address=dst_addr)
                var start = w * chunk
                var end = min(start + chunk, batch)
                for b in range(start, end):
                    _nanmatrix_2d_serial[Op](
                        p_src.unsafe_offset(b * in_slice_size),
                        p_dst.unsafe_offset(b * out_slice_size),
                        n_vars,
                        n_obs,
                    )

            parallelize(batch_worker, num_workers)
        else:
            for b in range(batch):
                _nanmatrix_2d_serial[Op](
                    src.unsafe_offset(b * in_slice_size),
                    dst.unsafe_offset(b * out_slice_size),
                    n_vars,
                    n_obs,
                )
    else:
        # Single 2D slice
        if (
            n_vars >= 4
            and total_elements >= threshold
            and effective_workers > 1
        ):
            var nan_val = nan_or_zero[Op.out_dtype]()
            if n_vars == 0:
                return
            if n_obs == 0:
                for i in range(n_vars * n_vars):
                    dst[unsafe_offset=i] = nan_val
                return

            var alloc_shift = alloc(Layout[Float64](count=n_vars))
            var shift = alloc_shift.unsafe_ptr()
            for i in range(n_vars):
                var s = Float64(0.0)
                var row_offset = i * n_obs
                for k in range(n_obs):
                    var v = src[unsafe_offset=row_offset + k]
                    if not isnan(v):
                        s = Float64(v)
                        break
                shift[unsafe_offset=i] = s

            var shift_addr = Int(shift)
            var num_workers = min(effective_workers, n_vars)
            var chunk = (n_vars + num_workers - 1) // num_workers

            def row_worker(
                w: Int,
            ) {
                imm src_addr,
                imm dst_addr,
                imm shift_addr,
                imm chunk,
                imm n_vars,
                imm n_obs,
            }:
                var p_src = Pointer[
                    mut=True, Scalar[Op.out_dtype], MutAnyOrigin
                ](unsafe_from_address=src_addr)
                var p_dst = Pointer[
                    mut=True, Scalar[Op.out_dtype], MutAnyOrigin
                ](unsafe_from_address=dst_addr)
                var p_shift = Pointer[mut=True, Float64, MutAnyOrigin](
                    unsafe_from_address=shift_addr
                )
                var start = w * chunk
                var end = min(start + chunk, n_vars)
                for i in range(start, end):
                    var p_i = p_src.unsafe_offset(i * n_obs)
                    var s_i = p_shift[unsafe_offset=i]
                    var row_idx = i * n_vars
                    for j in range(i, n_vars):
                        var p_j = p_src.unsafe_offset(j * n_obs)
                        var s_j = p_shift[unsafe_offset=j]
                        var acc = _accumulate_pair_simd[Op.out_dtype](
                            p_i, p_j, n_obs, s_i, s_j, i == j
                        )
                        var val = Op.finalize(acc, i == j)
                        p_dst[unsafe_offset=row_idx + j] = val

            parallelize(row_worker, num_workers)

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    dst[unsafe_offset=j * n_vars + i] = dst[
                        unsafe_offset=i * n_vars + j
                    ]

            dealloc(alloc_shift^)
        else:
            _nanmatrix_2d_serial[Op](src, dst, n_vars, n_obs)
