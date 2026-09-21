"""Shared execution and accumulators for NaN-aware matrix operations."""

from std.algorithm import vectorize
from std.collections import Span
from max.algorithm import parallelize
from std.math import isnan
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero


struct PairwiseAcc[dtype: DType](Copyable):
    var count: Scalar[Self.dtype]
    var sum_x: Scalar[Self.dtype]
    var sum_y: Scalar[Self.dtype]
    var sum_xx: Scalar[Self.dtype]
    var sum_yy: Scalar[Self.dtype]
    var sum_xy: Scalar[Self.dtype]

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
    def accumulate(
        p_i: Pointer[mut=False, Scalar[Self.out_dtype], ImmUntrackedOrigin],
        p_j: Pointer[mut=False, Scalar[Self.out_dtype], ImmUntrackedOrigin],
        n_obs: Int,
        shift_i: Scalar[Self.out_dtype],
        shift_j: Scalar[Self.out_dtype],
        is_diag: Bool,
    ) -> PairwiseAcc[Self.out_dtype]:
        ...

    @staticmethod
    def finalize(
        acc: PairwiseAcc[Self.out_dtype], is_diag: Bool
    ) -> Scalar[Self.out_dtype]:
        ...


@always_inline
def _accumulate_pair_simd[
    dtype: DType
](
    p_i: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    p_j: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    n_obs: Int,
    shift_i: Scalar[dtype],
    shift_j: Scalar[dtype],
    is_diag: Bool,
) -> PairwiseAcc[dtype]:
    """Accumulate one pair with dtype-native SIMD lanes and pairwise NaN masks.
    """

    # This kernel carries six dtype-native accumulators.  Expanding beyond the
    # native SIMD width spills those accumulators and is slower in practice.
    comptime width = simd_width_of[dtype]()
    var count = SIMD[dtype, width](0.0)
    var sum_x = SIMD[dtype, width](0.0)
    var sum_y = SIMD[dtype, width](0.0)
    var sum_xx = SIMD[dtype, width](0.0)
    var sum_yy = SIMD[dtype, width](0.0)
    var sum_xy = SIMD[dtype, width](0.0)
    var zero = SIMD[dtype, width](0.0)
    var one = SIMD[dtype, width](1.0)
    var shift_x = SIMD[dtype, width](shift_i)
    var shift_y = SIMD[dtype, width](shift_j)

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
            var x = raw_x
            var missing_x = isnan(raw_x)
            if is_diag:
                var dx = x - shift_x
                var dx2 = dx * dx
                count += missing_x.select(zero, one)
                sum_x += missing_x.select(zero, dx)
                sum_xx += missing_x.select(zero, dx2)
            else:
                var raw_y = p_j.unsafe_load[width=width](i)
                var y = raw_y
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
                    var x = raw_x
                    if is_diag:
                        if not isnan(raw_x):
                            var dx = x - shift_i
                            count[lane] += 1.0
                            sum_x[lane] += dx
                            sum_xx[lane] += dx * dx
                    else:
                        var raw_y = p_j[unsafe_offset=i + lane]
                        if not isnan(raw_x) and not isnan(raw_y):
                            var y = raw_y
                            var dx = x - shift_i
                            var dy = y - shift_j
                            count[lane] += 1.0
                            sum_x[lane] += dx
                            sum_y[lane] += dy
                            sum_xx[lane] += dx * dx
                            sum_yy[lane] += dy * dy
                            sum_xy[lane] += dx * dy

    vectorize[width, unroll_factor=1](n_obs, step)

    var acc = PairwiseAcc[dtype]()
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
def _accumulate_cov_diag_simd[
    dtype: DType
](
    p_i: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    n_obs: Int,
    shift_i: Scalar[dtype],
) -> PairwiseAcc[dtype]:
    """Accumulate a covariance diagonal without unused pair accumulators."""

    comptime width = simd_width_of[dtype]()
    var count = SIMD[dtype, width](0.0)
    var sum_x = SIMD[dtype, width](0.0)
    var sum_xx = SIMD[dtype, width](0.0)
    var zero = SIMD[dtype, width](0.0)
    var one = SIMD[dtype, width](1.0)
    var shift_x = SIMD[dtype, width](shift_i)

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {
        imm p_i,
        mut count,
        mut sum_x,
        mut sum_xx,
        imm zero,
        imm one,
        imm shift_x,
        imm shift_i,
    }:
        if evl == width:
            var raw_x = p_i.unsafe_load[width=width](i)
            var x = raw_x
            var missing_x = isnan(raw_x)
            var dx = x - shift_x
            count += missing_x.select(zero, one)
            sum_x += missing_x.select(zero, dx)
            sum_xx += missing_x.select(zero, dx * dx)
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var raw_x = p_i[unsafe_offset=i + lane]
                    if not isnan(raw_x):
                        var dx = raw_x - shift_i
                        count[lane] += 1.0
                        sum_x[lane] += dx
                        sum_xx[lane] += dx * dx

    vectorize[width, unroll_factor=1](n_obs, step)

    var acc = PairwiseAcc[dtype]()
    acc.count = count.reduce_add()
    acc.sum_x = sum_x.reduce_add()
    acc.sum_y = acc.sum_x
    acc.sum_xx = sum_xx.reduce_add()
    acc.sum_yy = acc.sum_xx
    acc.sum_xy = acc.sum_xx
    return acc^


@always_inline
def _accumulate_cov_offdiag_simd[
    dtype: DType
](
    p_i: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    p_j: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    n_obs: Int,
    shift_i: Scalar[dtype],
    shift_j: Scalar[dtype],
) -> PairwiseAcc[dtype]:
    """Accumulate covariance off-diagonals with only needed statistics."""

    comptime width = simd_width_of[dtype]()
    var count = SIMD[dtype, width](0.0)
    var sum_x = SIMD[dtype, width](0.0)
    var sum_y = SIMD[dtype, width](0.0)
    var sum_xy = SIMD[dtype, width](0.0)
    var zero = SIMD[dtype, width](0.0)
    var one = SIMD[dtype, width](1.0)
    var shift_x = SIMD[dtype, width](shift_i)
    var shift_y = SIMD[dtype, width](shift_j)

    def step[
        vector_width: Int
    ](i: Int, evl: Int) {
        imm p_i,
        imm p_j,
        mut count,
        mut sum_x,
        mut sum_y,
        mut sum_xy,
        imm zero,
        imm one,
        imm shift_x,
        imm shift_y,
        imm shift_i,
        imm shift_j,
    }:
        if evl == width:
            var raw_x = p_i.unsafe_load[width=width](i)
            var raw_y = p_j.unsafe_load[width=width](i)
            var x = raw_x
            var y = raw_y
            var missing = isnan(raw_x) | isnan(raw_y)
            var dx = x - shift_x
            var dy = y - shift_y
            count += missing.select(zero, one)
            sum_x += missing.select(zero, dx)
            sum_y += missing.select(zero, dy)
            sum_xy += missing.select(zero, dx * dy)
        else:
            comptime for lane in range(width):
                if lane < evl:
                    var raw_x = p_i[unsafe_offset=i + lane]
                    var raw_y = p_j[unsafe_offset=i + lane]
                    if not isnan(raw_x) and not isnan(raw_y):
                        var dx = raw_x - shift_i
                        var dy = raw_y - shift_j
                        count[lane] += 1.0
                        sum_x[lane] += dx
                        sum_y[lane] += dy
                        sum_xy[lane] += dx * dy

    vectorize[width, unroll_factor=1](n_obs, step)

    var acc = PairwiseAcc[dtype]()
    acc.count = count.reduce_add()
    acc.sum_x = sum_x.reduce_add()
    acc.sum_y = sum_y.reduce_add()
    acc.sum_xy = sum_xy.reduce_add()
    return acc^


@always_inline
def _accumulate_cov_pair_simd[
    dtype: DType
](
    p_i: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    p_j: Pointer[mut=False, Scalar[dtype], ImmUntrackedOrigin],
    n_obs: Int,
    shift_i: Scalar[dtype],
    shift_j: Scalar[dtype],
    is_diag: Bool,
) -> PairwiseAcc[dtype]:
    """Select the compact covariance reducer for a diagonal or pair."""

    if is_diag:
        return _accumulate_cov_diag_simd[dtype](p_i, n_obs, shift_i)
    return _accumulate_cov_offdiag_simd[dtype](
        p_i, p_j, n_obs, shift_i, shift_j
    )


@always_inline
def _pair_tile_count(n_vars: Int, tile_size: Int) -> Int:
    var tile_count = (n_vars + tile_size - 1) // tile_size
    return tile_count * (tile_count + 1) // 2


@always_inline
def _pair_tile_coordinates(tile_index: Int, tile_count: Int) -> Tuple[Int, Int]:
    """Map a packed upper-triangle tile index to (row, column).

    The binary search keeps scheduling overhead logarithmic in the number of
    tiles.  A linear row walk becomes visible for very wide matrices because
    the packed tile count is quadratic in the variable count.
    """

    var low = 0
    var high = tile_count
    while low < high:
        var middle = (low + high) // 2
        var row_start = middle * tile_count - middle * (middle - 1) // 2
        if row_start <= tile_index:
            low = middle + 1
        else:
            high = middle

    var row = low - 1
    var row_start = row * tile_count - row * (row - 1) // 2
    return (row, row + tile_index - row_start)


@always_inline
def _compute_pair_tile[
    Op: MatrixPairwiseOp
](
    source_address: Int,
    destination_address: Int,
    n_vars: Int,
    n_obs: Int,
    tile_size: Int,
    tile_index: Int,
):
    """Compute one disjoint upper-triangle tile and its mirror."""

    var source_ptr = Pointer[
        mut=False, Scalar[Op.out_dtype], ImmUntrackedOrigin
    ](unsafe_from_address=source_address)
    var destination_ptr = Pointer[mut=True, Scalar[Op.out_dtype], MutAnyOrigin](
        unsafe_from_address=destination_address
    )
    var tile_count = (n_vars + tile_size - 1) // tile_size
    var coordinates = _pair_tile_coordinates(tile_index, tile_count)
    var tile_i = coordinates[0]
    var tile_j = coordinates[1]
    var i_start = tile_i * tile_size
    var i_stop = min(i_start + tile_size, n_vars)
    var j_start = tile_j * tile_size
    var j_stop = min(j_start + tile_size, n_vars)

    for i in range(i_start, i_stop):
        var p_i = source_ptr.unsafe_offset(i * n_obs)
        var shift_i = destination_ptr[unsafe_offset=i * n_vars + i]
        var first_j = max(j_start, i + 1)
        for j in range(first_j, j_stop):
            var p_j = source_ptr.unsafe_offset(j * n_obs)
            var shift_j = destination_ptr[unsafe_offset=j * n_vars + j]
            var acc = Op.accumulate(p_i, p_j, n_obs, shift_i, shift_j, False)
            var value = Op.finalize(acc, False)
            destination_ptr[unsafe_offset=i * n_vars + j] = value
            destination_ptr[unsafe_offset=j * n_vars + i] = value


def _run_pair_tiles[
    Op: MatrixPairwiseOp
](
    source_address: Int,
    destination_address: Int,
    n_vars: Int,
    n_obs: Int,
    inner_workers: Int,
):
    """Run balanced upper-triangle tiles, with a barrier before diagonals."""

    comptime tile_size = 8
    var tile_count = _pair_tile_count(n_vars, tile_size)
    var tasks = min(max(inner_workers, 1), tile_count)
    if tasks <= 1:
        for tile_index in range(tile_count):
            _compute_pair_tile[Op](
                source_address,
                destination_address,
                n_vars,
                n_obs,
                tile_size,
                tile_index,
            )
        return

    var chunk = (tile_count + tasks - 1) // tasks

    def worker(
        index: Int,
    ) {
        imm source_address,
        imm destination_address,
        imm n_vars,
        imm n_obs,
        imm tile_count,
        imm chunk,
    }:
        var start = index * chunk
        var stop = min(start + chunk, tile_count)
        for tile_index in range(start, stop):
            _compute_pair_tile[Op](
                source_address,
                destination_address,
                n_vars,
                n_obs,
                tile_size,
                tile_index,
            )

    parallelize(worker, tasks)


@always_inline
def _nanmatrix_2d_contiguous[
    Op: MatrixPairwiseOp
](
    source: Span[Scalar[Op.out_dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[Op.out_dtype], MutUntrackedOrigin],
    n_vars: Int,
    n_obs: Int,
    inner_workers: Int,
):
    """Compute one matrix core received from the guvectorize driver."""

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
        var shift = Scalar[Op.out_dtype](0.0)
        var row_offset = i * n_obs
        for k in range(n_obs):
            var value = source_ptr[unsafe_offset=row_offset + k]
            if not isnan(value):
                shift = value
                break
        destination_ptr[unsafe_offset=i * n_vars + i] = shift

    _run_pair_tiles[Op](
        Int(source_ptr),
        Int(destination_ptr),
        n_vars,
        n_obs,
        inner_workers,
    )

    for i in range(n_vars):
        var p_i = source_ptr.unsafe_offset(i * n_obs)
        var shift_i = destination_ptr[unsafe_offset=i * n_vars + i]
        var acc = Op.accumulate(p_i, p_i, n_obs, shift_i, shift_i, True)
        destination_ptr[unsafe_offset=i * n_vars + i] = Op.finalize(acc, True)
