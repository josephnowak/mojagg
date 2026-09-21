"""Pairwise loops shared by the moving matrix kernels."""

from std.collections import Span
from std.math import isnan, sqrt
from std.sys.info import simd_width_of

from mojagg.core.numeric import nan_or_zero


@always_inline
def _leading_rows_shift[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    variable: Int,
    n_rows: Int,
) -> Float64:
    """Return the look-ahead-free rolling offset for one variable."""

    var source_ptr = source.unsafe_ptr()
    var total = Float64(0.0)
    var count = 0
    var limit = min(n_rows, n_obs)
    for t in range(limit):
        var value = source_ptr[unsafe_offset=t * n_vars + variable]
        if not isnan(value):
            total += Float64(value)
            count += 1

    if count > 0:
        return total / Float64(count)

    # A variable can be all-NaN in the leading rows.  Use its first observed
    # value so a later valid pair does not accumulate values far from zero.
    for t in range(n_obs):
        var value = source_ptr[unsafe_offset=t * n_vars + variable]
        if not isnan(value):
            return Float64(value)
    return Float64(0.0)


@always_inline
def _first_observation_shift[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    variable: Int,
) -> Float64:
    """Return the first non-NaN value for one variable."""

    var source_ptr = source.unsafe_ptr()
    for t in range(n_obs):
        var value = source_ptr[unsafe_offset=t * n_vars + variable]
        if not isnan(value):
            return Float64(value)
    return Float64(0.0)


@always_inline
def _write_pair[
    dtype: DType
](
    destination_ptr: Pointer[mut=True, Scalar[dtype], MutUntrackedOrigin],
    matrix_size: Int,
    n_vars: Int,
    time: Int,
    i: Int,
    j: Int,
    value: Scalar[dtype],
):
    var offset = time * matrix_size + i * n_vars + j
    destination_ptr[unsafe_offset=offset] = value
    if i != j:
        destination_ptr[
            unsafe_offset=time * matrix_size + j * n_vars + i
        ] = value


@always_inline
def _move_cov_pair_full_final[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    i: Int,
    j: Int,
    min_count: Int,
    shift_i: Float64,
    shift_j: Float64,
):
    """Finalize a full window with the static matrix reduction order."""

    comptime width = 1
    var source_ptr = source.unsafe_ptr()
    var count_lanes = SIMD[DType.float64, width](0.0)
    var sum_i_lanes = SIMD[DType.float64, width](0.0)
    var sum_j_lanes = SIMD[DType.float64, width](0.0)
    var product_lanes = SIMD[DType.float64, width](0.0)
    var offset = 0

    while offset < n_obs:
        var active = min(width, n_obs - offset)
        comptime for lane in range(width):
            if lane < active:
                var value_i = source_ptr[
                    unsafe_offset=(offset + lane) * n_vars + i
                ]
                var value_j = source_ptr[
                    unsafe_offset=(offset + lane) * n_vars + j
                ]
                if not isnan(value_i) and not isnan(value_j):
                    var shifted_i = Float64(value_i) - shift_i
                    var shifted_j = Float64(value_j) - shift_j
                    count_lanes[lane] += 1.0
                    sum_i_lanes[lane] += shifted_i
                    sum_j_lanes[lane] += shifted_j
                    product_lanes[lane] += shifted_i * shifted_j
        offset += active

    var count = count_lanes.reduce_add()
    var required = Float64(min_count)
    if required < 1.0:
        required = 1.0
    var nan_value = nan_or_zero[dtype]()
    if count >= required and count > 1.0:
        var sum_i = sum_i_lanes.reduce_add()
        var sum_j = sum_j_lanes.reduce_add()
        var product_sum = product_lanes.reduce_add()
        var mean_i = sum_i / count
        var mean_j = sum_j / count
        var covariance = (
            (product_sum / count - mean_i * mean_j) * count / (count - 1.0)
        )
        if i == j and covariance < 0.0:
            covariance = 0.0
        _write_pair[dtype](
            destination.unsafe_ptr(),
            n_vars * n_vars,
            n_vars,
            n_obs - 1,
            i,
            j,
            Scalar[dtype](covariance),
        )
    else:
        _write_pair[dtype](
            destination.unsafe_ptr(),
            n_vars * n_vars,
            n_vars,
            n_obs - 1,
            i,
            j,
            nan_value,
        )


@always_inline
def _move_corr_pair_full_final[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    i: Int,
    j: Int,
    min_count: Int,
    shift_i: Float64,
    shift_j: Float64,
):
    """Finalize a full window with the static matrix reduction order."""

    comptime width = 1
    var source_ptr = source.unsafe_ptr()
    var count_lanes = SIMD[DType.float64, width](0.0)
    var sum_i_lanes = SIMD[DType.float64, width](0.0)
    var sum_j_lanes = SIMD[DType.float64, width](0.0)
    var sum_i2_lanes = SIMD[DType.float64, width](0.0)
    var sum_j2_lanes = SIMD[DType.float64, width](0.0)
    var product_lanes = SIMD[DType.float64, width](0.0)
    var offset = 0

    while offset < n_obs:
        var active = min(width, n_obs - offset)
        comptime for lane in range(width):
            if lane < active:
                var value_i = source_ptr[
                    unsafe_offset=(offset + lane) * n_vars + i
                ]
                var value_j = source_ptr[
                    unsafe_offset=(offset + lane) * n_vars + j
                ]
                if not isnan(value_i) and not isnan(value_j):
                    var shifted_i = Float64(value_i) - shift_i
                    var shifted_j = Float64(value_j) - shift_j
                    count_lanes[lane] += 1.0
                    sum_i_lanes[lane] += shifted_i
                    sum_j_lanes[lane] += shifted_j
                    sum_i2_lanes[lane] += shifted_i * shifted_i
                    sum_j2_lanes[lane] += shifted_j * shifted_j
                    product_lanes[lane] += shifted_i * shifted_j
        offset += active

    var count = count_lanes.reduce_add()
    var required = Float64(min_count)
    if required < 2.0:
        required = 2.0
    var nan_value = nan_or_zero[dtype]()
    if count >= required:
        var sum_i = sum_i_lanes.reduce_add()
        var sum_j = sum_j_lanes.reduce_add()
        var sum_i2 = sum_i2_lanes.reduce_add()
        var sum_j2 = sum_j2_lanes.reduce_add()
        var product_sum = product_lanes.reduce_add()
        var mean_i = sum_i / count
        var mean_j = sum_j / count
        var variance_i = sum_i2 / count - mean_i * mean_i
        var variance_j = sum_j2 / count - mean_j * mean_j
        if variance_i < 0.0:
            variance_i = 0.0
        if variance_j < 0.0:
            variance_j = 0.0

        if variance_i > 0.0 and variance_j > 0.0:
            var covariance = product_sum / count - mean_i * mean_j
            var correlation: Float64
            if i == j:
                correlation = 1.0
            else:
                correlation = covariance / sqrt(variance_i) / sqrt(variance_j)
            if correlation > 1.0:
                correlation = 1.0
            elif correlation < -1.0:
                correlation = -1.0
            _write_pair[dtype](
                destination.unsafe_ptr(),
                n_vars * n_vars,
                n_vars,
                n_obs - 1,
                i,
                j,
                Scalar[dtype](correlation),
            )
        else:
            _write_pair[dtype](
                destination.unsafe_ptr(),
                n_vars * n_vars,
                n_vars,
                n_obs - 1,
                i,
                j,
                nan_value,
            )
    else:
        _write_pair[dtype](
            destination.unsafe_ptr(),
            n_vars * n_vars,
            n_vars,
            n_obs - 1,
            i,
            j,
            nan_value,
        )


@always_inline
def _move_cov_pair[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    i: Int,
    j: Int,
    window: Int,
    min_count: Int,
    shift_i: Float64,
    shift_j: Float64,
):
    """Compute one rolling covariance pair and its symmetric mirror."""

    var source_ptr = source.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    var matrix_size = n_vars * n_vars
    var nan_value = nan_or_zero[dtype]()
    var sum_i = Float64(0.0)
    var sum_j = Float64(0.0)
    var sum_ij = Float64(0.0)
    var count = Float64(0.0)
    var required = Float64(min_count)
    if required < 1.0:
        required = 1.0

    for t in range(n_obs):
        if t >= window:
            var old_i = source_ptr[unsafe_offset=(t - window) * n_vars + i]
            var old_j = source_ptr[unsafe_offset=(t - window) * n_vars + j]
            if not isnan(old_i) and not isnan(old_j):
                var old_i_shifted = Float64(old_i) - shift_i
                var old_j_shifted = Float64(old_j) - shift_j
                sum_i -= old_i_shifted
                sum_j -= old_j_shifted
                sum_ij -= old_i_shifted * old_j_shifted
                count -= 1.0

        var new_i = source_ptr[unsafe_offset=t * n_vars + i]
        var new_j = source_ptr[unsafe_offset=t * n_vars + j]
        if not isnan(new_i) and not isnan(new_j):
            var new_i_shifted = Float64(new_i) - shift_i
            var new_j_shifted = Float64(new_j) - shift_j
            sum_i += new_i_shifted
            sum_j += new_j_shifted
            sum_ij += new_i_shifted * new_j_shifted
            count += 1.0

        if count >= required and count > 1.0:
            var mean_i = sum_i / count
            var mean_j = sum_j / count
            var covariance = (
                (sum_ij / count - mean_i * mean_j) * count / (count - 1.0)
            )
            if i == j and covariance < 0.0:
                covariance = 0.0
            _write_pair[dtype](
                destination_ptr,
                matrix_size,
                n_vars,
                t,
                i,
                j,
                Scalar[dtype](covariance),
            )
        else:
            _write_pair[dtype](
                destination_ptr,
                matrix_size,
                n_vars,
                t,
                i,
                j,
                nan_value,
            )


@always_inline
def _move_corr_pair[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    i: Int,
    j: Int,
    window: Int,
    min_count: Int,
    shift_i: Float64,
    shift_j: Float64,
):
    """Compute one rolling correlation pair and its symmetric mirror."""

    var source_ptr = source.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    var matrix_size = n_vars * n_vars
    var nan_value = nan_or_zero[dtype]()
    var sum_i = Float64(0.0)
    var sum_j = Float64(0.0)
    var sum_i2 = Float64(0.0)
    var sum_j2 = Float64(0.0)
    var sum_ij = Float64(0.0)
    var count = Float64(0.0)
    var required = Float64(min_count)
    if required < 2.0:
        required = 2.0

    for t in range(n_obs):
        if t >= window:
            var old_i = source_ptr[unsafe_offset=(t - window) * n_vars + i]
            var old_j = source_ptr[unsafe_offset=(t - window) * n_vars + j]
            if not isnan(old_i) and not isnan(old_j):
                var old_i_shifted = Float64(old_i) - shift_i
                var old_j_shifted = Float64(old_j) - shift_j
                sum_i -= old_i_shifted
                sum_j -= old_j_shifted
                sum_i2 -= old_i_shifted * old_i_shifted
                sum_j2 -= old_j_shifted * old_j_shifted
                sum_ij -= old_i_shifted * old_j_shifted
                count -= 1.0

        var new_i = source_ptr[unsafe_offset=t * n_vars + i]
        var new_j = source_ptr[unsafe_offset=t * n_vars + j]
        if not isnan(new_i) and not isnan(new_j):
            var new_i_shifted = Float64(new_i) - shift_i
            var new_j_shifted = Float64(new_j) - shift_j
            sum_i += new_i_shifted
            sum_j += new_j_shifted
            sum_i2 += new_i_shifted * new_i_shifted
            sum_j2 += new_j_shifted * new_j_shifted
            sum_ij += new_i_shifted * new_j_shifted
            count += 1.0

        if count >= required:
            var mean_i = sum_i / count
            var mean_j = sum_j / count
            var variance_i = sum_i2 / count - mean_i * mean_i
            var variance_j = sum_j2 / count - mean_j * mean_j
            if variance_i < 0.0:
                variance_i = 0.0
            if variance_j < 0.0:
                variance_j = 0.0

            if variance_i > 0.0 and variance_j > 0.0:
                var covariance = sum_ij / count - mean_i * mean_j
                # Match the static matrix finalizer and avoid rounding the
                # product of the two variances before taking square roots.
                var correlation = (
                    covariance / sqrt(variance_i) / sqrt(variance_j)
                )
                if correlation > 1.0:
                    correlation = 1.0
                elif correlation < -1.0:
                    correlation = -1.0
                _write_pair[dtype](
                    destination_ptr,
                    matrix_size,
                    n_vars,
                    t,
                    i,
                    j,
                    Scalar[dtype](correlation),
                )
            else:
                _write_pair[dtype](
                    destination_ptr,
                    matrix_size,
                    n_vars,
                    t,
                    i,
                    j,
                    nan_value,
                )
        else:
            _write_pair[dtype](
                destination_ptr,
                matrix_size,
                n_vars,
                t,
                i,
                j,
                nan_value,
            )


@always_inline
def _move_exp_cov_pair[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    alphas: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    i: Int,
    j: Int,
    min_weight: Float64,
    shift_i: Float64,
    shift_j: Float64,
):
    """Compute one exponential moving covariance pair and its mirror."""

    var source_ptr = source.unsafe_ptr()
    var alpha_ptr = alphas.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    var matrix_size = n_vars * n_vars
    var nan_value = nan_or_zero[dtype]()
    var sum_i = Float64(0.0)
    var sum_j = Float64(0.0)
    var sum_ij = Float64(0.0)
    var sum_weight = Float64(0.0)
    var sum_weight_2 = Float64(0.0)
    var weight = Float64(0.0)

    for t in range(n_obs):
        var alpha = Float64(alpha_ptr[unsafe_offset=t * n_vars])
        var decay = 1.0 - alpha
        sum_i *= decay
        sum_j *= decay
        sum_ij *= decay
        sum_weight *= decay
        sum_weight_2 *= decay * decay
        weight *= decay

        var value_i = source_ptr[unsafe_offset=t * n_vars + i]
        var value_j = source_ptr[unsafe_offset=t * n_vars + j]
        if not isnan(value_i) and not isnan(value_j):
            var shifted_i = Float64(value_i) - shift_i
            var shifted_j = Float64(value_j) - shift_j
            sum_i += shifted_i
            sum_j += shifted_j
            sum_ij += shifted_i * shifted_j
            sum_weight += 1.0
            sum_weight_2 += 1.0
            weight += alpha

        if sum_weight != 0.0:
            var bias = 1.0 - sum_weight_2 / (sum_weight * sum_weight)
            if weight >= min_weight and bias > 0.0:
                var mean_i = sum_i / sum_weight
                var mean_j = sum_j / sum_weight
                var covariance = (sum_ij / sum_weight - mean_i * mean_j) / bias
                if i == j and covariance < 0.0:
                    covariance = 0.0
                _write_pair[dtype](
                    destination_ptr,
                    matrix_size,
                    n_vars,
                    t,
                    i,
                    j,
                    Scalar[dtype](covariance),
                )
            else:
                _write_pair[dtype](
                    destination_ptr,
                    matrix_size,
                    n_vars,
                    t,
                    i,
                    j,
                    nan_value,
                )
        else:
            _write_pair[dtype](
                destination_ptr,
                matrix_size,
                n_vars,
                t,
                i,
                j,
                nan_value,
            )


@always_inline
def _move_exp_corr_pair[
    dtype: DType
](
    source: Span[Scalar[dtype], ImmUntrackedOrigin],
    alphas: Span[Scalar[dtype], ImmUntrackedOrigin],
    destination: Span[Scalar[dtype], MutUntrackedOrigin],
    n_obs: Int,
    n_vars: Int,
    i: Int,
    j: Int,
    min_weight: Float64,
    shift_i: Float64,
    shift_j: Float64,
):
    """Compute one exponential moving correlation pair and its mirror."""

    var source_ptr = source.unsafe_ptr()
    var alpha_ptr = alphas.unsafe_ptr()
    var destination_ptr = destination.unsafe_ptr()
    var matrix_size = n_vars * n_vars
    var nan_value = nan_or_zero[dtype]()
    var sum_i = Float64(0.0)
    var sum_j = Float64(0.0)
    var sum_i2 = Float64(0.0)
    var sum_j2 = Float64(0.0)
    var sum_ij = Float64(0.0)
    var sum_weight = Float64(0.0)
    var sum_weight_2 = Float64(0.0)
    var weight = Float64(0.0)

    for t in range(n_obs):
        var alpha = Float64(alpha_ptr[unsafe_offset=t * n_vars])
        var decay = 1.0 - alpha
        sum_i *= decay
        sum_j *= decay
        sum_i2 *= decay
        sum_j2 *= decay
        sum_ij *= decay
        sum_weight *= decay
        sum_weight_2 *= decay * decay
        weight *= decay

        var value_i = source_ptr[unsafe_offset=t * n_vars + i]
        var value_j = source_ptr[unsafe_offset=t * n_vars + j]
        if not isnan(value_i) and not isnan(value_j):
            var shifted_i = Float64(value_i) - shift_i
            var shifted_j = Float64(value_j) - shift_j
            sum_i += shifted_i
            sum_j += shifted_j
            sum_i2 += shifted_i * shifted_i
            sum_j2 += shifted_j * shifted_j
            sum_ij += shifted_i * shifted_j
            sum_weight += 1.0
            sum_weight_2 += 1.0
            weight += alpha

        if sum_weight != 0.0:
            var bias = 1.0 - sum_weight_2 / (sum_weight * sum_weight)
            if weight >= min_weight and bias > 0.0:
                var mean_i = sum_i / sum_weight
                var mean_j = sum_j / sum_weight
                var variance_i = (sum_i2 / sum_weight - mean_i * mean_i) / bias
                var variance_j = (sum_j2 / sum_weight - mean_j * mean_j) / bias
                if variance_i < 0.0:
                    variance_i = 0.0
                if variance_j < 0.0:
                    variance_j = 0.0

                if variance_i > 0.0 and variance_j > 0.0:
                    var covariance = (
                        sum_ij / sum_weight - mean_i * mean_j
                    ) / bias
                    var correlation = (
                        covariance / sqrt(variance_i) / sqrt(variance_j)
                    )
                    if correlation > 1.0:
                        correlation = 1.0
                    elif correlation < -1.0:
                        correlation = -1.0
                    _write_pair[dtype](
                        destination_ptr,
                        matrix_size,
                        n_vars,
                        t,
                        i,
                        j,
                        Scalar[dtype](correlation),
                    )
                else:
                    _write_pair[dtype](
                        destination_ptr,
                        matrix_size,
                        n_vars,
                        t,
                        i,
                        j,
                        nan_value,
                    )
            else:
                _write_pair[dtype](
                    destination_ptr,
                    matrix_size,
                    n_vars,
                    t,
                    i,
                    j,
                    nan_value,
                )
        else:
            _write_pair[dtype](
                destination_ptr,
                matrix_size,
                n_vars,
                t,
                i,
                j,
                nan_value,
            )
