"""NaN-aware trailing moving correlation matrices."""

from std.collections import Span

from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)
from mojagg.moving.moving_matrix_helpers import (
    _first_observation_shift,
    _leading_rows_shift,
    _move_corr_pair,
    _move_corr_pair_full_final,
)


struct MoveCorrMatrixKernel[dtype: DType](GUFuncKernel, ImplicitlyCopyable):
    """SIMD-driver-compatible ``(m,n) -> (m,n,n)`` rolling correlation."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0], Dim[1]]],
        GUTensor[
            Self.dtype,
            True,
            CoreSpec[Dim[0], Dim[1], Dim[1]],
        ],
    ]

    var n_vars: Int
    var n_obs: Int
    var window: Int
    var min_count: Int

    def __init__(
        out self,
        n_vars: Int,
        n_obs: Int,
        window: Int,
        min_count: Int,
    ):
        self.n_vars = n_vars
        self.n_obs = n_obs
        self.window = window
        self.min_count = min_count

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        var source = input.read_span()
        var destination = output.write_span()
        var leading_rows = min(self.window, max(self.min_count, 1))

        for i in range(self.n_vars):
            var shift_i: Float64
            if leading_rows == self.n_obs:
                shift_i = _first_observation_shift[Self.dtype](
                    source, self.n_obs, self.n_vars, i
                )
            else:
                shift_i = _leading_rows_shift[Self.dtype](
                    source,
                    self.n_obs,
                    self.n_vars,
                    i,
                    leading_rows,
                )
            for j in range(i, self.n_vars):
                var shift_j: Float64
                if leading_rows == self.n_obs:
                    shift_j = _first_observation_shift[Self.dtype](
                        source, self.n_obs, self.n_vars, j
                    )
                else:
                    shift_j = _leading_rows_shift[Self.dtype](
                        source,
                        self.n_obs,
                        self.n_vars,
                        j,
                        leading_rows,
                    )
                _move_corr_pair[Self.dtype](
                    source,
                    destination,
                    self.n_obs,
                    self.n_vars,
                    i,
                    j,
                    self.window,
                    self.min_count,
                    shift_i,
                    shift_j,
                )
                if leading_rows == self.n_obs:
                    _move_corr_pair_full_final[Self.dtype](
                        source,
                        destination,
                        self.n_obs,
                        self.n_vars,
                        i,
                        j,
                        self.min_count,
                        shift_i,
                        shift_j,
                    )
