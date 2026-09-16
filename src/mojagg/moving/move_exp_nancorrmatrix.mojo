"""NaN-aware exponentially weighted moving correlation matrices."""

from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)
from mojagg.moving.moving_matrix_helpers import (
    _first_observation_shift,
    _move_exp_corr_pair,
)


struct MoveExpNanCorrMatrixKernel[dtype: DType](
    GUFuncKernel, ImplicitlyCopyable
):
    """SIMD-driver-compatible ``(m,n), (m,n) -> (m,n,n)`` correlation."""

    comptime Signature = Tuple[
        GUTensor[Self.dtype, False, CoreSpec[Dim[0], Dim[1]]],
        GUTensor[Self.dtype, False, CoreSpec[Dim[0], Dim[1]]],
        GUTensor[
            Self.dtype,
            True,
            CoreSpec[Dim[0], Dim[1], Dim[1]],
        ],
    ]

    var n_vars: Int
    var n_obs: Int
    var min_weight: Float64

    def __init__(
        out self,
        n_vars: Int,
        n_obs: Int,
        min_weight: Float64,
    ):
        self.n_vars = n_vars
        self.n_obs = n_obs
        self.min_weight = min_weight

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, alpha, output = tensors
        var source = input.read_span()
        var alphas = alpha.read_span()
        var destination = output.write_span()

        for i in range(self.n_vars):
            var shift_i = _first_observation_shift[Self.dtype](
                source,
                self.n_obs,
                self.n_vars,
                i,
            )
            for j in range(i, self.n_vars):
                var shift_j = _first_observation_shift[Self.dtype](
                    source,
                    self.n_obs,
                    self.n_vars,
                    j,
                )
                _move_exp_corr_pair[Self.dtype](
                    source,
                    alphas,
                    destination,
                    self.n_obs,
                    self.n_vars,
                    i,
                    j,
                    self.min_weight,
                    shift_i,
                    shift_j,
                )
