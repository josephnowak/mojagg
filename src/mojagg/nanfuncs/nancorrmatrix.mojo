"""NaN-aware correlation matrix kernel."""

from std.math import sqrt

from mojagg.drivers.guvectorize import (
    CoreSpec,
    Dim,
    GUTensor,
    GUFuncKernel,
)
from mojagg.core.numeric import nan_or_zero
from mojagg.nanfuncs.nanmatrix import (
    MatrixPairwiseOp,
    PairwiseAcc,
    _accumulate_pair_simd,
    _nanmatrix_2d_contiguous,
)


struct NanCorrOp[dtype: DType](
    GUFuncKernel, ImplicitlyCopyable, MatrixPairwiseOp
):
    comptime value_dtype = Self.dtype
    comptime out_dtype = Self.dtype
    comptime Signature = Tuple[
        GUTensor[
            Self.value_dtype,
            False,
            CoreSpec[Dim[0], Dim[1]],
        ],
        GUTensor[
            Self.out_dtype,
            True,
            CoreSpec[Dim[0], Dim[0]],
        ],
    ]

    var n_vars: Int
    var n_obs: Int
    var inner_workers: Int

    def __init__(out self, n_vars: Int, n_obs: Int, inner_workers: Int):
        self.n_vars = n_vars
        self.n_obs = n_obs
        self.inner_workers = inner_workers

    @always_inline
    @staticmethod
    def accumulate(
        p_i: Pointer[mut=False, Scalar[Self.out_dtype], ImmUntrackedOrigin],
        p_j: Pointer[mut=False, Scalar[Self.out_dtype], ImmUntrackedOrigin],
        n_obs: Int,
        shift_i: Float64,
        shift_j: Float64,
        is_diag: Bool,
    ) -> PairwiseAcc:
        return _accumulate_pair_simd[Self.out_dtype](
            p_i, p_j, n_obs, shift_i, shift_j, is_diag
        )

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

        # Divide by each standard deviation separately.  Multiplying the two
        # variances first can overflow even when the correlation is finite.
        var correlation = covariance / sqrt(variance_i) / sqrt(variance_j)
        if correlation > 1.0:
            correlation = 1.0
        elif correlation < -1.0:
            correlation = -1.0
        return Scalar[Self.out_dtype](correlation)

    @always_inline
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        _nanmatrix_2d_contiguous[Self](
            input.read_span(),
            output.write_span(),
            self.n_vars,
            self.n_obs,
            self.inner_workers,
        )
