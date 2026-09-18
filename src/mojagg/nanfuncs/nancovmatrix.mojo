"""NaN-aware covariance matrix kernel."""

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
    _accumulate_cov_pair_simd,
    _nanmatrix_2d_contiguous,
)


struct NanCovOp[dtype: DType](
    GUFuncKernel, ImplicitlyCopyable, MatrixPairwiseOp
):
    comptime out_dtype = Self.dtype
    comptime Signature = Tuple[
        GUTensor[
            Self.out_dtype,
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
        shift_i: Scalar[Self.out_dtype],
        shift_j: Scalar[Self.out_dtype],
        is_diag: Bool,
    ) -> PairwiseAcc[Self.out_dtype]:
        return _accumulate_cov_pair_simd[Self.out_dtype](
            p_i, p_j, n_obs, shift_i, shift_j, is_diag
        )

    @always_inline
    @staticmethod
    def finalize(
        acc: PairwiseAcc[Self.out_dtype], is_diag: Bool
    ) -> Scalar[Self.out_dtype]:
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
    def __call__(mut self, tensors: Self.Signature):
        var input, output = tensors
        _nanmatrix_2d_contiguous[Self](
            input.read_span(),
            output.write_span(),
            self.n_vars,
            self.n_obs,
            self.inner_workers,
        )
