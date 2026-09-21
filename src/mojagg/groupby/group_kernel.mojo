"""Shared grouped-kernel marker and output initialization constants."""

from mojagg.drivers.guvectorize import GUFuncKernel


comptime GROUP_INIT_ZERO = 0
comptime GROUP_INIT_ONE = 1
comptime GROUP_INIT_NAN_OR_ZERO = 2
comptime GROUP_INIT_NAN_OR_POS_INF = 3
comptime GROUP_INIT_NAN_OR_NEG_INF = 4
comptime GROUP_INIT_NAN_OR_NEG_ONE = 5


trait GroupKernel(GUFuncKernel):
    """Marker for grouped kernels.

    Output dtypes and output counts belong to the concrete ``Signature``
    declared by each operation. Initialization identities are selected by the
    Python binding at the call site, so they are not part of the kernel type.
    """
