"""Public façade for the combined-signature ``guvectorize`` driver.

The implementation is split across signature types, tensor descriptors,
planning, and execution modules. Existing callers can continue importing the
public driver surface from this module. The complete behavioral contract is
documented in ``docs/guvectorize.md``: a kernel receives one prepared core
tuple, while this package handles symbolic dimensions, outer broadcasting,
strided read scratch, writable-core validation, and worker scheduling.
"""

from mojagg.core.dispatch import DispatchPolicy
from mojagg.drivers.guvectorize_layout import (
    DimArray,
    MAX_RANK,
    OperandPlan,
)
from mojagg.drivers.gutensor import (
    AnyGUTensor,
    GUTensor,
    contiguous_strides,
    tensor_element_count,
)
from mojagg.drivers.guvectorize_execute import (
    align_up_64,
    execute_range,
    execute_serial_or_parallel,
    guvectorize,
    max_input_core_length,
    operand_requires_scratch,
    outer_offset,
)
from mojagg.drivers.guvectorize_plan import (
    BroadcastDomain,
    CoreBindings,
    GUVectorizePlan,
    bind_core_dimensions,
    build_signature,
    build_signature_with_bindings,
    resolve_core_dimensions,
    validate_core_symbol,
)
from mojagg.drivers.guvectorize_spec import (
    AxisSpec,
    BoolArray,
    CoreDim,
    CoreSpec,
    CoreSpecProtocol,
    Dim,
    FixedDim,
    GUFuncKernel,
)
