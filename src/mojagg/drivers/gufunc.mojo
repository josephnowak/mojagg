"""Native-tuple universal function driver.

Every operation supplies one concrete heterogeneous Mojo ``Tuple`` through
``GUFuncOperation.Tensors`` and implements only ``apply(tensors)``.  Tuple
elements are ``TensorArg[dtype, writable]`` values.  The driver specializes
its traversal over the tuple's compile-time type pack, so it can handle mixed
dtypes without boxing or runtime dtype dispatch.

For each outer slice the driver:

1. computes an affine address for every tuple element;
2. copies only non-writable, non-contiguous read cores into one reusable
   scratch region per worker; and
3. binds all prepared descriptors before calling the operation.

Writable elements always point directly into caller-owned output storage.  A
worker receives a copy of the operation, allowing operation-owned temporary
workspace to remain private to that worker.
"""

from max.algorithm import parallelize
from std.collections import InlineArray
from std.memory import alloc, dealloc
from std.memory.alloc import Layout as AllocLayout

from mojagg.core.tensor_view import (
    DimArray,
    MAX_RANK,
    OperandPlan,
    TensorArgProtocol,
)


trait GUFuncOperation(Copyable & Deinitable):
    """Operation contract consumed by ``GUFunc``.

    Each operation declares one fixed-arity native tuple, for example::

        comptime Tensors = Tuple[
            TensorArg[DType.float64, False],
            TensorArg[DType.float32, True],
        ]

    The first element is conventionally a read tensor and writable elements
    are direct output tensors.  The tuple is part of the operation's
    compilation-time contract: dtypes, mutability, and arity are fixed for
    every specialization.  ``GUFunc.execute`` accepts the corresponding
    native tuple pack (``Tuple[*Args]`` with ``TensorArgProtocol`` elements)
    and rebinds it to ``Op.Tensors`` only at the final ``apply`` call.  The
    rebind has no runtime cost and prevents boxing or dynamic dtype dispatch.

    Mojo traits cannot currently parameterize a variadic tuple pack directly,
    so the associated ``Tensors`` type is the portable way to express this
    contract while preserving a fully static operation interface.
    """

    comptime Tensors: AnyType

    def apply(mut self, tensors: Self.Tensors):
        ...


@fieldwise_init
struct DispatchPolicy(Copyable):
    """Value-only CPU scheduling policy."""

    var workers: Int
    var parallel_threshold: Int

    @always_inline
    def effective_workers(self, outer_count: Int, inner_length: Int) -> Int:
        var requested = self.workers if self.workers > 0 else 16
        if outer_count <= 1 or requested <= 1:
            return 1
        # The configured threshold is inclusive and applies to the complete
        # outer domain, not just one slice.
        var total_length = outer_count * inner_length
        if total_length < max(self.parallel_threshold, 0):
            return 1
        return min(requested, outer_count)


@fieldwise_init
struct GUFuncPlan[NUM_TENSORS: Int](Copyable):
    """Validated runtime metadata shared by all workers."""

    var outer_rank: Int
    var outer_count: Int
    var outer_shape: DimArray
    var tensors: InlineArray[OperandPlan, Self.NUM_TENSORS]

    @staticmethod
    def _build_operand[
        index: Int,
        *Args: TensorArgProtocol,
    ](
        tensors: Tuple[*Args],
        input_axes: DimArray,
        input_core_rank: Int,
        output_axes: DimArray,
        output_core_rank: Int,
        expected_outer_rank: Int,
        expected_outer_shape: DimArray,
    ) raises -> OperandPlan:
        var operand = tensors[index].copy()
        var rank = operand.rank()
        if rank < 0 or rank > MAX_RANK:
            raise Error("tensor rank exceeds gufunc capacity")

        var axes = input_axes.copy()
        var core_rank = input_core_rank
        if operand.is_writable():
            axes = output_axes.copy()
            core_rank = output_core_rank
        if core_rank < 0 or core_rank > rank:
            raise Error("invalid tensor core rank")

        var plan = OperandPlan.empty()
        plan.rank = rank
        plan.core_rank = core_rank
        plan.outer_rank = rank - core_rank
        plan.core_length = 1
        plan.outer_count = 1
        plan.base_address = operand.base_address()
        var selected = InlineArray[Bool, MAX_RANK](fill=False)
        var has_zero_extent = False

        for d in range(rank):
            plan.shape[d] = operand.shape_at(d)
            plan.stride[d] = operand.stride_at(d)
            if plan.shape[d] < 0:
                raise Error("negative tensor dimension")
            if plan.shape[d] == 0:
                has_zero_extent = True

        for d in range(core_rank):
            var axis = axes[d]
            if axis < 0 or axis >= rank or selected[axis]:
                raise Error("core axes must be normalized and unique")
            selected[axis] = True
            plan.core_shape[d] = plan.shape[axis]
            plan.core_stride[d] = plan.stride[axis]
            plan.core_length *= plan.core_shape[d]

        var outer_index = 0
        for d in range(rank):
            if not selected[d]:
                plan.outer_shape[outer_index] = plan.shape[d]
                plan.outer_stride[outer_index] = plan.stride[d]
                plan.outer_count *= plan.outer_shape[outer_index]
                outer_index += 1

        if expected_outer_rank >= 0:
            if plan.outer_rank != expected_outer_rank:
                raise Error("tensor outer ranks do not match")
            for d in range(expected_outer_rank):
                if plan.outer_shape[d] != expected_outer_shape[d]:
                    raise Error("tensor outer shapes do not match")

        # A zero-length core receives an empty span and needs no copy.
        plan.core_contiguous = True
        if plan.core_rank > 0 and plan.core_length > 0:
            var expected_stride = 1
            for d in range(plan.core_rank - 1, -1, -1):
                if (
                    plan.core_shape[d] > 1
                    and plan.core_stride[d] != expected_stride
                ):
                    plan.core_contiguous = False
                expected_stride *= plan.core_shape[d]

        if operand.is_writable():
            # Empty outputs may have arbitrary NumPy strides because no store
            # occurs.  Non-empty writable views must be direct C-contiguous
            # storage, including the selected output core.
            if not has_zero_extent:
                var expected_stride = 1
                for d in range(rank - 1, -1, -1):
                    if plan.shape[d] > 1 and plan.stride[d] != expected_stride:
                        raise Error("writable tensor must be C-contiguous")
                    expected_stride *= plan.shape[d]
            if plan.core_rank > 0 and plan.core_length > 0:
                var core_expected = 1
                for d in range(plan.core_rank - 1, -1, -1):
                    if (
                        plan.core_shape[d] > 1
                        and plan.core_stride[d] != core_expected
                    ):
                        raise Error("writable core must be contiguous")
                    core_expected *= plan.core_shape[d]

        return plan^

    @staticmethod
    def build[
        *Args: TensorArgProtocol
    ](
        tensors: Tuple[*Args],
        input_axes: DimArray,
        input_core_rank: Int,
        output_axes: DimArray,
        output_core_rank: Int,
    ) raises -> Self:
        """Construct and validate the complete execution plan."""

        comptime assert len(Args) == Self.NUM_TENSORS
        comptime assert Self.NUM_TENSORS > 1

        var first = tensors[0].copy()
        if first.is_writable():
            raise Error("the first GUFunc tensor must be a read input")

        var plans = InlineArray[OperandPlan, Self.NUM_TENSORS](
            fill=OperandPlan.empty()
        )
        var reference = Self._build_operand[0, *Args](
            tensors,
            input_axes,
            input_core_rank,
            output_axes,
            output_core_rank,
            -1,
            DimArray(fill=1),
        )
        plans[0] = reference.copy()

        comptime for i in range(1, Self.NUM_TENSORS):
            plans[i] = Self._build_operand[i, *Args](
                tensors,
                input_axes,
                input_core_rank,
                output_axes,
                output_core_rank,
                reference.outer_rank,
                reference.outer_shape.copy(),
            )

        return Self(
            reference.outer_rank,
            reference.outer_count,
            reference.outer_shape.copy(),
            plans^,
        )


@always_inline
def outer_offset(plan: OperandPlan, flat_outer: Int) -> Int:
    """Decode one row-major outer index into an element offset."""

    var remaining = flat_outer
    var offset = 0
    for d in range(plan.outer_rank - 1, -1, -1):
        var coordinate = remaining % plan.outer_shape[d]
        remaining //= plan.outer_shape[d]
        offset += coordinate * plan.outer_stride[d]
    return offset


def execute_range[
    Op: GUFuncOperation,
    *Args: TensorArgProtocol,
](
    mut op: Op,
    tensors: Tuple[*Args],
    plan: GUFuncPlan[len(Args)],
    start: Int,
    end: Int,
    worker_id: Int,
    scratch_base: Int,
    scratch_stride: Int,
    scratch_offsets: InlineArray[Int, len(Args)],
):
    """Prepare and execute one worker's range of outer slices."""

    var local_tensors = tensors.copy()
    for flat_outer in range(start, end):
        comptime for i in range(len(Args)):
            var operand = local_tensors[i].copy()
            var offset = outer_offset(plan.tensors[i], flat_outer)
            var address = (
                plan.tensors[i].base_address + offset * operand.item_size()
            )
            if (
                not operand.is_writable()
                and not plan.tensors[i].core_contiguous
                and plan.tensors[i].core_length > 0
            ):
                var destination = (
                    scratch_base
                    + worker_id * scratch_stride
                    + scratch_offsets[i]
                )
                operand.copy_core(plan.tensors[i], offset, destination)
                address = destination
            operand.set_slice(address, plan.tensors[i].core_length)
            local_tensors[i] = operand^

        # ``rebind`` is representation-only: both types are the same native
        # Tuple and the operation's associated tuple is compile-time exact.
        op.apply(rebind[Op.Tensors](local_tensors))


def execute_serial_or_parallel[
    Op: GUFuncOperation,
    *Args: TensorArgProtocol,
](
    op: Op,
    tensors: Tuple[*Args],
    plan: GUFuncPlan[len(Args)],
    policy: DispatchPolicy,
    scratch_base: Int,
    scratch_stride: Int,
    scratch_offsets: InlineArray[Int, len(Args)],
):
    """Run serially or create one independent operation per worker."""

    var inner_length = 0
    comptime for i in range(len(Args)):
        var operand = tensors[i].copy()
        if not operand.is_writable():
            inner_length = max(inner_length, plan.tensors[i].core_length)
    var tasks = policy.effective_workers(plan.outer_count, inner_length)

    if tasks <= 1:
        var worker_op = op.copy()
        execute_range[Op, *Args](
            worker_op,
            tensors,
            plan,
            0,
            plan.outer_count,
            0,
            scratch_base,
            scratch_stride,
            scratch_offsets,
        )
        return

    var chunk = (plan.outer_count + tasks - 1) // tasks

    def worker(
        index: Int,
    ) {
        imm op,
        imm tensors,
        imm plan,
        imm chunk,
        imm scratch_base,
        imm scratch_stride,
        imm scratch_offsets,
    }:
        var worker_op = op.copy()
        var start = index * chunk
        var stop = min(start + chunk, plan.outer_count)
        execute_range[Op, *Args](
            worker_op,
            tensors,
            plan,
            start,
            stop,
            index,
            scratch_base,
            scratch_stride,
            scratch_offsets,
        )

    parallelize(worker, tasks)


struct GUFunc[Operation: GUFuncOperation](Copyable):
    """Apply one operation over every outer slice of a native tensor tuple."""

    var operation: Self.Operation

    def __init__(out self, operation: Self.Operation):
        self.operation = operation.copy()

    def execute[
        *Args: TensorArgProtocol
    ](
        self,
        tensors: Tuple[*Args],
        input_axes: DimArray,
        input_core_rank: Int,
        output_axes: DimArray,
        output_core_rank: Int,
        policy: DispatchPolicy,
    ) raises:
        """Build plans, materialize read cores, and call ``apply``."""

        var plan = GUFuncPlan[len(Args)].build[*Args](
            tensors,
            input_axes,
            input_core_rank,
            output_axes,
            output_core_rank,
        )
        if plan.outer_count == 0:
            return

        var inner_length = 0
        comptime for i in range(len(Args)):
            var operand = tensors[i].copy()
            if not operand.is_writable():
                inner_length = max(inner_length, plan.tensors[i].core_length)
        var tasks = policy.effective_workers(plan.outer_count, inner_length)

        var scratch_offsets = InlineArray[Int, len(Args)](fill=0)
        var scratch_stride = 0
        comptime for i in range(len(Args)):
            var operand = tensors[i].copy()
            if (
                not operand.is_writable()
                and not plan.tensors[i].core_contiguous
                and plan.tensors[i].core_length > 0
            ):
                scratch_offsets[i] = ((scratch_stride + 63) // 64) * 64
                scratch_stride = (
                    scratch_offsets[i]
                    + plan.tensors[i].core_length * operand.item_size()
                )

        if scratch_stride > 0:
            scratch_stride = ((scratch_stride + 63) // 64) * 64
            var scratch_count = max(tasks * scratch_stride, 1)
            var scratch = alloc(AllocLayout[UInt8](count=scratch_count))
            execute_serial_or_parallel[Self.Operation, *Args](
                self.operation,
                tensors,
                plan,
                policy,
                Int(scratch.unsafe_ptr()),
                scratch_stride,
                scratch_offsets,
            )
            dealloc(scratch^)
            return

        execute_serial_or_parallel[Self.Operation, *Args](
            self.operation,
            tensors,
            plan,
            policy,
            0,
            0,
            scratch_offsets,
        )
