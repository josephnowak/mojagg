"""Outer-slice execution for the combined-signature ``guvectorize`` driver.

The executor owns the hot outer loop. It computes one address per operand from
the broadcast outer coordinate, copies only non-contiguous read cores into a
64-byte-aligned worker-local scratch block, rebinds descriptors to the active
core length, and calls the operation. Parallelism is over outer slices; one
large core is never split by this generic driver.

Each worker receives an independent operation copy, descriptor tuple, and
scratch region. This makes mutable operation state safe for kernels such as
quantile that sort or otherwise mutate worker-private storage.
"""

from max.algorithm import parallelize
from std.memory import alloc, dealloc
from std.memory.alloc import Layout as AllocLayout
from std.sys import size_of

from mojagg.core.dispatch import DispatchPolicy
from mojagg.drivers.guvectorize_layout import OperandPlan
from mojagg.drivers.gutensor import AnyGUTensor
from mojagg.drivers.guvectorize_plan import GUVectorizePlan
from mojagg.drivers.guvectorize_spec import GUFuncKernel


@always_inline
def align_up_64(value: Int) -> Int:
    return ((value + 63) // 64) * 64


@always_inline
def max_input_core_length[
    *Args: AnyGUTensor,
](plan: GUVectorizePlan[len(Args)],) -> Int:
    var result = 0
    comptime for i in range(len(Args)):
        if not Args[i].is_output:
            result = max(result, plan.operands[i].core_length)
    return result


@always_inline
def operand_requires_scratch[
    index: Int,
    *Args: AnyGUTensor,
](plan: GUVectorizePlan[len(Args)],) -> Bool:
    comptime if Args[index].is_output:
        return False
    else:
        return (
            not plan.operands[index].core_contiguous
            and plan.operands[index].core_length > 0
        )


@always_inline
def outer_offset(plan: OperandPlan, flat_outer: Int) -> Int:
    """Convert a flat common outer index to an operand element offset."""
    var remaining = flat_outer
    var offset = 0
    for axis in range(plan.outer_rank - 1, -1, -1):
        var coordinate = remaining % plan.outer_shape[axis]
        remaining //= plan.outer_shape[axis]
        offset += coordinate * plan.outer_stride[axis]
    return offset


def execute_range[
    Operation: GUFuncKernel,
    *Args: AnyGUTensor,
](
    mut operation: Operation,
    tensors: Tuple[*Args],
    plan: GUVectorizePlan[len(Args)],
    start: Int,
    end: Int,
    worker_id: Int,
    scratch_base: Int,
    scratch_stride: Int,
    scratch_offsets: InlineArray[Int, len(Args)],
):
    """Run a contiguous outer-index range with one worker's state."""
    # One descriptor copy per worker. The loop only rebinds its address and
    # active length for each outer slice.
    var local_tensors = tensors.copy()
    for flat_outer in range(start, end):
        comptime for i in range(len(Args)):
            var offset = outer_offset(plan.operands[i], flat_outer)
            var address = (
                tensors[i].data_address()
                + offset * size_of[Scalar[Args[i].dtype]]()
            )
            if operand_requires_scratch[i, *Args](plan):
                var destination = (
                    scratch_base
                    + worker_id * scratch_stride
                    + scratch_offsets[i]
                )
                tensors[i].copy_core(plan.operands[i], offset, destination)
                address = destination
            local_tensors[i].bind_address(address, plan.operands[i].core_length)

        operation(rebind[Operation.Signature](local_tensors))


def execute_serial_or_parallel[
    Operation: GUFuncKernel,
    *Args: AnyGUTensor,
](
    operation: Operation,
    tensors: Tuple[*Args],
    plan: GUVectorizePlan[len(Args)],
    tasks: Int,
    scratch_base: Int,
    scratch_stride: Int,
    scratch_offsets: InlineArray[Int, len(Args)],
):
    """Choose serial execution or copy state into parallel workers."""
    if tasks <= 1:
        var local_operation = operation.copy()
        execute_range[Operation, *Args](
            local_operation,
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
        imm operation,
        imm tensors,
        imm plan,
        imm chunk,
        imm scratch_base,
        imm scratch_stride,
        imm scratch_offsets,
    }:
        var local_operation = operation.copy()
        var start = index * chunk
        var stop = min(start + chunk, plan.outer_count)
        execute_range[Operation, *Args](
            local_operation,
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


def guvectorize[
    Operation: GUFuncKernel,
    *Args: AnyGUTensor,
](
    operation: Operation,
    tensors: Tuple[*Args],
    plan: GUVectorizePlan[len(Args)],
    policy: DispatchPolicy,
) raises:
    """Execute an already planned combined-signature operation.

    The signature carries the current input and output addresses. The plan
    carries only shape and stride metadata, so output materialization can bind
    caller-owned NumPy arrays without patching a second plan object.
    """

    comptime assert (
        Operation.Signature == Tuple[*Args]
    ), "tensor tuple does not match the kernel signature"

    if plan.outer_count == 0:
        return

    var scratch_offsets = InlineArray[Int, len(Args)](fill=0)
    var inner_length = max_input_core_length[*Args](plan)
    var workers = policy.effective_workers(plan.outer_count, inner_length)
    var scratch_stride = 0
    comptime for i in range(len(Args)):
        if operand_requires_scratch[i, *Args](plan):
            scratch_offsets[i] = align_up_64(scratch_stride)
            scratch_stride = (
                scratch_offsets[i]
                + plan.operands[i].core_length
                * size_of[Scalar[Args[i].dtype]]()
            )

    if scratch_stride > 0:
        scratch_stride = align_up_64(scratch_stride)
        var scratch_count = max(workers * scratch_stride, 1)
        var scratch = alloc(AllocLayout[UInt8](count=scratch_count))
        execute_serial_or_parallel[Operation, *Args](
            operation,
            tensors,
            plan,
            workers,
            Int(scratch.unsafe_ptr()),
            scratch_stride,
            scratch_offsets,
        )
        dealloc(scratch^)
        return

    execute_serial_or_parallel[Operation, *Args](
        operation,
        tensors,
        plan,
        workers,
        0,
        0,
        scratch_offsets,
    )
