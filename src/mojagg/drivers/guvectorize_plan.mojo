"""Signature planning and outer broadcast resolution for ``guvectorize``.

This module is the boundary between a complete tensor descriptor and one
prepared core invocation. It first binds the compile-time ``CoreSpec`` symbols
from input shapes, then removes selected physical core axes to form outer
shapes. Input outer shapes are right-aligned and broadcast like NumPy. Outputs
must have the resolved outer rank and shape exactly because duplicating a
writable destination would make ownership ambiguous.

The planner does not copy ordinary inputs. It records element strides and
marks a read core for scratch only when its selected axes are not contiguous in
the requested logical order. Writable cores are rejected at plan time when
they cannot be consumed as a direct contiguous span.
"""

from std.collections import InlineArray

from mojagg.drivers.guvectorize_layout import (
    DimArray,
    MAX_RANK,
    OperandPlan,
)
from mojagg.drivers.gutensor import (
    AnyGUTensor,
    GUTensor,
    tensor_element_count,
)
from mojagg.drivers.guvectorize_spec import (
    AxisSpec,
    CoreSpecProtocol,
    GUFuncKernel,
)


@fieldwise_init
struct BroadcastDomain(Copyable):
    """The common right-aligned outer domain of the read inputs."""

    var rank: Int
    var count: Int
    var shape: DimArray


@fieldwise_init
struct GUVectorizePlan[NUM_TENSORS: Int](Copyable):
    """One complete layout plan shared by all outer-slice workers."""

    var outer_rank: Int
    var outer_count: Int
    var outer_shape: DimArray
    var operands: InlineArray[OperandPlan, Self.NUM_TENSORS]

    @staticmethod
    def resolve_broadcast[
        *Args: AnyGUTensor,
    ](
        plans: InlineArray[OperandPlan, Self.NUM_TENSORS],
    ) raises -> BroadcastDomain:
        """Resolve only input outer shapes using right-aligned broadcasting.

        Outputs are deliberately excluded here. They are checked against the
        resulting domain in ``build`` so a writable operand can never silently
        broadcast into multiple destinations.
        """
        comptime assert len(Args) == Self.NUM_TENSORS

        var common_rank = 0
        comptime for i in range(Self.NUM_TENSORS):
            if not Args[i].is_output:
                common_rank = max(common_rank, plans[i].outer_rank)
        var common_shape = DimArray(fill=1)

        comptime for i in range(Self.NUM_TENSORS):
            if not Args[i].is_output:
                var local_rank = plans[i].outer_rank
                for local_axis in range(local_rank):
                    var common_axis = common_rank - local_rank + local_axis
                    var candidate = plans[i].outer_shape[local_axis]
                    var current = common_shape[common_axis]
                    if current == 1:
                        common_shape[common_axis] = candidate
                    elif candidate != 1 and candidate != current:
                        raise Error("input outer shapes cannot broadcast")

        var common_count = 1
        for axis in range(common_rank):
            common_count *= common_shape[axis]

        return BroadcastDomain(common_rank, common_count, common_shape^)

    @staticmethod
    def build_operand[
        index: Int,
        *Args: AnyGUTensor,
    ](
        tensors: Tuple[*Args],
        input_axes: AxisSpec,
        output_axes: AxisSpec,
    ) raises -> OperandPlan:
        """Split one tensor into selected core and remaining outer metadata."""
        # Mojo's variadic trait pack exposes only trait members.  Rebind the
        # concrete element so the planner can read the public shape/stride
        # fields directly instead of adding shape_at/stride_at accessors.
        var tensor = rebind[
            GUTensor[
                Args[index].dtype,
                Args[index].is_output,
                Args[index].core_spec,
            ]
        ](tensors[index])
        var rank = tensor.ndim
        if rank < 0 or rank > MAX_RANK:
            raise Error("tensor rank exceeds guvectorize capacity")

        var core_rank = input_axes.count
        var axes = input_axes.values.copy()
        if Args[index].is_output:
            core_rank = output_axes.count
            axes = output_axes.values.copy()
        if core_rank < 0 or core_rank > rank:
            raise Error("invalid core rank")
        if not Args[index].is_output and not tensor.bound:
            raise Error("tensor address is unbound")

        var plan = OperandPlan.empty()
        plan.rank = rank
        plan.core_rank = core_rank
        plan.outer_rank = rank - core_rank
        plan.core_length = 1
        plan.outer_count = 1
        var selected = InlineArray[Bool, MAX_RANK](fill=False)
        for axis in range(rank):
            plan.shape[axis] = tensor.shape[axis]
            plan.stride[axis] = tensor.stride[axis]
            if plan.shape[axis] < 0:
                raise Error("negative tensor dimension")

        for core_axis in range(core_rank):
            var axis = axes[core_axis]
            if axis < 0 or axis >= rank or selected[axis]:
                raise Error("core axes must be normalized and unique")
            selected[axis] = True
            plan.core_shape[core_axis] = plan.shape[axis]
            plan.core_stride[core_axis] = plan.stride[axis]
            plan.core_length *= plan.core_shape[core_axis]

        var outer_axis = 0
        for axis in range(rank):
            if not selected[axis]:
                plan.outer_shape[outer_axis] = plan.shape[axis]
                plan.outer_stride[outer_axis] = plan.stride[axis]
                plan.outer_count *= plan.outer_shape[outer_axis]
                outer_axis += 1

        plan.core_contiguous = True
        if plan.core_rank > 0 and plan.core_length > 0:
            var expected_stride = 1
            for axis in range(plan.core_rank - 1, -1, -1):
                if (
                    plan.core_shape[axis] > 1
                    and plan.core_stride[axis] != expected_stride
                ):
                    plan.core_contiguous = False
                expected_stride *= plan.core_shape[axis]

        if Args[index].is_output and plan.core_length > 0:
            var expected_stride = 1
            for axis in range(plan.core_rank - 1, -1, -1):
                if (
                    plan.core_shape[axis] > 1
                    and plan.core_stride[axis] != expected_stride
                ):
                    raise Error("writable core must be contiguous")
                expected_stride *= plan.core_shape[axis]

        return plan^

    @staticmethod
    def build[
        *Args: AnyGUTensor,
    ](
        tensors: Tuple[*Args],
        input_axes: AxisSpec,
        output_axes: AxisSpec,
    ) raises -> Self:
        """Build the executable plan and validate every writable layout."""
        comptime assert len(Args) == Self.NUM_TENSORS
        comptime assert Self.NUM_TENSORS > 1

        if Args[0].is_output:
            raise Error("the first tensor must be a read input")

        var plans = InlineArray[OperandPlan, Self.NUM_TENSORS](
            fill=OperandPlan.empty()
        )
        comptime for i in range(Self.NUM_TENSORS):
            plans[i] = Self.build_operand[i, *Args](
                tensors,
                input_axes,
                output_axes,
            )

        # Inputs define the broadcast domain.  Outputs must match it exactly.
        var broadcast = Self.resolve_broadcast[*Args](plans)
        var common_rank = broadcast.rank
        var common_count = broadcast.count
        var common_shape = broadcast.shape.copy()

        comptime for i in range(Self.NUM_TENSORS):
            var local_rank = plans[i].outer_rank
            if Args[i].is_output and local_rank != common_rank:
                raise Error("output outer rank does not match broadcast rank")

            var local_shape = plans[i].outer_shape.copy()
            var local_stride = plans[i].outer_stride.copy()
            plans[i].outer_rank = common_rank
            plans[i].outer_count = common_count
            plans[i].outer_shape = common_shape.copy()
            plans[i].outer_stride = DimArray(fill=0)

            for common_axis in range(common_rank):
                var local_axis = common_axis - (common_rank - local_rank)
                if local_axis < 0:
                    if Args[i].is_output and common_shape[common_axis] != 1:
                        raise Error("output is missing a broadcast dimension")
                    continue

                var candidate = local_shape[local_axis]
                if Args[i].is_output:
                    if candidate != common_shape[common_axis]:
                        raise Error(
                            "output shape does not match broadcast shape"
                        )
                    plans[i].outer_stride[common_axis] = local_stride[
                        local_axis
                    ]
                elif candidate != 1:
                    plans[i].outer_stride[common_axis] = local_stride[
                        local_axis
                    ]

        return Self(
            common_rank,
            common_count,
            common_shape^,
            plans^,
        )


@fieldwise_init
struct CoreBindings(Copyable):
    """Runtime values resolved for symbolic core dimensions."""

    var values: DimArray
    var bound: InlineArray[Bool, MAX_RANK]

    @staticmethod
    def empty() -> Self:
        return Self(DimArray(fill=0), InlineArray[Bool, MAX_RANK](fill=False))

    @always_inline
    def bind(mut self, symbol: Int, extent: Int) raises:
        """Seed or validate one runtime value for a symbolic dimension."""

        validate_core_symbol(symbol)
        if extent < 0:
            raise Error("core dimension extent cannot be negative")
        if not self.bound[symbol]:
            self.bound[symbol] = True
            self.values[symbol] = extent
        elif self.values[symbol] != extent:
            raise Error("named core dimensions must have equal extents")


@always_inline
def validate_core_symbol(symbol: Int) raises:
    if symbol < 0 or symbol >= MAX_RANK:
        raise Error("core dimension symbol exceeds capacity")


def bind_core_dimensions[
    Spec: CoreSpecProtocol,
](mut bindings: CoreBindings, logical_core_shape: DimArray,) raises:
    """Bind or validate each symbolic/fixed dimension in a logical core."""
    var core_values = Spec.values()
    var core_fixed = Spec.fixed()
    comptime for core_axis in range(Spec.rank):
        if core_fixed[core_axis]:
            if logical_core_shape[core_axis] != core_values[core_axis]:
                raise Error("input extent does not match fixed core dimension")
        else:
            var symbol = core_values[core_axis]
            var extent = logical_core_shape[core_axis]
            bindings.bind(symbol, extent)


def resolve_core_dimensions[
    Spec: CoreSpecProtocol,
](bindings: CoreBindings,) raises -> DimArray:
    """Materialize an output core shape from fixed and bound dimensions."""
    var result = DimArray(fill=1)
    var core_values = Spec.values()
    var core_fixed = Spec.fixed()
    comptime for core_axis in range(Spec.rank):
        if core_fixed[core_axis]:
            result[core_axis] = core_values[core_axis]
        else:
            var symbol = core_values[core_axis]
            validate_core_symbol(symbol)
            if not bindings.bound[symbol]:
                raise Error("output named dimension has no input extent")
            result[core_axis] = bindings.values[symbol]
    return result^


def _build_signature_plan[
    Operation: GUFuncKernel,
    *Args: AnyGUTensor,
](
    mut tensors: Tuple[*Args],
    input_axes: AxisSpec,
    output_axes: AxisSpec,
    initial_bindings: CoreBindings,
) raises -> GUVectorizePlan[len(Args)]:
    """Resolve the signature in place and return its executable layout plan.

    A rank-one ``CoreSpec[Dim[id]]`` can consume several selected input axes;
    in that reduction form the selected extents are multiplied and bound as
    one logical dimension. Multi-dimensional core specs instead bind each
    selected axis in order.
    """

    comptime assert (
        Operation.Signature == Tuple[*Args]
    ), "tensor tuple does not match the kernel signature"
    comptime assert len(Args) > 1, "a gufunc needs an input and an output"

    var plans = InlineArray[OperandPlan, len(Args)](fill=OperandPlan.empty())
    var empty_output_axes = AxisSpec.empty()

    # Build plans for bound inputs first. Outputs are unbound templates and
    # receive their metadata after the common broadcast domain is resolved.
    comptime for i in range(len(Args)):
        comptime if not Args[i].is_output:
            plans[i] = GUVectorizePlan[len(Args)].build_operand[i, *Args](
                tensors,
                input_axes,
                empty_output_axes,
            )

    var bindings = initial_bindings.copy()

    # Reusing a symbolic ID enforces equality across input operands.
    comptime for i in range(len(Args)):
        comptime if not Args[i].is_output:
            var logical_core_shape = DimArray(fill=1)
            if Args[i].core_spec.rank == input_axes.count:
                for core_axis in range(input_axes.count):
                    logical_core_shape[core_axis] = plans[i].core_shape[
                        core_axis
                    ]
            elif Args[i].core_spec.rank == 1 and input_axes.count > 0:
                # A one-dimensional ``(n)`` core is the flexible flattened
                # form used by reductions over one or several selected axes.
                logical_core_shape[0] = plans[i].core_length
            else:
                raise Error("input axis count does not match its core spec")
            bind_core_dimensions[Args[i].core_spec](
                bindings,
                logical_core_shape,
            )

    # Inputs define the right-aligned broadcast domain.
    var broadcast = GUVectorizePlan[len(Args)].resolve_broadcast[*Args](plans)
    var common_rank = broadcast.rank
    var common_shape = broadcast.shape.copy()

    # Materialize every output descriptor from the broadcast domain and its
    # declared core dimensions.
    comptime for i in range(len(Args)):
        comptime if Args[i].is_output:
            var output_was_bound = tensors[i].is_bound()
            var output_address = tensors[i].data_address()
            var logical_core_rank = Args[i].core_spec.rank
            var physical_core_rank = logical_core_rank
            var flattened_output = False
            if output_axes.count != logical_core_rank:
                # A rank-one transform can consume several physical axes as
                # one flattened logical span.  Preserve those physical core
                # extents on the output while retaining the rank-one kernel
                # contract.  Fill uses this form.
                if (
                    logical_core_rank == 1
                    and input_axes.count > 1
                    and output_axes.count == input_axes.count
                ):
                    physical_core_rank = output_axes.count
                    flattened_output = True
                else:
                    raise Error(
                        "output axis count does not match its core spec"
                    )
            var output_rank = common_rank + physical_core_rank
            if output_rank < 0 or output_rank > MAX_RANK:
                raise Error("output rank exceeds guvectorize capacity")

            var output_shape = DimArray(fill=1)
            var selected = InlineArray[Bool, MAX_RANK](fill=False)
            for core_axis in range(physical_core_rank):
                var axis = output_axes[core_axis]
                if axis < 0 or axis >= output_rank or selected[axis]:
                    raise Error(
                        "output core axes must be normalized and unique"
                    )
                selected[axis] = True

            var resolved_core_shape = DimArray(fill=1)
            if flattened_output:
                for core_axis in range(physical_core_rank):
                    resolved_core_shape[core_axis] = plans[0].core_shape[
                        core_axis
                    ]
            else:
                resolved_core_shape = resolve_core_dimensions[
                    Args[i].core_spec
                ](bindings)

            var core_index = 0
            var outer_index = 0
            for axis in range(output_rank):
                if selected[axis]:
                    output_shape[axis] = resolved_core_shape[core_index]
                    core_index += 1
                else:
                    output_shape[axis] = common_shape[outer_index]
                    outer_index += 1

            tensors[i].set_unbound_layout(output_shape, output_rank)
            if output_was_bound:
                # Low-level callers may supply an already allocated output.
                # The Python materialization path leaves this false and binds
                # the address only after the plan is complete.
                tensors[i].bind_address(
                    output_address,
                    tensor_element_count(output_shape, output_rank),
                )

    return GUVectorizePlan[len(Args)].build[*Args](
        tensors,
        input_axes,
        output_axes,
    )


def build_signature_plan[
    Operation: GUFuncKernel,
    *Args: AnyGUTensor,
](
    mut tensors: Tuple[*Args],
    input_axes: AxisSpec,
    output_axes: AxisSpec,
) raises -> GUVectorizePlan[len(Args)]:
    """Resolve a kernel signature in place and return its layout plan."""

    var bindings = CoreBindings.empty()
    return _build_signature_plan[Operation, *Args](
        tensors,
        input_axes,
        output_axes,
        bindings,
    )


def build_signature_plan_with_bindings[
    Operation: GUFuncKernel,
    *Args: AnyGUTensor,
](
    mut tensors: Tuple[*Args],
    input_axes: AxisSpec,
    output_axes: AxisSpec,
    initial_bindings: CoreBindings,
) raises -> GUVectorizePlan[len(Args)]:
    """Resolve a signature while seeding symbols from an external dimension.

    Quantile uses this form because the output core length comes from the
    separate quantile array rather than from an input tensor core.
    """

    return _build_signature_plan[Operation, *Args](
        tensors,
        input_axes,
        output_axes,
        initial_bindings,
    )
