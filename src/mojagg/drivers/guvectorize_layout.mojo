"""Runtime metadata shared by the ``guvectorize`` planner and tensor views.

The driver intentionally uses inline arrays. A plan is built once per native
call and records both the original physical layout and the logical layout after
core axes have been selected. Execution can then compute outer addresses and
decide whether a read core needs scratch without consulting Python or walking
dynamic collections.
"""

from std.collections import InlineArray


comptime MAX_RANK = 8
comptime DimArray = InlineArray[Int, MAX_RANK]


@fieldwise_init
struct OperandPlan(Copyable):
    """Runtime address and axis metadata for one tensor operand.

    ``core_shape``/``core_stride`` and ``outer_shape``/``outer_stride`` are in
    logical order. ``outer_shape`` is replaced by the right-aligned common
    broadcast shape after all input plans have been inspected. Output plans
    must match that shape exactly; input singleton or missing dimensions get a
    zero outer stride and are reused by broadcasting.
    """

    var rank: Int
    var core_rank: Int
    var outer_rank: Int
    var core_length: Int
    var outer_count: Int
    var base_address: Int
    var shape: DimArray
    var stride: DimArray
    var core_shape: DimArray
    var core_stride: DimArray
    var outer_shape: DimArray
    var outer_stride: DimArray
    var core_contiguous: Bool

    @staticmethod
    def empty() -> Self:
        """Construct the zero metadata value used to initialize plans."""

        return Self(
            0,
            0,
            0,
            0,
            0,
            0,
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            DimArray(fill=0),
            True,
        )
