"""Runtime metadata shared by the guvectorize planner and tensor views."""

from std.collections import InlineArray


comptime MAX_RANK = 8
comptime DimArray = InlineArray[Int, MAX_RANK]


@fieldwise_init
struct OperandPlan(Copyable):
    """Runtime address and axis metadata for one tensor operand."""

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
