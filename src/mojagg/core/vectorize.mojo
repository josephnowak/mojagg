"""Vectorization helpers for kernels with explicit scalar tails."""

from std.algorithm import vectorize


@always_inline
def vectorize_no_evl[
    func: def[vector_width: Int](Int, Int) -> None,
    //,
    width: Int,
    /,
    *,
    unroll_factor: Int = 1,
](length: Int, closure: func):
    """Vectorize full-width blocks and process the remaining tail at width 1.

    Unlike ``std.algorithm.vectorize`` over an unaligned length, this helper
    never asks the body to handle a partial SIMD block.  The tail callback is
    shifted back to its absolute input offset and receives an EVL of one.
    """

    comptime assert width > 0, "vector width must be positive"
    var aligned_length = length - (length % width)
    vectorize[width, unroll_factor=unroll_factor](aligned_length, closure)

    var tail = length - aligned_length
    if tail > 0:

        def tail_body[
            vector_width: Int
        ](i: Int, evl: Int) {imm closure, imm aligned_length}:
            closure[1](i + aligned_length, evl)

        vectorize[1, unroll_factor=unroll_factor](tail, tail_body)
