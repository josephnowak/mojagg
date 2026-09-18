"""Shared value-only CPU scheduling policy."""

from std.sys.info import num_logical_cores


@fieldwise_init
struct DispatchPolicy(Copyable):
    var workers: Int
    var parallel_threshold: Int
    var parallel_min_groups: Int

    @always_inline
    def effective_workers(self, outer_count: Int, inner_length: Int) -> Int:
        var requested = (
            self.workers if self.workers > 0 else num_logical_cores()
        )
        if outer_count <= 1 or requested <= 1:
            return 1
        # Parallel dispatch has two independent, inclusive gates: enough outer
        # invocations to distribute and enough work in each input core to make
        # the worker-pool launch worthwhile.
        if outer_count < max(self.parallel_min_groups, 1):
            return 1
        if inner_length < max(self.parallel_threshold, 0):
            return 1
        return min(requested, outer_count)
