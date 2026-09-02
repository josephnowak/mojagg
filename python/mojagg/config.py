"""Layered runtime configuration for mojagg.

Resolution order (highest priority first):

1. context manager overrides (thread-local)  — ``mojagg.config(...)``
2. global settings                            — ``mojagg.set_config(...)``
3. environment variables                      — ``MOJAGG_*``
4. tuned defaults                             — benchmark-derived, in this file

The effective config is resolved ONCE per public call and passed down to the
native kernels by value. Kernels never read globals or env vars themselves.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

_ENV_PREFIX = "MOJAGG_"

# env var name for each field
_ENV_VARS = {
    "backend": "MOJAGG_BACKEND",
    "threads": "MOJAGG_THREADS",
    "parallel_threshold": "MOJAGG_PARALLEL_THRESHOLD",
    "parallel_min_groups": "MOJAGG_PARALLEL_MIN_GROUPS",
    "gpu_min_bytes": "MOJAGG_GPU_MIN_BYTES",
    "simd_width": "MOJAGG_SIMD_WIDTH",
}


@dataclass(frozen=True, slots=True)
class MojaggConfig:
    """Resolved dispatch configuration. Passed by value into native calls."""

    backend: str = "auto"           # "auto" | "cpu" | "gpu"
    threads: int = 0                # 0 = physical cores
    parallel_threshold: int = 500_000
    parallel_min_groups: int = 64
    gpu_min_bytes: int = 1 << 26    # 64 MiB
    simd_width: int = 0             # 0 = native

    @classmethod
    def from_env(cls) -> MojaggConfig:
        """Build a config from MOJAGG_* environment variables (read at import)."""
        kwargs: dict[str, Any] = {}
        for field, env in _ENV_VARS.items():
            raw = os.environ.get(env)
            if raw is None:
                continue
            if field == "backend":
                kwargs[field] = raw.strip().lower()
            else:
                kwargs[field] = int(raw)
        return cls(**kwargs)


# Global singleton (set via set_config), layered over env-derived defaults.
_global = MojaggConfig.from_env()

# Thread-local stack for the context manager.
_tls = threading.local()


def _stack() -> list[dict[str, Any]]:
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


def get_config() -> MojaggConfig:
    """Resolve the effective config for the current thread."""
    overrides: dict[str, Any] = {}
    for frame in _stack():
        overrides.update(frame)
    if overrides:
        return replace(_global, **overrides)
    return _global


@contextmanager
def config(**overrides: Any) -> Iterator[None]:
    """Temporarily override config within a with-block (thread-local).

    >>> with mojagg.config(parallel_threshold=50_000, threads=8):
    ...     mojagg.group_nansum(values, labels)
    """
    unknown = set(overrides) - set(_ENV_VARS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}; valid: {sorted(_ENV_VARS)}")
    _stack().append(overrides)
    try:
        yield
    finally:
        _stack().pop()


def set_config(**overrides: Any) -> None:
    """Set global config (process-wide). Lower priority than config()."""
    global _global
    unknown = set(overrides) - set(_ENV_VARS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}; valid: {sorted(_ENV_VARS)}")
    _global = replace(_global, **overrides)
