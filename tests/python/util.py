"""Adversarial array generator for parity testing — ported from numbagg's
test/util.py so mojagg is validated against the same hostile inputs.

Yields NaN/inf mixes, byte-swapped dtypes, 0-d arrays, ties, huge arrays,
non-contiguous views, and shuffled random data across float32/float64.
"""

from __future__ import annotations

import numpy as np

DTYPES = [np.float32, np.float64]


def arrays(func_name, dtypes=DTYPES):
    result = list(array_iter(array_generator, func_name, dtypes))
    assert len(result) > 0
    return result


def array_iter(arrays_func, *args):
    for a in arrays_func(*args):
        if a.ndim < 2:
            yield a
        else:
            yield a
            yield a.T


def array_generator(func_name, dtypes):
    """Iterator that yields arrays to use for unit testing."""
    if func_name in ("partition", "argpartition"):
        nan = 0
    else:
        nan = np.nan
    if func_name in ("move_sum", "move_mean", "move_std", "move_var"):
        # these functions can't handle inf
        inf = 8
    else:
        inf = np.inf

    # nan and inf
    yield np.array([inf, nan], dtype=np.float64)
    yield np.array([inf, -inf], dtype=np.float64)
    yield np.array([nan, 2, 3], dtype=np.float64)
    yield np.array([-inf, 2, 3], dtype=np.float64)
    if func_name != "nanargmin":
        yield np.array([nan, inf], dtype=np.float64)

    # byte swapped
    yield np.array([1, 2, 3], dtype=">f4")
    yield np.array([1, 2, 3], dtype="<f4")

    # float16 (exercises promotion path)
    yield np.array([1, 2, 3], dtype=np.float16)

    # regression tests
    yield np.array([1, 2, 3], dtype=np.float64) + 1e9
    yield np.array([0, 0, 0], dtype=np.float64)
    yield np.array([1, nan, nan, 2], dtype=np.float64)
    yield np.array([2**31], dtype=np.int64)
    yield np.array([[1.0, 2], [3, 4]], dtype=np.float64)[..., np.newaxis]

    # ties
    yield np.array([0, 0, 0], dtype=np.float64)
    yield np.array([1, 1, 1], dtype=np.float64)

    # 0d input
    if not func_name.startswith("move"):
        for v in (-9, 0, 9, -9.0, 0.0, 9.0, -inf, inf, nan):
            yield np.array(v, dtype=np.float64)

    # automated arrays of increasing size
    ss = {
        0: {"size": 0, "shapes": [(0,), (0, 0), (2, 0), (2, 0, 1)]},
        1: {"size": 8, "shapes": [(8,)]},
        2: {"size": 12, "shapes": [(2, 6), (3, 4)]},
        3: {"size": 16, "shapes": [(2, 2, 4)]},
        4: {"size": 24, "shapes": [(1, 2, 3, 4)]},
        5: {"size": 1_000, "shapes": [(1_000,)]},
        6: {"size": 10_000, "shapes": [(100, 100)]},
    }
    for seed in (1, 2):
        rs = np.random.RandomState(seed)
        for ndim in ss:
            size = ss[ndim]["size"]
            for dtype in dtypes:
                a = np.arange(size, dtype=dtype)
                if issubclass(a.dtype.type, np.inexact):
                    if func_name not in ("nanargmin", "nanargmax"):
                        idx = rs.rand(*a.shape) < 0.2
                        a[idx] = inf
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = nan
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] *= -1
                rs.shuffle(a)
                for shape in ss[ndim]["shapes"]:
                    yield a.reshape(shape)

    # non-contiguous arrays
    yield np.array([[1, 2], [3, 4]], dtype=np.int64)[:, [1]]
    for dtype in dtypes:
        a = np.arange(12).astype(dtype)
        for start in range(3):
            for step in range(1, 3):
                yield a[start::step]
    for dtype in dtypes:
        a2 = np.arange(12).reshape(4, 3).astype(dtype)
        yield a2[::2]
        yield a2[:, ::2]
        yield a2[::2][:, ::2]
    for dtype in dtypes:
        a3 = np.arange(24).reshape(2, 3, 4).astype(dtype)
        for start in range(2):
            for step in range(1, 3):
                yield a3[start::step]
                yield a3[:, start::step]
                yield a3[:, :, start::step]
                yield a3[start::step][::2]
                yield a3[start::step][::2][:, ::2]
