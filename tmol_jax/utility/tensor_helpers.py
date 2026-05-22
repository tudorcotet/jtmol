"""Small JAX/NumPy tensor utilities used across the tmol_jax port.

These replace ad-hoc helpers scattered through upstream tmol's
``utility/tensor`` and ``utility/ndarray`` modules. They are deliberately
JAX-first (``jnp`` operations, jit-friendly) and do not allocate on a
specific device.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def segment_sum_2d(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    """Sum ``data`` rows grouped by ``segment_ids`` into ``num_segments`` buckets.

    Args:
        data: ``(N, D)`` array of row vectors.
        segment_ids: ``(N,)`` int array in ``[0, num_segments)``. Negative ids
            are dropped (standard JAX convention).
        num_segments: Size of the output's leading dimension.

    Returns:
        ``(num_segments, D)`` array where row ``k`` is the sum of rows whose
        ``segment_ids == k``.
    """
    data = jnp.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"segment_sum_2d expects 2D data, got shape {data.shape}")
    segment_ids = jnp.asarray(segment_ids)
    return jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)


def exclusive_cumsum(
    arr: jnp.ndarray,
    axis: int = -1,
) -> jnp.ndarray:
    """Exclusive cumulative sum along ``axis``: output[i] = sum(arr[:i])."""
    arr = jnp.asarray(arr)
    cs = jnp.cumsum(arr, axis=axis)
    # shift by one along axis, zero-padding the front
    zero_shape = list(arr.shape)
    zero_shape[axis] = 1
    zeros = jnp.zeros(zero_shape, dtype=arr.dtype)
    # drop the last slice along axis
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(0, arr.shape[axis] - 1)
    return jnp.concatenate([zeros, cs[tuple(slicer)]], axis=axis)


# ---------------------------------------------------------------------------
# Gather / mask helpers
# ---------------------------------------------------------------------------


def gather_with_default(
    source: jnp.ndarray,
    indices: jnp.ndarray,
    default,
    fill_value=None,
) -> jnp.ndarray:
    """Gather ``source[indices]`` but replace slots where ``indices == default``
    with ``fill_value`` (broadcast along the source's trailing dims).

    Designed for sentinel-indexed lookups (e.g. ``-1`` means "missing").

    Args:
        source: ``(M, *trailing)`` array.
        indices: ``(*leading,)`` int array of gather positions; entries equal
            to ``default`` are masked out.
        default: Sentinel index value (e.g. ``-1``).
        fill_value: Value used for masked entries. Defaults to ``0``.
    """
    source = jnp.asarray(source)
    indices = jnp.asarray(indices)
    if fill_value is None:
        fill_value = 0

    # clamp to a safe index so the gather itself is well-defined, then mask.
    safe = jnp.where(indices == default, 0, indices)
    gathered = source[safe]

    mask_shape = indices.shape + (1,) * (gathered.ndim - indices.ndim)
    mask = (indices == default).reshape(mask_shape)
    fill = jnp.asarray(fill_value, dtype=gathered.dtype)
    return jnp.where(mask, fill, gathered)


def where_broadcast(
    condition: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """``jnp.where`` with explicit up-front broadcasting.

    Thin wrapper that avoids subtle shape surprises when ``condition`` has a
    different rank than ``x`` / ``y``.
    """
    condition = jnp.asarray(condition)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    out_shape = jnp.broadcast_shapes(condition.shape, x.shape, y.shape)
    return jnp.where(
        jnp.broadcast_to(condition, out_shape),
        jnp.broadcast_to(x, out_shape),
        jnp.broadcast_to(y, out_shape),
    )


# ---------------------------------------------------------------------------
# Padding / one-hot
# ---------------------------------------------------------------------------


def pad_to_shape(
    arr: jnp.ndarray,
    shape: Sequence[int],
    fill: float = 0,
) -> jnp.ndarray:
    """Pad ``arr`` on the right of each dimension to ``shape`` with ``fill``.

    ``shape`` must have the same rank as ``arr`` and every entry must be at
    least as large as the current size along that axis.
    """
    arr = jnp.asarray(arr)
    shape = tuple(int(s) for s in shape)
    if len(shape) != arr.ndim:
        raise ValueError(
            f"pad_to_shape: rank mismatch — arr.ndim={arr.ndim}, target={shape}"
        )
    pad_width: Tuple[Tuple[int, int], ...] = tuple(
        (0, s - a) for a, s in zip(arr.shape, shape)
    )
    for a, s in zip(arr.shape, shape):
        if s < a:
            raise ValueError(
                f"pad_to_shape: target {shape} is smaller than source {arr.shape}"
            )
    return jnp.pad(arr, pad_width, mode="constant", constant_values=fill)


def one_hot(indices: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """One-hot encode ``indices`` over ``num_classes`` categories.

    Delegates to :func:`jax.nn.one_hot`. Output dtype is ``float32``; callers
    that need integer masks should cast downstream.
    """
    indices = jnp.asarray(indices)
    return jax.nn.one_hot(indices, num_classes)


__all__ = [
    "segment_sum_2d",
    "exclusive_cumsum",
    "gather_with_default",
    "where_broadcast",
    "pad_to_shape",
    "one_hot",
]


# ---------------------------------------------------------------------------
# Private numpy fallback (occasionally useful outside of jit contexts).
# ---------------------------------------------------------------------------


def _np_exclusive_cumsum(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy-side exclusive cumsum. Convenience for host-side index math."""
    arr = np.asarray(arr)
    cs = np.cumsum(arr, axis=axis)
    zero_shape = list(arr.shape)
    zero_shape[axis] = 1
    zeros = np.zeros(zero_shape, dtype=arr.dtype)
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(0, arr.shape[axis] - 1)
    return np.concatenate([zeros, cs[tuple(slicer)]], axis=axis)
