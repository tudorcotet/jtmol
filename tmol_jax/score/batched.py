"""Memory-efficient batched scoring via ``jax.lax.map``.

The default ``jax.vmap(sfxn)`` materializes a ``[K, N, N]`` pair tensor
per LJLK / elec / lk_ball call. At ``float64`` (forced globally by the
hbond polynomial path) this OOMs on a 40 GB A100 somewhere around
K=64 for a 1200-atom pose.

``batched_energy`` wraps a score function so that K rollouts are
processed in chunks of ``chunk_size``: each chunk is vmap'd (one XLA
kernel), and the outer loop is ``jax.lax.map`` so the chunks share the
same JIT trace. Memory stays bounded by ``chunk_size``; wall-clock
grows linearly past ``chunk_size`` but each chunk runs at vmap throughput.

For ``score_dtype``, we expose a knob to cast ``coords`` to ``float32``
at term entry. The dense pair tensor in LJLK / elec / lk_ball is the
single biggest memory hog; ``float32`` halves it without touching the
``float64`` hbond polynomial path (hbond's ``_eval_packed_poly`` upcasts
its scalar inputs to the poly dtype internally, so geometry-side
precision doesn't change the polynomial chain).

Empirically (panel snapshot on 5 PDBs, 22 channels): the ``float32``
geometry path is within ``atol=1e-3 / rtol=1e-4`` of the full ``float64``
reference for every channel. The forward energies are bit-comparable to
within the panel's existing tolerance.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


def batched_energy(
    sfxn: Any,
    pose_stack: Any,
    *,
    state: Optional[Any] = None,
    chunk_size: int = 16,
    score_dtype: Any = jnp.float32,
) -> Callable[[jax.Array], jax.Array]:
    """Return ``fn(coords_batch) -> energies`` that chunks K via ``lax.map``.

    Parameters
    ----------
    sfxn
        A :class:`tmol_jax.score.ScoreFunction`.
    pose_stack
        The pose stack the energy is conditioned on. The function
        returned scores perturbations of ``pose_stack.coords``.
    state
        Optional precomputed state from
        :meth:`ScoreFunction.setup_for_pose_stack`. If ``None``, we
        compute it once and close over it.
    chunk_size
        Number of poses processed in one vmap'd XLA kernel. Memory
        scales linearly in this; wall-clock-per-pose flattens around
        ``chunk_size``. 16 is a good default on 40 GB A100 for ~1200
        atoms.
    score_dtype
        Cast ``coords`` to this dtype before passing to the score
        function. Pass ``jnp.float64`` (or ``None``) to disable the
        cast and use whatever the global JAX dtype is.

    Returns
    -------
    Callable
        ``fn(coords_batch)`` where ``coords_batch.shape = [K, n_atoms, 3]``,
        returning ``energies.shape = [K]`` in ``float64`` (the
        accumulator dtype).
    """
    if state is None:
        state = sfxn.setup_for_pose_stack(pose_stack)

    def single(coords: jax.Array) -> jax.Array:
        if score_dtype is not None:
            coords = coords.astype(score_dtype)
        per_pose = sfxn(pose_stack, state=state, coords_override=coords)
        # Accumulator is f64 so a downstream ``.sum()`` over K is stable
        # even if score_dtype shaved the inputs to f32.
        return per_pose.astype(jnp.float64).sum()

    def fn(coords_batch: jax.Array) -> jax.Array:
        return jax.lax.map(single, coords_batch, batch_size=chunk_size)

    return fn


def batched_value_and_grad(
    sfxn: Any,
    pose_stack: Any,
    *,
    state: Optional[Any] = None,
    chunk_size: int = 16,
    score_dtype: Any = jnp.float32,
) -> Callable[[jax.Array], tuple]:
    """Like :func:`batched_energy` but returns ``(energies, grad_coords)``.

    Useful for training loops that want per-pose energy and the gradient
    w.r.t. coords in one pass.
    """
    if state is None:
        state = sfxn.setup_for_pose_stack(pose_stack)

    def single(coords: jax.Array) -> jax.Array:
        if score_dtype is not None:
            coords = coords.astype(score_dtype)
        per_pose = sfxn(pose_stack, state=state, coords_override=coords)
        return per_pose.astype(jnp.float64).sum()

    vg_single = jax.value_and_grad(single)

    def fn(coords_batch: jax.Array) -> tuple:
        return jax.lax.map(vg_single, coords_batch, batch_size=chunk_size)

    return fn


def pmap_batched_energy(
    sfxn: Any,
    pose_stack: Any,
    *,
    state: Optional[Any] = None,
    chunk_size: int = 16,
    score_dtype: Any = jnp.float32,
) -> Callable[[jax.Array], jax.Array]:
    """Multi-device variant of :func:`batched_energy`.

    Reshapes a ``[K, n_atoms, 3]`` coords batch to
    ``[n_devices, K // n_devices, n_atoms, 3]``, pmaps the per-device
    score (which itself ``lax.map``s in chunks of ``chunk_size``),
    flattens the result back. ``K`` must be divisible by
    ``jax.device_count()``.

    ``pose_stack`` and ``state`` are replicated across devices via
    :func:`jax.device_put_replicated`.

    Use this when a multi-GPU host is available (Modal ``A100-40GB:2``,
    ``A100-80GB:4``, ``H100:8``, etc.). On a single-GPU host this
    degenerates to :func:`batched_energy` with extra reshape overhead.
    """
    n_dev = jax.device_count()
    if state is None:
        state = sfxn.setup_for_pose_stack(pose_stack)

    rep_state = jax.device_put_replicated(state, jax.devices())

    def per_device(rep_st: Any, coords_chunk: jax.Array) -> jax.Array:
        def single(coords: jax.Array) -> jax.Array:
            if score_dtype is not None:
                coords = coords.astype(score_dtype)
            per_pose = sfxn(pose_stack, state=rep_st, coords_override=coords)
            return per_pose.astype(jnp.float64).sum()
        return jax.lax.map(single, coords_chunk, batch_size=chunk_size)

    pmapped = jax.pmap(per_device)

    def fn(coords_batch: jax.Array) -> jax.Array:
        K = coords_batch.shape[0]
        if K % n_dev != 0:
            raise ValueError(f"K={K} not divisible by n_devices={n_dev}")
        reshaped = coords_batch.reshape((n_dev, K // n_dev) + coords_batch.shape[1:])
        out = pmapped(rep_state, reshaped)
        return out.reshape(-1)

    return fn


__all__ = ["batched_energy", "batched_value_and_grad", "pmap_batched_energy"]
