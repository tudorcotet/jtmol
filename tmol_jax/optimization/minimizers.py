"""High-level energy minimizers — pure JAX port.

Upstream reference: ``tmol/optimization/minimizers.py`` and
``tmol/optimization/sfxn_modules.py``. Upstream ties the minimizer to
``torch.nn.Module`` subclasses whose ``forward()`` method evaluates the
score function against a set of ``nn.Parameter`` coords / DOFs; the
optimizer then calls ``.step(closure)`` on them.

In the JAX port we drop the module indirection entirely. A minimizer
consumes:

* a ``PoseStack`` (immutable pytree; coords live on the pytree leaf
  ``coords``);
* a callable ``ScoreFunction`` that returns per-pose energies;
* optional move maps that mark coords / DOFs as frozen.

We flatten the parameters to a 1D vector, wrap the energy in a pure
function ``fn(x) -> scalar``, feed that to :func:`lbfgs_minimize`, and
unpack the optimized parameters back into a new ``PoseStack``. ``jax.grad``
handles every derivative; no bespoke autograd code is needed.

Two minimization flavours are provided:

* :func:`run_cart_min` — minimize directly in Cartesian atom-coordinate
  space. Corresponds to ``CartesianSfxnNetwork`` upstream.
* :func:`run_kin_min` — minimize in kinematic-DOF space. Builds a
  ``KinForest`` via :meth:`FoldForest.build_kinforest`, recovers the
  DOF 9-vector per node via :func:`inverse_kin`, then drives L-BFGS on
  the flattened DOF tensor. Forward kinematics and energy are composed
  under ``jax.grad``. Corresponds to ``KinForestSfxnNetwork`` upstream.

:func:`run_min` dispatches between the two based on a ``cartesian`` flag.
:func:`build_kinforest_network` is a small helper retained for API
compatibility with callers that used the upstream
``build_kinforest_network`` contract.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from tmol_jax.optimization.lbfgs import lbfgs_minimize


# ---------------------------------------------------------------------------
# Coord <-> flat vector helpers
# ---------------------------------------------------------------------------


def _flat_coord_mask(pose_stack: Any, move_map: Optional[Any]) -> jnp.ndarray:
    """Resolve the per-atom movable mask into the flat coord layout.

    The caller can pass either:

    * ``None`` — every atom is free.
    * a ``CartesianMoveMap`` — the ``move_atom`` field is broadcast
      across the three xyz columns.
    * a raw ``bool[n_poses, max_n_atoms_per_pose]`` or
      ``bool[n_poses, max_n_atoms_per_pose, 3]`` array — used directly.

    Returns a flat ``float32`` mask of shape ``(n_poses *
    max_n_atoms_per_pose * 3,)`` where ``1.0`` marks movable entries and
    ``0.0`` marks frozen entries. Using floats (not bool) makes the mask
    directly multiplicable against gradients inside the inner loop.
    """
    coords_shape = pose_stack.coords.shape  # [P, A, 3]
    ones = jnp.ones(coords_shape, dtype=jnp.float32)

    if move_map is None:
        return ones.reshape(-1)

    # CartesianMoveMap-like
    move_atom = getattr(move_map, "move_atom", None)
    if move_atom is not None:
        ma = jnp.asarray(move_atom, dtype=jnp.float32)  # [P, A]
        mask = jnp.broadcast_to(ma[..., None], coords_shape)
        return mask.reshape(-1)

    # Raw mask passed as-is
    mm = jnp.asarray(move_map, dtype=jnp.float32)
    if mm.ndim == 2:
        mm = jnp.broadcast_to(mm[..., None], coords_shape)
    return mm.reshape(-1)


# ---------------------------------------------------------------------------
# Cartesian minimization
# ---------------------------------------------------------------------------


def run_cart_min(
    pose_stack: Any,
    score_function: Any,
    move_map: Optional[Any] = None,
    max_iter: int = 100,
    history: int = 5,
    gtol: float = 1e-5,
    ftol: float = 1e-6,
    verbose: bool = False,
) -> Any:
    """Minimize energy w.r.t. Cartesian coordinates.

    Flattens ``pose_stack.coords`` to a 1D vector, calls
    :func:`lbfgs_minimize` on an energy closure that reshapes back and
    runs the score function, then unpacks the optimized coords into a
    new :class:`~tmol_jax.pose.pose_stack.PoseStack`.

    Parameters
    ----------
    pose_stack
        Input pose stack. Its ``coords`` leaf is the optimization target.
    score_function
        Callable returning per-pose energies. Must accept a
        ``coords_override`` keyword so we can inject the candidate coords
        each iteration without mutating the pose stack.
    move_map
        Optional :class:`CartesianMoveMap` or raw mask (see
        :func:`_flat_coord_mask`).
    max_iter, history, gtol, ftol
        Passed through to :func:`lbfgs_minimize`.
    verbose
        Forward to the minimizer's verbose flag.

    Returns
    -------
    PoseStack
        A new pose stack with optimized coordinates.
    """
    coords = jnp.asarray(pose_stack.coords)
    orig_shape = coords.shape
    x0 = coords.reshape(-1)

    mask = _flat_coord_mask(pose_stack, move_map)

    # Precompute per-pose term state so the setup doesn't retrace each
    # iteration. This mirrors upstream where ``render_whole_pose_scoring_module``
    # is called once in ``CartesianSfxnNetwork.__init__``.
    try:
        state = score_function.setup_for_pose_stack(pose_stack)
    except AttributeError:
        state = None

    # Prefer the ScoreFunction's own jit cache when available: it closes
    # over ``pose_stack`` and ``state`` as pytrees (static shape, traced
    # array leaves) and gets reused across ``run_cart_min`` calls for
    # the same pose stack. Falls back to a local closure if the user
    # passed a score-function-like object without ``jitted_value_and_grad``
    # (e.g. a legacy callable).
    jitted_vg = getattr(score_function, "jitted_value_and_grad", None)
    if jitted_vg is not None:
        vg_on_coords = jitted_vg(pose_stack, state=state)

        def value_and_grad_flat(x_flat: jnp.ndarray):
            c = x_flat.reshape(orig_shape).astype(coords.dtype)
            per_pose, grad_coords = vg_on_coords(c)
            return jnp.sum(per_pose), grad_coords.reshape(-1).astype(x_flat.dtype)

        def energy(x_flat: jnp.ndarray) -> jnp.ndarray:
            # Retained so ``lbfgs_minimize`` has a scalar ``fn`` for any
            # diagnostic path, but it's never differentiated — the
            # value-and-grad above shortcuts the whole thing.
            c = x_flat.reshape(orig_shape).astype(coords.dtype)
            per_pose, _ = vg_on_coords(c)
            return jnp.sum(per_pose)

        result = lbfgs_minimize(
            energy,
            x0,
            max_iter=max_iter,
            history=history,
            gtol=gtol,
            ftol=ftol,
            grad_mask=mask,
            verbose=verbose,
            value_and_grad=value_and_grad_flat,
        )
    else:
        def energy(x_flat: jnp.ndarray) -> jnp.ndarray:
            c = x_flat.reshape(orig_shape).astype(coords.dtype)
            if state is None:
                per_pose = score_function(pose_stack, coords_override=c)
            else:
                per_pose = score_function(pose_stack, state=state, coords_override=c)
            return jnp.sum(per_pose)

        result = lbfgs_minimize(
            energy,
            x0,
            max_iter=max_iter,
            history=history,
            gtol=gtol,
            ftol=ftol,
            grad_mask=mask,
            verbose=verbose,
        )

    new_coords = result["x"].reshape(orig_shape).astype(coords.dtype)
    return pose_stack.replace(coords=new_coords)


# ---------------------------------------------------------------------------
# Kinematic-DOF minimization
# ---------------------------------------------------------------------------


def _build_kinforest(pose_stack: Any, fold_forest: Optional[Any]):
    """Construct a :class:`KinForest` for the given pose stack / fold forest.

    If ``fold_forest`` is ``None``, we fall back to
    :meth:`FoldForest.from_pose_stack`, matching upstream's implicit
    default (and the behaviour of ``build_kinforest_network`` in
    ``sfxn_modules.py`` when no fold forest was passed in).
    """
    from tmol_jax.kinematics import FoldForest

    if fold_forest is None:
        fold_forest = FoldForest.from_pose_stack(pose_stack)
    kinforest = fold_forest.build_kinforest(pose_stack)
    return fold_forest, kinforest


def _kfo_to_pose_coords(
    kin_coords: jnp.ndarray,
    kinforest: Any,
    orig_shape: Tuple[int, int, int],
) -> jnp.ndarray:
    """Scatter kin-forest-ordered coords back into the pose-stack layout.

    ``kinforest.id`` holds the target-order index for every node (with
    ``id == -1`` for the forest root and optionally other sentinel
    slots). We only copy rows where ``id >= 0``; everything else is left
    at the pose-stack default (zero-initialized here because we'll be
    overwriting the coords tensor wholesale).
    """
    n_poses, max_n_atoms, _ = orig_shape
    flat_coords = jnp.zeros((n_poses * max_n_atoms, 3), dtype=kin_coords.dtype)
    ids = jnp.asarray(kinforest.id, dtype=jnp.int32)
    # Valid entries (id >= 0) scatter into flat_coords.
    valid = ids >= 0
    safe_ids = jnp.where(valid, ids, 0)
    # Segment-update: flat_coords[safe_ids] = kin_coords; then zero out
    # the invalid writes. Using jnp.where on the source dodges needing
    # a scatter-with-mask primitive.
    src = jnp.where(valid[:, None], kin_coords, jnp.zeros_like(kin_coords))
    flat_coords = flat_coords.at[safe_ids].set(src)
    return flat_coords.reshape(orig_shape)


def build_kinforest_network(
    pose_stack: Any,
    fold_forest: Optional[Any] = None,
    move_map: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the KinForest plumbing needed for an internal-coord minimization.

    Upstream exposes a ``build_kinforest_network`` helper returning a
    ``KinForestSfxnNetwork`` module. In the JAX port we replace that
    stateful object with a plain dict describing the pieces needed by
    :func:`run_kin_min`:

    ``fold_forest``
        The resolved :class:`FoldForest` (auto-built from the pose stack
        if none was passed).
    ``kinforest``
        Atom-level :class:`KinForest`.
    ``full_dofs``
        The ``[N, 9]`` DOF tensor recovered from the initial coords via
        :func:`inverse_kin`.
    ``dof_mask``
        Float mask of the same shape as ``full_dofs``. ``1.0`` marks
        movable DOFs. Built from the supplied ``move_map`` if present,
        else defaults to the upstream ``phi_c`` for bond nodes plus the
        first six DOFs for jump nodes.

    This matches the upstream initializer's data layout so downstream
    callers (tests, relaxation pipelines, etc.) can pull pieces out
    independently.
    """
    from tmol_jax.kinematics import inverse_kin
    from tmol_jax.kinematics.datatypes import NodeType, BondDOFTypes, JumpDOFTypes

    fold_forest, kinforest = _build_kinforest(pose_stack, fold_forest)

    # Flatten pose-stack coords then gather into kin-forest order.
    orig_shape = pose_stack.coords.shape  # [P, A, 3]
    flat_coords = jnp.asarray(pose_stack.coords, dtype=jnp.float64).reshape(-1, 3)
    ids = jnp.asarray(kinforest.id, dtype=jnp.int32)
    n_nodes = int(ids.shape[0])
    safe_ids = jnp.where(ids >= 0, ids, 0)
    kincoords = flat_coords[safe_ids]
    # Force the root node (id == -1) to origin to match the upstream C++
    # ``hts_from_frames`` contract that depends on HTs[0] = identity.
    kincoords = kincoords.at[0].set(jnp.zeros(3, dtype=jnp.float64))

    dof = inverse_kin(kinforest, kincoords)
    full_dofs = dof.raw  # [N, 9], float64

    # DOF mask. Without a move map we use upstream's default: torsion
    # DOFs on bond nodes, first six DOFs on jump nodes.
    if move_map is not None and hasattr(move_map, "to_dof_mask"):
        dof_mask_bool = jnp.asarray(move_map.to_dof_mask(kinforest), dtype=jnp.bool_)
    else:
        dof_mask_bool = jnp.zeros((n_nodes, 9), dtype=jnp.bool_)
        doftype = jnp.asarray(kinforest.doftype, dtype=jnp.int32)
        bond_rows = doftype == int(NodeType.bond)
        jump_rows = doftype == int(NodeType.jump)
        # Enable phi_c on bond nodes, first six DOFs on jump nodes.
        col_idx = jnp.arange(9)
        bond_col_mask = col_idx == int(BondDOFTypes.phi_c)
        jump_col_mask = col_idx < int(JumpDOFTypes.RBdel_gamma) + 1
        dof_mask_bool = jnp.where(
            bond_rows[:, None] & bond_col_mask[None, :], True, dof_mask_bool
        )
        dof_mask_bool = jnp.where(
            jump_rows[:, None] & jump_col_mask[None, :], True, dof_mask_bool
        )

    dof_mask = dof_mask_bool.astype(jnp.float64)

    return {
        "fold_forest": fold_forest,
        "kinforest": kinforest,
        "full_dofs": full_dofs,
        "dof_mask": dof_mask,
    }


def run_kin_min(
    pose_stack: Any,
    score_function: Any,
    fold_forest: Optional[Any] = None,
    move_map: Optional[Any] = None,
    max_iter: int = 100,
    history: int = 5,
    gtol: float = 1e-5,
    ftol: float = 1e-6,
    verbose: bool = False,
) -> Any:
    """Minimize energy w.r.t. kinematic DOFs (internal coordinates).

    Procedure:

    1. Build the :class:`KinForest` via
       :func:`build_kinforest_network`.
    2. Recover the initial DOF 9-vector per node via
       :func:`inverse_kin`.
    3. Flatten the movable DOFs into a 1D vector; compose
       ``forward_kin -> scatter-to-pose-layout -> score_function`` into a
       scalar-output energy function.
    4. Run :func:`lbfgs_minimize` on that energy.
    5. Re-run ``forward_kin`` with the optimized DOFs and write the
       updated coords into a new :class:`PoseStack`.

    ``jax.grad`` walks through the forward-kin + score composition
    automatically.
    """
    from tmol_jax.kinematics.operations import forward_kin

    net = build_kinforest_network(pose_stack, fold_forest=fold_forest, move_map=move_map)
    kinforest = net["kinforest"]
    full_dofs = jnp.asarray(net["full_dofs"], dtype=jnp.float64)
    dof_mask = jnp.asarray(net["dof_mask"], dtype=jnp.float64)

    orig_shape = pose_stack.coords.shape  # [P, A, 3]
    dofs_shape = full_dofs.shape  # [N, 9]
    x0 = full_dofs.reshape(-1)
    mask = dof_mask.reshape(-1)

    try:
        state = score_function.setup_for_pose_stack(pose_stack)
    except AttributeError:
        state = None

    def energy(x_flat: jnp.ndarray) -> jnp.ndarray:
        dofs = x_flat.reshape(dofs_shape).astype(jnp.float64)
        kin_coords = forward_kin(kinforest, dofs)  # [N, 3], float64
        pose_coords = _kfo_to_pose_coords(kin_coords, kinforest, orig_shape)
        pose_coords = pose_coords.astype(jnp.float32)
        if state is None:
            per_pose = score_function(pose_stack, coords_override=pose_coords)
        else:
            per_pose = score_function(
                pose_stack, state=state, coords_override=pose_coords
            )
        return jnp.sum(per_pose)

    result = lbfgs_minimize(
        energy,
        x0,
        max_iter=max_iter,
        history=history,
        gtol=gtol,
        ftol=ftol,
        grad_mask=mask,
        verbose=verbose,
    )

    final_dofs = result["x"].reshape(dofs_shape).astype(jnp.float64)
    # Apply the mask once more so frozen DOFs are pinned at their
    # initial values (lbfgs_minimize already does this inside, but be
    # defensive — external callers rely on exact DOF preservation).
    final_dofs = jnp.where(dof_mask != 0.0, final_dofs, full_dofs)

    final_kin_coords = forward_kin(kinforest, final_dofs)
    final_pose_coords = _kfo_to_pose_coords(
        final_kin_coords, kinforest, orig_shape
    ).astype(pose_stack.coords.dtype)

    return pose_stack.replace(coords=final_pose_coords)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def run_min(
    pose_stack: Any,
    score_function: Any,
    cartesian: bool = True,
    **kwargs,
) -> Any:
    """Run energy minimization on ``pose_stack``.

    Dispatches to :func:`run_cart_min` when ``cartesian=True`` (the
    default, matching upstream's common usage) or :func:`run_kin_min`
    otherwise. Unknown keyword arguments are forwarded verbatim.
    """
    if cartesian:
        return run_cart_min(pose_stack, score_function, **kwargs)
    return run_kin_min(pose_stack, score_function, **kwargs)


__all__ = [
    "build_kinforest_network",
    "run_cart_min",
    "run_kin_min",
    "run_min",
]
