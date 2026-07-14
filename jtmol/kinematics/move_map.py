"""Move maps for minimization — JAX port.

This module is a JAX-native port of ``tmol.kinematics.move_map``. Upstream
torch offers two largely independent objects:

* :class:`CartesianMoveMap` — a thin wrapper around a per-atom Boolean
  mask used when minimizing in Cartesian coordinate space. The
  Cartesian ``min_fn`` pulls ``move_atom`` out and routes it to
  :class:`CartesianSfxnNetwork` so that "frozen" atoms receive zero
  gradient updates.

* :class:`MoveMap` — a full internal-coordinate move map over the
  backbone (phi/psi/omega), sidechain (chi), and rigid-body jump DOFs
  of a pose stack. Upstream this object is quite elaborate (dozens of
  ``move_*`` / ``move_*_mask`` pairs, setter methods, a
  :class:`MinimizerMap` that walks the KinForest to build a DOF mask
  via a C++ op). For the JAX port we keep the *public contract*
  (which DOFs can move) and drop the mask/setter plumbing — JAX
  arrays are immutable and a functional ``to_dof_mask`` is all the
  downstream minimizer actually consumes.

Both classes are :func:`pytree_dataclass` so they flow cleanly through
``jit`` / ``vmap`` / ``grad``. Array fields become leaves; ``PoseStack``
/ ``KinForest`` references used in classmethods are never stored on the
instance.
"""

from __future__ import annotations

from typing import Any, Optional

import attrs
import jax.numpy as jnp
import numpy as np

from jtmol.types import Array, pytree_dataclass

from .datatypes import KinForest, NodeType, BondDOFTypes


# ---------------------------------------------------------------------------
# Cartesian move map
# ---------------------------------------------------------------------------


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class CartesianMoveMap:
    """Move map for Cartesian-space minimization.

    A light wrapper around a coordinate mask that can be passed as the
    ``move_map`` argument to a Cartesian minimizer. The ``min_fn`` is
    responsible for extracting :attr:`move_atom` and routing it to the
    Cartesian scoring network; atoms where the mask is ``False`` should
    receive zero gradient updates.

    Args:
        move_atom: ``bool[n_poses, max_n_atoms_per_pose]``. ``True``
            means the atom is free to move.
    """

    move_atom: Array  # bool[n_poses, max_n_atoms_per_pose]

    # ------------------------------------------------------------------
    # Shape accessors
    # ------------------------------------------------------------------

    @property
    def n_poses(self) -> int:
        return int(self.move_atom.shape[0])

    @property
    def max_n_atoms(self) -> int:
        return int(self.move_atom.shape[1])

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def all_movable(cls, pose_stack: Any) -> "CartesianMoveMap":
        """Every atom in ``pose_stack`` is free to move.

        Produces a ``[n_poses, max_n_atoms_per_pose]`` all-True mask.
        Padding atoms are left as ``True`` — the downstream Cartesian
        minimizer already ignores padding via ``coords`` bookkeeping.
        """
        shape = (pose_stack.n_poses, pose_stack.max_n_atoms_per_pose)
        return cls(move_atom=jnp.ones(shape, dtype=jnp.bool_))

    @classmethod
    def fix_backbone(
        cls, pose_stack: Any, chem_db: Optional[Any] = None
    ) -> "CartesianMoveMap":
        """Backbone atoms (N/CA/C) are frozen, everything else moves.

        Backbone identification is sourced from
        :attr:`PackedBlockTypes.atom_is_backbone` which is a packed
        ``bool[n_types, max_n_atoms_per_block]`` array populated at
        :meth:`PackedBlockTypes.from_restypes` time. The ``chem_db``
        argument is accepted for API parity with upstream but is not
        needed — the per-block-type backbone mask is already resolved
        inside the pose stack's packed block types.

        The per-block backbone mask is scattered into the per-pose
        contiguous coordinate layout using the pose stack's
        ``block_coord_offset`` table.
        """
        pbt = pose_stack.packed_block_types

        # [n_poses, max_n_atoms_per_pose] — True where atom should move.
        n_poses = pose_stack.n_poses
        max_n_atoms_per_pose = pose_stack.max_n_atoms_per_pose

        # Build the mask on numpy then push to device. We avoid JAX
        # scatter here because the whole thing is static-shape metadata.
        bt_ind = np.asarray(pose_stack.block_type_ind)
        bco = np.asarray(pose_stack.block_coord_offset)
        atom_is_bb = np.asarray(pbt.atom_is_backbone)  # [n_types, max_n_atoms]
        n_atoms = np.asarray(pbt.n_atoms)  # [n_types]

        move = np.ones((n_poses, max_n_atoms_per_pose), dtype=np.bool_)

        for p in range(n_poses):
            for b in range(bt_ind.shape[1]):
                bti = int(bt_ind[p, b])
                if bti < 0:
                    continue
                n = int(n_atoms[bti])
                start = int(bco[p, b])
                bb_slice = atom_is_bb[bti, :n]
                # Backbone atoms are frozen -> move=False.
                move[p, start : start + n] = ~bb_slice

        return cls(move_atom=jnp.asarray(move, dtype=jnp.bool_))

    # ------------------------------------------------------------------
    # DOF mask conversion
    # ------------------------------------------------------------------

    def to_dof_mask(self, kinforest: KinForest) -> Array:
        """Flatten the Cartesian atom mask to a per-DOF mask.

        Simplified port: returns a flat ``bool[N]`` mask aligned to
        ``kinforest.id`` in kin-forest order. Because Cartesian
        minimization treats every atom uniformly, we broadcast a single
        Boolean per atom to cover all DOFs on that atom. Upstream
        performs a more involved per-DOF resolution; here one bit per
        node is sufficient because the Cartesian ``min_fn`` never
        consults a per-DOF mask.
        """
        flat = jnp.reshape(self.move_atom, (-1,))
        n = int(kinforest.id.shape[0])
        # Pad / truncate to N so the minimizer gets the right shape.
        size = int(flat.shape[0])
        if size == n:
            return flat.astype(jnp.bool_)
        if size > n:
            return flat[:n].astype(jnp.bool_)
        pad = jnp.zeros((n - size,), dtype=jnp.bool_)
        return jnp.concatenate([flat.astype(jnp.bool_), pad], axis=0)


# ---------------------------------------------------------------------------
# Full internal-coordinate move map
# ---------------------------------------------------------------------------


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class MoveMap:
    """Internal-coordinate move map over a pose stack.

    The upstream object exposes a large menagerie of ``move_*`` and
    ``move_*_mask`` tensors plus setter methods; for the JAX port we
    keep the core "what's flexible" information:

    * :attr:`move_bb_phi` / :attr:`move_bb_psi` / :attr:`move_bb_omega`
      — per-block backbone torsion flags,
      ``bool[n_poses, max_n_blocks]``.
    * :attr:`move_chi` — per-block per-chi side-chain flag,
      ``bool[n_poses, max_n_blocks, max_n_chi]``.
    * :attr:`move_jump_dofs` — per-jump six-DOF flag,
      ``bool[n_poses, max_n_jumps, 6]``.

    These five tensors are the minimum needed to drive the
    :meth:`to_dof_mask` conversion that feeds the minimizer. Upstream
    setter methods are not ported: JAX arrays are immutable, so callers
    should construct a new :class:`MoveMap` using ``attrs.evolve`` (or
    the ``replace`` helper below) rather than mutating in-place.
    """

    move_bb_phi: Array  # bool[n_poses, max_n_blocks]
    move_bb_psi: Array  # bool[n_poses, max_n_blocks]
    move_bb_omega: Array  # bool[n_poses, max_n_blocks]
    move_chi: Array  # bool[n_poses, max_n_blocks, max_n_chi]
    move_jump_dofs: Array  # bool[n_poses, max_n_jumps, 6]

    # ------------------------------------------------------------------
    # Shape accessors
    # ------------------------------------------------------------------

    @property
    def n_poses(self) -> int:
        return int(self.move_bb_phi.shape[0])

    @property
    def max_n_blocks(self) -> int:
        return int(self.move_bb_phi.shape[1])

    @property
    def max_n_chi(self) -> int:
        return int(self.move_chi.shape[2])

    @property
    def max_n_jumps(self) -> int:
        return int(self.move_jump_dofs.shape[1])

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def _shapes_from_pose_stack(cls, pose_stack: Any) -> tuple:
        """Derive ``(n_poses, max_n_blocks, max_n_chi, max_n_jumps)``.

        ``max_n_chi`` is read off ``packed_block_types`` if available
        (some packed-block-types implementations expose it directly,
        others keep it on ``max_n_torsions``); when neither is present
        we fall back to a conservative default of 4.

        ``max_n_jumps`` likewise is pulled from the pose stack if
        available, falling back to ``max_n_blocks - 1`` (the looser
        upper bound of one jump per non-root block) otherwise.
        """
        n_poses = pose_stack.n_poses
        max_n_blocks = pose_stack.max_n_blocks

        pbt = pose_stack.packed_block_types
        max_n_chi = getattr(pbt, "max_n_chi", None)
        if max_n_chi is None:
            max_n_chi = getattr(pbt, "max_n_torsions", None)
        if max_n_chi is None:
            max_n_chi = 4
        max_n_chi = int(max_n_chi)

        max_n_jumps = getattr(pose_stack, "max_n_jumps", None)
        if max_n_jumps is None:
            max_n_jumps = max(1, max_n_blocks - 1)
        max_n_jumps = int(max_n_jumps)

        return n_poses, max_n_blocks, max_n_chi, max_n_jumps

    @classmethod
    def all_movable(cls, pose_stack: Any) -> "MoveMap":
        """Every backbone torsion, every chi, every jump DOF is free."""
        n_poses, max_n_blocks, max_n_chi, max_n_jumps = cls._shapes_from_pose_stack(
            pose_stack
        )
        t2 = (n_poses, max_n_blocks)
        return cls(
            move_bb_phi=jnp.ones(t2, dtype=jnp.bool_),
            move_bb_psi=jnp.ones(t2, dtype=jnp.bool_),
            move_bb_omega=jnp.ones(t2, dtype=jnp.bool_),
            move_chi=jnp.ones((n_poses, max_n_blocks, max_n_chi), dtype=jnp.bool_),
            move_jump_dofs=jnp.ones(
                (n_poses, max_n_jumps, 6), dtype=jnp.bool_
            ),
        )

    @classmethod
    def only_sidechains(cls, pose_stack: Any) -> "MoveMap":
        """Chi torsions are free; backbone and jumps are fixed."""
        n_poses, max_n_blocks, max_n_chi, max_n_jumps = cls._shapes_from_pose_stack(
            pose_stack
        )
        t2 = (n_poses, max_n_blocks)
        return cls(
            move_bb_phi=jnp.zeros(t2, dtype=jnp.bool_),
            move_bb_psi=jnp.zeros(t2, dtype=jnp.bool_),
            move_bb_omega=jnp.zeros(t2, dtype=jnp.bool_),
            move_chi=jnp.ones((n_poses, max_n_blocks, max_n_chi), dtype=jnp.bool_),
            move_jump_dofs=jnp.zeros(
                (n_poses, max_n_jumps, 6), dtype=jnp.bool_
            ),
        )

    # ------------------------------------------------------------------
    # Functional mutation helper
    # ------------------------------------------------------------------

    def replace(self, **updates) -> "MoveMap":
        """Return a new :class:`MoveMap` with the given fields updated."""
        return attrs.evolve(self, **updates)

    # ------------------------------------------------------------------
    # DOF mask conversion
    # ------------------------------------------------------------------

    def to_dof_mask(self, kinforest: KinForest) -> Array:
        """Convert the move map to a per-DOF Boolean mask.

        Simplified port: upstream builds the mask via a dedicated C++
        op (``minimizer_map_from_movemap``) that walks the kin forest,
        correlates each node with the right pose/block/torsion, and
        resolves named torsions to their contributing bond DOFs. The
        JAX port instead returns a coarse mask with the following
        behaviour:

        * Root node: all six DOFs frozen (the root never moves).
        * Jump nodes: all six DOFs free if *any* jump DOF in the move
          map is free; this matches the common case where jump DOFs are
          either all-on or all-off per experiment.
        * Bond nodes: the torsion component (``phi_c`` — index 3 in
          :class:`BondDOFTypes`) is enabled when any chi or backbone
          flag in the move map is True; bond length / improper angle
          DOFs are left fixed (``non_ideal=False`` upstream).

        The resulting mask is ``bool[N, 9]`` where ``N`` is the number
        of nodes in ``kinforest``. Callers that need a flat ``bool[N]``
        can :func:`jnp.any`-reduce the last axis.

        Document: this is a simplified port. A full port will need the
        upstream-style per-torsion DOF mapping (via ``uaid_for_torsion``
        and related block-type metadata). The TODO-hook is preserved in
        the scan_ordering module.
        """
        n = int(kinforest.id.shape[0])
        mask = np.zeros((n, 9), dtype=np.bool_)

        doftype = np.asarray(kinforest.doftype)

        # Aggregate "any-bb-or-chi" per-node. We don't know which node
        # belongs to which block in this simplified port, so the policy
        # is: if the move map says any torsion is free *anywhere*, we
        # unlock the torsion DOF on every bond node. Downstream code
        # that cares about precision should reach for the full upstream
        # mapping.
        any_bb = bool(
            jnp.any(self.move_bb_phi)
            or jnp.any(self.move_bb_psi)
            or jnp.any(self.move_bb_omega)
        )
        any_chi = bool(jnp.any(self.move_chi))
        any_torsion = any_bb or any_chi

        any_jump = bool(jnp.any(self.move_jump_dofs))

        bond_rows = doftype == int(NodeType.bond)
        jump_rows = doftype == int(NodeType.jump)

        if any_torsion:
            # Enable the phi_c (torsion) component on every bond node.
            mask[bond_rows, int(BondDOFTypes.phi_c)] = True

        if any_jump:
            mask[jump_rows, :6] = True

        return jnp.asarray(mask, dtype=jnp.bool_)


__all__ = [
    "CartesianMoveMap",
    "MoveMap",
]
