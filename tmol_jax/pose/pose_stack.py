"""PoseStack — a batch (stack) of molecular systems, ported to JAX.

Upstream reference: ``tmol/pose/pose_stack.py`` (torch TensorGroup).

This JAX port keeps the same public surface but:

* Stores arrays as ``jax.Array`` rather than ``torch.Tensor``.
* Is immutable (``frozen=True``) — JAX pytree semantics require it, and
  upstream's "you may write to ``coords``" contract is replaced by the
  functional "produce a new ``PoseStack`` with updated coords" pattern.
* Registers itself as a pytree via the shared ``pytree_dataclass`` helper
  in :mod:`tmol_jax.types`, so ``jit``, ``vmap``, ``grad``, and ``scan``
  flow through it transparently.

We drop the 64-bit companion fields by default (``*_64``) because JAX
prefers a single dtype for gather/scatter ops and we can cast on demand.
We still surface ``block_coord_offset64`` for upstream-compatibility and
construct it lazily from ``block_coord_offset``.

Fields:
    packed_block_types: reference to the PackedBlockTypes palette.
    coords: ``float32[n_poses, max_n_atoms_per_pose, 3]``.
    block_coord_offset: ``int32[n_poses, max_n_blocks]`` — atom start
        offset per block within each pose's coords slab.
    block_coord_offset64: ``int64`` copy (useful for some gather ops).
    inter_residue_connections: ``int32[n_poses, max_n_blocks,
        max_n_conn, 2]`` — (other_block, other_conn) pairs; sentinel -1
        means "no connection".
    inter_block_bondsep: ``int32[n_poses, max_n_blocks, max_n_blocks,
        max_n_conn, max_n_conn]`` — chemical-bond path distance between
        every pair of inter-residue connection points.
    block_type_ind: ``int32[n_poses, max_n_blocks]`` — index into
        ``packed_block_types`` for each block (sentinel -1).
    pdb_info: optional PDBInfo for PDB/mmCIF round-trips.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import as_jnp, as_np, pytree_dataclass


def block_coord_offset_from_counts(
    block_type_ind: jax.Array,
    packed_block_types: Any,
) -> jax.Array:
    """Compute ``block_coord_offset`` from block-type indices.

    For each pose, the offset of block ``b`` is the sum of atom counts
    for blocks ``0..b-1``. Padding blocks (``block_type_ind < 0``)
    contribute zero atoms.

    Returns ``int32[n_poses, max_n_blocks]``.
    """

    bti = as_jnp(block_type_ind, dtype=jnp.int32)
    # Per-block-type atom count, gathered by block_type_ind. Sentinel
    # -1 positions should contribute 0; we clamp to 0 for the gather
    # and then zero them out with a mask.
    n_atoms_per_type = as_jnp(packed_block_types.n_atoms, dtype=jnp.int32)
    safe_idx = jnp.where(bti >= 0, bti, 0)
    per_block_counts = n_atoms_per_type[safe_idx]
    per_block_counts = jnp.where(bti >= 0, per_block_counts, 0)

    # Exclusive cumulative sum along the blocks axis gives start offsets.
    cumsum = jnp.cumsum(per_block_counts, axis=1)
    offsets = cumsum - per_block_counts
    return offsets.astype(jnp.int32)


@pytree_dataclass
class PoseStack:
    """A batch (stack) of molecular systems — see module docstring."""

    packed_block_types: Any  # PackedBlockTypes; aux (non-array) leaf.
    coords: jax.Array  # float32[n_poses, max_n_atoms_per_pose, 3]
    block_coord_offset: jax.Array  # int32[n_poses, max_n_blocks]
    block_coord_offset64: jax.Array  # int64[n_poses, max_n_blocks]
    inter_residue_connections: jax.Array  # int32[P, B, C, 2]
    inter_block_bondsep: jax.Array  # int32[P, B, B, C, C]
    block_type_ind: jax.Array  # int32[n_poses, max_n_blocks]
    pdb_info: Optional[Any] = None  # PDBInfo; aux.

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def n_poses(self) -> int:
        return int(self.block_type_ind.shape[0])

    @property
    def max_n_blocks(self) -> int:
        return int(self.block_type_ind.shape[1])

    @property
    def max_n_atoms_per_pose(self) -> int:
        """Largest number of atoms packed into any pose's coord slab."""
        return int(self.coords.shape[1])

    # Upstream alias
    @property
    def max_n_pose_atoms(self) -> int:
        return self.max_n_atoms_per_pose

    @property
    def max_n_atoms_per_block(self) -> int:
        """Maximum atom count over all block types in the palette."""
        return int(self.packed_block_types.max_n_atoms)

    # Upstream aliases
    @property
    def max_n_atoms(self) -> int:
        return self.max_n_atoms_per_block

    @property
    def max_n_block_atoms(self) -> int:
        return self.max_n_atoms_per_block

    @property
    def n_res_per_pose(self) -> jax.Array:
        """Per-pose count of non-padding blocks (``block_type_ind >= 0``)."""
        return jnp.sum(self.block_type_ind >= 0, axis=1).astype(jnp.int32)

    @property
    def device(self):
        """Placeholder device — JAX doesn't track device per-pytree.

        Returns the device of a trivial on-backend array so downstream
        code that expects a device handle (e.g. for building new
        arrays) keeps working.
        """
        return jnp.asarray(0).device

    @property
    def expanded_coords(self) -> jax.Array:
        """4D-ish view of coords. Upstream pads with zeros; here we
        return the flat ``[n_poses, max_n_atoms_per_pose, 3]`` tensor
        directly since JAX doesn't benefit from an in-place expand."""
        return self.coords

    # ---------------------------------------------------------------
    # Slicing / indexing
    # ---------------------------------------------------------------

    def coords_for_block(self, pose: int, block: int) -> jax.Array:
        """Return the atoms of a specific ``(pose, block)`` pair.

        Uses ``jax.lax.dynamic_slice`` so this remains tracer-friendly
        under ``jit`` when ``pose`` / ``block`` are concrete Python ints
        (the common case).
        """

        bti = int(self.block_type_ind[pose, block])
        if bti < 0:
            # Padding block — empty slice.
            return jnp.empty((0, 3), dtype=self.coords.dtype)
        n_atoms = int(self.packed_block_types.n_atoms[bti])
        start = int(self.block_coord_offset[pose, block])
        return jax.lax.dynamic_slice(
            self.coords[pose], (start, 0), (n_atoms, 3)
        )

    def __len__(self) -> int:
        return self.n_poses

    def __getitem__(self, i) -> "PoseStack":
        """Slice along the pose axis. Single-index returns a 1-pose stack."""

        if isinstance(i, (int, np.integer)):
            sl = slice(int(i), int(i) + 1)
        elif isinstance(i, slice):
            sl = i
        else:  # fancy indexing — delegate to jnp
            sl = i

        def _take(x):
            return x[sl]

        return PoseStack(
            packed_block_types=self.packed_block_types,
            coords=_take(self.coords),
            block_coord_offset=_take(self.block_coord_offset),
            block_coord_offset64=_take(self.block_coord_offset64),
            inter_residue_connections=_take(self.inter_residue_connections),
            inter_block_bondsep=_take(self.inter_block_bondsep),
            block_type_ind=_take(self.block_type_ind),
            pdb_info=self.pdb_info,
        )

    # ---------------------------------------------------------------
    # Copy semantics
    # ---------------------------------------------------------------

    def to(self, device) -> "PoseStack":
        """No-op in JAX — device placement is tracer-driven, not
        attached to pytrees. Returns ``self`` unchanged so upstream
        call sites don't break."""
        return self

    def clone(self) -> "PoseStack":
        """JAX arrays are immutable — ``clone`` returns ``self``.

        Downstream mutation patterns should produce a new PoseStack
        via :meth:`replace` / functional updates instead.
        """
        return self

    def replace(self, **updates) -> "PoseStack":
        """Functional field update helper."""
        current = {
            "packed_block_types": self.packed_block_types,
            "coords": self.coords,
            "block_coord_offset": self.block_coord_offset,
            "block_coord_offset64": self.block_coord_offset64,
            "inter_residue_connections": self.inter_residue_connections,
            "inter_block_bondsep": self.inter_block_bondsep,
            "block_type_ind": self.block_type_ind,
            "pdb_info": self.pdb_info,
        }
        current.update(updates)
        return PoseStack(**current)

    # ---------------------------------------------------------------
    # Block helpers (mirror the upstream Python-level conveniences)
    # ---------------------------------------------------------------

    def is_real_block(self, pose_ind: int, block_ind: int) -> bool:
        """True if ``(pose_ind, block_ind)`` is a real block, not padding."""
        return bool(self.block_type_ind[pose_ind, block_ind] >= 0)

    def block_type(self, pose_ind: int, block_ind: int):
        """Look up the ``RefinedResidueType``-equivalent block type.

        Delegates to ``packed_block_types.active_block_types``.
        ``is_real_block`` must be True.
        """
        return self.packed_block_types.active_block_types[
            int(self.block_type_ind[pose_ind, block_ind])
        ]

    # ---------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------

    @classmethod
    def from_components(
        cls,
        packed_block_types: Any,
        coords: Any,
        block_type_ind: Any,
        inter_residue_connections: Any,
        inter_block_bondsep: Any,
        pdb_info: Optional[Any] = None,
    ) -> "PoseStack":
        """Build a ``PoseStack`` from its raw components.

        ``block_coord_offset`` is computed from per-block atom counts
        via :func:`block_coord_offset_from_counts`.
        """

        coords_j = as_jnp(coords, dtype=jnp.float32)
        bti = as_jnp(block_type_ind, dtype=jnp.int32)
        irc = as_jnp(inter_residue_connections, dtype=jnp.int32)
        ibs = as_jnp(inter_block_bondsep, dtype=jnp.int32)

        bco = block_coord_offset_from_counts(bti, packed_block_types)
        bco64 = bco.astype(jnp.int64)

        return cls(
            packed_block_types=packed_block_types,
            coords=coords_j,
            block_coord_offset=bco,
            block_coord_offset64=bco64,
            inter_residue_connections=irc,
            inter_block_bondsep=ibs,
            block_type_ind=bti,
            pdb_info=pdb_info,
        )


__all__ = [
    "PoseStack",
    "block_coord_offset_from_counts",
]
