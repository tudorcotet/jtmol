"""Bond-dependent term mixin (pure JAX port).

Upstream reference: :mod:`tmol.score.bond_dependent_term`.

Upstream's :class:`BondDependentTerm` is a mixin that walks every
:class:`RefinedResidueType` / :class:`PackedBlockTypes` it encounters
and attaches:

* ``intrares_indexed_bonds`` -- an :class:`IndexedBonds` view per
  block type,
* ``bond_separation``        -- a ``[n_types, max_n_atoms, max_n_atoms]``
  per-block-type path-distance table (clipped at ``MAX_SIG_BOND_SEPARATION``),
* ``all_bonds`` / ``n_all_bonds`` / ``atom_all_bond_ranges`` --
  the packed combined (intra + inter-residue) bond table,
* and on the pose-stack side, ``min_block_bondsep`` -- the min over
  connection pairs of ``inter_block_bondsep``.

In the JAX port the per-block-type bond tables are already baked into
:class:`tmol_jax.chemical.restypes.RefinedResidueType`
(``bond_indices``, ``path_distance``, ``all_bonds``,
``atom_all_bond_ranges``) and the intra-residue path-distance cube is
already on :class:`PackedBlockTypes` as ``intrares_path_distance``.
This mixin does the small amount of residual work:

1. Concatenates each restype's ``all_bonds`` / ``atom_all_bond_ranges``
   into packed PBT-level arrays, once, caching them on the PBT.
2. Reduces ``pose_stack.inter_block_bondsep`` -> ``min_block_bondsep``
   (if the pose stack exposes ``inter_block_bondsep``).

Base class for cartbonded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import attrs
import jax.numpy as jnp
import numpy as np

from tmol_jax.chemical.constants import MAX_SIG_BOND_SEPARATION
from tmol_jax.score.energy_term import EnergyTerm


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


@attrs.define(slots=True, frozen=True)
class IndexedBonds:
    """JAX-friendly re-expression of upstream's :class:`IndexedBonds`.

    ``bonds`` is ``int32[n_stacks, n_bonds, 2]`` and ``bond_spans`` is
    ``int32[n_stacks, n_atoms, 2]``. We pack the single-block-type case
    as ``n_stacks=1`` so this class can be used interchangeably at the
    per-restype level.
    """

    bonds: np.ndarray
    bond_spans: np.ndarray

    @classmethod
    def from_bonds(
        cls, src_bonds: np.ndarray, minlength: Optional[int] = None
    ) -> "IndexedBonds":
        """Build from a ``[n_bonds, 3]`` ``(stack, i, j)`` edge list.

        This is a numpy-only port of the upstream classmethod; the
        output is plain numpy because it is only consumed during setup,
        not inside jit/grad.
        """
        if src_bonds.size == 0:
            nstacks = 1
            ml = minlength if minlength is not None else 0
            return cls(
                bonds=np.zeros((nstacks, 0, 2), dtype=np.int32),
                bond_spans=np.zeros((nstacks, ml, 2), dtype=np.int32),
            )

        uniq_bonds, _ = np.unique(src_bonds, axis=0, return_index=True)
        nstacks = int(uniq_bonds[:, 0].max()) + 1
        if not minlength:
            minlength = int(uniq_bonds[:, 1:].max()) + 1

        bond_counts = np.bincount(uniq_bonds[:, 0], minlength=nstacks).reshape(
            nstacks, 1
        )
        max_nbonds = int(bond_counts.max())

        bonds = np.full((nstacks, max_nbonds, 2), -1, dtype=np.int32)
        counts = np.arange(max_nbonds).reshape(1, max_nbonds)
        lowinds = counts < bond_counts
        nz = np.nonzero(lowinds)
        bonds[nz[0], nz[1], :] = uniq_bonds[:, 1:]

        bond_spans = np.full((nstacks, minlength, 2), -1, dtype=np.int32)

        max_atom_offsets = np.arange(nstacks).reshape(nstacks, 1) * (minlength + 1)
        first_at_shifted = np.copy(bonds[:, :, 0])
        first_at_shifted[first_at_shifted < 0] = (minlength + 1) * nstacks - 1
        first_at_shifted = (first_at_shifted + max_atom_offsets).ravel()
        bincounts = np.bincount(
            first_at_shifted, minlength=((minlength + 1) * nstacks)
        ).reshape(nstacks, -1)[:, :minlength]
        bonds_cumsum = np.cumsum(bincounts, axis=1)
        bond_spans[:, 0, 0] = 0
        bond_spans[:, 1:minlength, 0] = bonds_cumsum[:, :-1]
        bond_spans[:, :minlength, 1] = bonds_cumsum

        return cls(bonds=bonds, bond_spans=bond_spans)


@attrs.define(slots=True)
class BondAnnotation:
    """Cache of bond-dependent arrays attached to a PackedBlockTypes.

    Fields mirror the attributes upstream mutates onto
    ``PackedBlockTypes`` in
    :meth:`BondDependentTerm.setup_packed_block_types`.

    Fields
    ------
    bond_separation
        ``jnp.int32[n_types, max_n_atoms, max_n_atoms]`` intra-residue
        path-distance table, clipped at ``MAX_SIG_BOND_SEPARATION``. Alias
        of :attr:`PackedBlockTypes.intrares_path_distance`.
    n_all_bonds
        ``jnp.int32[n_types]`` number of entries per block type in
        :attr:`all_bonds`.
    all_bonds
        ``jnp.int32[n_types, max_n_all_bonds, 3]`` packed combined
        (intra + inter-residue) bond table; each row is
        ``(atom_ind, other_atom_ind_or_-1, conn_id_or_-1)``.
    atom_all_bond_ranges
        ``jnp.int32[n_types, max_n_atoms, 2]`` start / stop indices into
        :attr:`all_bonds` for each atom.
    """

    bond_separation: jnp.ndarray
    n_all_bonds: jnp.ndarray
    all_bonds: jnp.ndarray
    atom_all_bond_ranges: jnp.ndarray


# --------------------------------------------------------------------- #
# BondDependentTerm                                                      #
# --------------------------------------------------------------------- #


class BondDependentTerm(EnergyTerm):
    """Mixin base class for terms that need per-bond metadata.

    Subclasses should invoke ``self._ensure_bond_annotation(pbt)`` from
    their :meth:`setup_for_pose_stack`. The cached :class:`BondAnnotation`
    is stashed on the PBT and reused by all other bond-dependent terms.

    For pose-stack-level bond separations, use
    :meth:`min_block_bondsep`, which reduces
    ``pose_stack.inter_block_bondsep`` over the two connection axes.
    """

    # ------------------------------------------------------------------ #
    # Block-type -> per-block annotation                                 #
    # ------------------------------------------------------------------ #
    def setup_block_type(self, restype: Any) -> IndexedBonds:
        """Ensure a restype has an :class:`IndexedBonds` cached under
        ``intrares_indexed_bonds`` and return it.

        This mirrors the semantics of upstream's block-type setup: first
        call builds the bonds view from the directed bond-indices table;
        subsequent calls reuse the cached value.
        """
        cached = getattr(restype, "intrares_indexed_bonds", None)
        if cached is not None:
            return cached

        bond_indices = restype.bond_indices  # directed [n_bonds, 2]
        n_atoms = restype.n_atoms
        # Prepend a zero "stack" column so the single-residue case fits
        # the IndexedBonds interface.
        src = np.zeros((bond_indices.shape[0], 3), dtype=np.int32)
        src[:, 1:] = bond_indices.astype(np.int32)
        ib = IndexedBonds.from_bonds(src, minlength=n_atoms)
        try:
            setattr(restype, "intrares_indexed_bonds", ib)
        except AttributeError:
            pass
        return ib

    # ------------------------------------------------------------------ #
    # PBT-level bond annotation                                          #
    # ------------------------------------------------------------------ #
    def _ensure_bond_annotation(self, pbt: Any) -> BondAnnotation:
        """Return the cached :class:`BondAnnotation` for ``pbt``."""
        cached = getattr(pbt, "_bond_annotation", None)
        if cached is not None:
            return cached

        # -------------------------------------------------------------- #
        # bond_separation                                                 #
        # -------------------------------------------------------------- #
        # PBT already has ``intrares_path_distance`` -- clip to
        # ``MAX_SIG_BOND_SEPARATION`` to match the upstream semantics
        # (upstream fills in MAX_SIG_BOND_SEPARATION for padding slots
        # *and* for distances above the cutoff; our PBT fills padding
        # with 0, so we fix that up here).
        n_types = pbt.n_types
        max_n_atoms = pbt.max_n_atoms
        pd_np = np.asarray(pbt.intrares_path_distance).astype(np.int32)
        n_atoms_np = np.asarray(pbt.n_atoms).astype(np.int32)

        bond_separation_np = np.full(
            (n_types, max_n_atoms, max_n_atoms),
            MAX_SIG_BOND_SEPARATION,
            dtype=np.int32,
        )
        for i in range(n_types):
            k = int(n_atoms_np[i])
            bond_separation_np[i, :k, :k] = np.clip(
                pd_np[i, :k, :k], 0, MAX_SIG_BOND_SEPARATION
            )

        # -------------------------------------------------------------- #
        # all_bonds / atom_all_bond_ranges                                #
        # -------------------------------------------------------------- #
        active = pbt.active_block_types
        # Ensure each block type has a cached all_bonds table (it always
        # does for RefinedResidueType, but be defensive).
        for bt in active:
            self.setup_block_type(bt)

        max_n_all_bonds = max(
            (bt.all_bonds.shape[0] for bt in active), default=0
        )
        n_all_bonds_np = np.full((n_types,), -1, dtype=np.int32)
        all_bonds_np = np.full(
            (n_types, max_n_all_bonds, 3), -1, dtype=np.int32
        )
        atom_all_bond_ranges_np = np.full(
            (n_types, max_n_atoms, 2), -1, dtype=np.int32
        )

        for i, bt in enumerate(active):
            n_bonds_i = bt.all_bonds.shape[0]
            n_all_bonds_np[i] = n_bonds_i
            if n_bonds_i > 0:
                all_bonds_np[i, :n_bonds_i, :] = bt.all_bonds.astype(np.int32)
            n_atoms_i = bt.n_atoms
            if n_atoms_i > 0:
                atom_all_bond_ranges_np[i, :n_atoms_i, :] = (
                    bt.atom_all_bond_ranges.astype(np.int32)
                )

        ann = BondAnnotation(
            bond_separation=jnp.asarray(bond_separation_np),
            n_all_bonds=jnp.asarray(n_all_bonds_np),
            all_bonds=jnp.asarray(all_bonds_np),
            atom_all_bond_ranges=jnp.asarray(atom_all_bond_ranges_np),
        )
        try:
            setattr(pbt, "_bond_annotation", ann)
        except AttributeError:
            pass
        return ann

    # ------------------------------------------------------------------ #
    # Pose-stack-level reduction                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def min_block_bondsep(pose_stack: Any) -> Optional[jnp.ndarray]:
        """Return ``min_block_bondsep`` for a pose stack (or None).

        Computed as ``inter_block_bondsep.min(axis=-1).min(axis=-1)``
        (minimum over both connection axes), matching upstream's
        :meth:`BondDependentTerm.setup_poses` reduction.

        Returns ``None`` when the pose stack has no
        ``inter_block_bondsep`` field set.
        """
        bondsep = getattr(pose_stack, "inter_block_bondsep", None)
        if bondsep is None:
            return None
        x = jnp.asarray(bondsep)
        x = jnp.min(x, axis=-1)
        x = jnp.min(x, axis=-1)
        return x

    # ------------------------------------------------------------------ #
    # Default pose-stack setup                                           #
    # ------------------------------------------------------------------ #
    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> Dict[str, Any]:
        """Return a term-state fragment with bond annotations.

        Emits ``{"bond_annotation", "min_block_bondsep"}``, where the
        latter is ``None`` if the pose stack has no inter-block bondsep.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types
        return {
            "bond_annotation": self._ensure_bond_annotation(pbt),
            "min_block_bondsep": BondDependentTerm.min_block_bondsep(pose_stack),
        }


__all__ = ["BondAnnotation", "BondDependentTerm", "IndexedBonds"]
