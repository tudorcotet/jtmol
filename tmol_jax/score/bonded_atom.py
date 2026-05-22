"""Bonded-atom helpers (pure JAX port).

Upstream reference: the small Python wrappers around
``tmol/chemical/bonded_atom.hh`` used throughout the scoring code --
principally :class:`tmol.score.bonded_atom.IndexedBonds` and the
``atom_is_bonded`` predicate.

We keep the interface minimal:

* :class:`IndexedBonds` is re-exported from
  :mod:`tmol_jax.score.bond_dependent_term` for parity with upstream
  import paths.
* :func:`atom_is_bonded` is a simple numpy predicate: given a
  :class:`PackedBlockTypes`, a block-type (residue) index, and two atom
  indices within that block, it returns ``True`` iff the two atoms are
  connected by one (intra-residue) chemical bond.

This is deliberately host-side / numpy: the callers we have all use
this predicate during setup (building exclusion tables, detecting
geminal pairs, etc.). If a traced version is ever needed it is easy to
express as ``pbt.intrares_path_distance[r, a1, a2] == 1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tmol_jax.score.bond_dependent_term import IndexedBonds


# --------------------------------------------------------------------- #
# atom_is_bonded                                                         #
# --------------------------------------------------------------------- #


def atom_is_bonded(
    pbt: Any, residue_ind: int, atom1: int, atom2: int
) -> bool:
    """Return ``True`` iff ``atom1`` and ``atom2`` share an intra-residue
    bond on block type ``residue_ind``.

    Parameters
    ----------
    pbt
        A :class:`tmol_jax.pose.packed_block_types.PackedBlockTypes` (or
        anything with the same duck-typed ``intrares_path_distance`` /
        ``bond_indices`` fields).
    residue_ind
        Index into ``pbt.active_block_types``.
    atom1, atom2
        Atom indices within that block type.

    Notes
    -----
    * Self-loops return ``False`` -- an atom is not bonded to itself.
    * Out-of-range atom indices return ``False``.
    * We use the already-computed ``intrares_path_distance == 1`` test
      when it is available; otherwise we fall back to scanning the
      ``bond_indices`` table.
    """
    if residue_ind < 0:
        return False
    if atom1 == atom2:
        return False

    n_atoms_arr = getattr(pbt, "n_atoms", None)
    if n_atoms_arr is not None:
        try:
            n = int(np.asarray(n_atoms_arr)[residue_ind])
        except Exception:
            n = None
        if n is not None:
            if not (0 <= atom1 < n and 0 <= atom2 < n):
                return False

    pd = getattr(pbt, "intrares_path_distance", None)
    if pd is not None:
        arr = np.asarray(pd)
        # Path distance of 1 => directly bonded.
        return bool(arr[residue_ind, atom1, atom2] == 1)

    # Fallback: scan bond_indices for this block type.
    bi = np.asarray(pbt.bond_indices)
    n_bonds = np.asarray(getattr(pbt, "n_bonds"))[residue_ind]
    slab = bi[residue_ind, : int(n_bonds), :]
    hits = (slab[:, 0] == atom1) & (slab[:, 1] == atom2)
    return bool(hits.any())


# --------------------------------------------------------------------- #
# Convenience: bulk predicate                                            #
# --------------------------------------------------------------------- #


def atoms_are_bonded(
    pbt: Any, residue_inds: np.ndarray, atom1: np.ndarray, atom2: np.ndarray
) -> np.ndarray:
    """Vectorised version of :func:`atom_is_bonded` over aligned
    ``(residue_ind, atom1, atom2)`` arrays.

    Returns a boolean numpy array of the same shape as the inputs.
    """
    residue_inds = np.asarray(residue_inds, dtype=np.int64)
    atom1 = np.asarray(atom1, dtype=np.int64)
    atom2 = np.asarray(atom2, dtype=np.int64)

    pd = getattr(pbt, "intrares_path_distance", None)
    if pd is None:
        out = np.zeros(residue_inds.shape, dtype=bool)
        it = np.nditer(residue_inds, flags=["multi_index"])
        while not it.finished:
            mi = it.multi_index
            out[mi] = atom_is_bonded(
                pbt,
                int(residue_inds[mi]),
                int(atom1[mi]),
                int(atom2[mi]),
            )
            it.iternext()
        return out

    arr = np.asarray(pd)

    # valid indices -> path distance == 1. Out-of-range indices come out
    # as False via a safe-index + explicit mask.
    n_types = arr.shape[0]
    safe_res = np.clip(residue_inds, 0, n_types - 1)
    safe_a1 = np.clip(atom1, 0, arr.shape[1] - 1)
    safe_a2 = np.clip(atom2, 0, arr.shape[2] - 1)
    looked_up = arr[safe_res, safe_a1, safe_a2]

    valid = (
        (residue_inds >= 0)
        & (residue_inds < n_types)
        & (atom1 >= 0)
        & (atom1 < arr.shape[1])
        & (atom2 >= 0)
        & (atom2 < arr.shape[2])
        & (atom1 != atom2)
    )
    return np.where(valid, looked_up == 1, False)


__all__ = ["IndexedBonds", "atom_is_bonded", "atoms_are_bonded"]
