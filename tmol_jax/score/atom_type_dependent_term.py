"""Atom-type dependent term mixin (pure JAX port).

Upstream reference: :mod:`tmol.score.atom_type_dependent_term`.

Upstream's :class:`AtomTypeDependentTerm` is a mixin that augments every
:class:`RefinedResidueType` / :class:`PackedBlockTypes` it encounters
with three annotations:

* ``atom_types``           -- int32 index into ``chem_db.atom_types``
  for every atom,
* ``n_heavy_atoms`` +
  ``heavy_atom_inds``      -- the compacted "heavy only" atom subset,
* ``atom_unique_ids`` /
  ``atom_wildcard_ids``    -- per-atom integer tags used by atom-name
  dependent potentials (e.g. rama variant lookups).

In the JAX port, the first two annotations are already baked into the
:class:`tmol_jax.pose.packed_block_types.PackedBlockTypes` pytree
(``atom_types``, ``atom_is_hydrogen``, ``atom_is_real``). This file
provides a small mixin that:

1. Surfaces those arrays under the same names upstream uses, so term
   code can be shared / ported without renames.
2. Computes the heavy-atom compacted index arrays once per
   ``PackedBlockTypes`` (upstream's ``n_heavy_atoms`` / ``heavy_atom_inds``).
3. Builds the unique-id / wildcard-id tag arrays for terms that care
   (e.g. the Dunbrack/rama setup uses them for variant lookups).

Instances do not own any JAX arrays; they stash a small attrs record on
the ``PackedBlockTypes`` the first time ``setup_for_packed_block_types``
is called and reuse it thereafter.

This is a base class for LJ/LK, hbond, elec energy terms.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional

import attrs
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.chemical_database import ChemicalDatabaseScore
from tmol_jax.score.energy_term import EnergyTerm


# --------------------------------------------------------------------- #
# Per-PBT annotation                                                     #
# --------------------------------------------------------------------- #


@attrs.define(slots=True)
class AtomTypeAnnotation:
    """Cache of atom-type-dependent arrays attached to a PackedBlockTypes.

    All arrays are plain :mod:`jax.numpy` / :mod:`numpy`. We keep
    bool/int arrays on the host (numpy) when they are used only for
    masking during setup of other terms, and upload them to jax.numpy
    when we expect to consume them inside jitted code.

    Fields
    ------
    atom_type_ind
        Alias of ``pbt.atom_types`` — included so downstream code can
        refer to a single canonical name.
    atom_is_heavyatom
        ``[n_types, max_n_atoms]`` bool, the logical negation of
        ``pbt.atom_is_hydrogen`` masked by ``pbt.atom_is_real``.
    n_heavy_atoms
        ``[n_types]`` int32 count of heavy atoms per block type.
    heavy_atom_inds
        ``[n_types, max_n_heavy]`` int32 packed heavy-atom indices per
        block type (pad with ``-1``).
    atom_unique_ids
        ``[n_types, max_n_atoms]`` int32 per-atom unique string tag
        (``UNIQUE_ID:<basename>:<atomname>``). ``-1`` for padding slots.
    atom_wildcard_ids
        ``[n_types, max_n_atoms]`` int32 per-atom wildcard tag
        (``WILDCARD_ID:<atomname>``). ``-1`` for padding slots.
    atom_unique_id_index
        OrderedDict mapping the raw string tag to its integer id.
    """

    atom_type_ind: jnp.ndarray
    atom_is_heavyatom: jnp.ndarray
    n_heavy_atoms: jnp.ndarray
    heavy_atom_inds: jnp.ndarray
    atom_unique_ids: jnp.ndarray
    atom_wildcard_ids: jnp.ndarray
    atom_unique_id_index: Mapping[str, int]


# --------------------------------------------------------------------- #
# AtomTypeDependentTerm                                                  #
# --------------------------------------------------------------------- #


class AtomTypeDependentTerm(EnergyTerm):
    """Mixin base class for terms that need per-atom atom-type metadata.

    Subclasses should invoke ``self._ensure_atom_type_annotation(pbt)``
    from their :meth:`setup_for_pose_stack` before they try to read
    atom-type indices; the cached :class:`AtomTypeAnnotation` is stashed
    on the PBT (and the chemical database score record) so repeated
    setups are cheap.

    The mixin also exposes :attr:`chem_db_score` which wraps the
    underlying :class:`PatchedChemicalDatabase` with score-specific
    index helpers (see :mod:`tmol_jax.score.chemical_database`).
    """

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        self._chem_db_score: Optional[ChemicalDatabaseScore] = None

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #
    @property
    def chem_db_score(self) -> ChemicalDatabaseScore:
        """The score-flavoured chem-db wrapper for ``self.param_db``.

        Built lazily on first access.
        """
        if self._chem_db_score is None:
            chem_db = getattr(self.param_db, "chemical", None) or getattr(
                self.param_db, "chem_db", None
            )
            if chem_db is None:
                raise AttributeError(
                    "param_db has neither 'chemical' nor 'chem_db' attribute; "
                    "cannot build ChemicalDatabaseScore."
                )
            self._chem_db_score = ChemicalDatabaseScore.from_chem_db(chem_db)
        return self._chem_db_score

    def get_atom_unique_id_name(self, block_name: str, atom_name: str) -> str:
        """Build a per-block-type per-atom unique tag.

        Mirrors upstream's :meth:`AtomTypeDependentTerm.get_atom_unique_id_name`.
        """
        return f"UNIQUE_ID:{block_name}:{atom_name}"

    def get_atom_wildcard_id_name(self, atom_name: str) -> str:
        """Build a per-atom-name wildcard tag.

        Mirrors upstream's :meth:`AtomTypeDependentTerm.get_atom_wildcard_id_name`.
        """
        return f"WILDCARD_ID:{atom_name}"

    # ------------------------------------------------------------------ #
    # Setup helpers                                                      #
    # ------------------------------------------------------------------ #
    def _ensure_atom_type_annotation(self, pbt: Any) -> AtomTypeAnnotation:
        """Return the cached :class:`AtomTypeAnnotation` for ``pbt``.

        The annotation is lazily built on first call and stashed on
        ``pbt._atom_type_annotation``. Safe to call from multiple terms
        in any order -- the second and subsequent calls are O(1).

        ``PackedBlockTypes`` is a pytree dataclass so arbitrary
        attribute assignment is supported (no slots) -- see
        :func:`tmol_jax.types.pytree_dataclass`.
        """
        cached = getattr(pbt, "_atom_type_annotation", None)
        if cached is not None:
            return cached

        atom_type_ind = pbt.atom_types  # [n_types, max_n_atoms] int32
        atom_is_real = pbt.atom_is_real
        atom_is_hydrogen = pbt.atom_is_hydrogen

        # Heavy = real AND not hydrogen. Keep as jax.Array so downstream
        # masks can participate in traced code if necessary.
        atom_is_heavyatom = jnp.logical_and(
            atom_is_real, jnp.logical_not(atom_is_hydrogen)
        )

        # Build compact heavy-atom index arrays on the host. These are
        # static metadata -- the packed indices themselves do not need
        # to flow through JAX.
        atom_is_heavyatom_np = np.asarray(atom_is_heavyatom)
        n_heavy_list: List[int] = [
            int(atom_is_heavyatom_np[i].sum()) for i in range(pbt.n_types)
        ]
        max_n_heavy = max(n_heavy_list) if n_heavy_list else 0
        heavy_atom_inds_np = np.full(
            (pbt.n_types, max_n_heavy), -1, dtype=np.int32
        )
        for i in range(pbt.n_types):
            inds = np.nonzero(atom_is_heavyatom_np[i])[0]
            heavy_atom_inds_np[i, : inds.size] = inds.astype(np.int32)

        # Unique/wildcard tag arrays. We build a global string->int
        # index table sized across all active block types. Padding
        # positions get -1.
        atom_unique_id_index: OrderedDict[str, int] = OrderedDict()
        atom_unique_ids_np = np.full(
            (pbt.n_types, pbt.max_n_atoms), -1, dtype=np.int32
        )
        atom_wildcard_ids_np = np.full(
            (pbt.n_types, pbt.max_n_atoms), -1, dtype=np.int32
        )

        for i, bt in enumerate(pbt.active_block_types):
            base_name = getattr(bt, "base_name", bt.name)
            for j, atom in enumerate(bt.atoms):
                uid = self.get_atom_unique_id_name(base_name, atom.name)
                wid = self.get_atom_wildcard_id_name(atom.name)
                if uid not in atom_unique_id_index:
                    atom_unique_id_index[uid] = len(atom_unique_id_index)
                if wid not in atom_unique_id_index:
                    atom_unique_id_index[wid] = len(atom_unique_id_index)
                atom_unique_ids_np[i, j] = atom_unique_id_index[uid]
                atom_wildcard_ids_np[i, j] = atom_unique_id_index[wid]

        ann = AtomTypeAnnotation(
            atom_type_ind=atom_type_ind,
            atom_is_heavyatom=atom_is_heavyatom,
            n_heavy_atoms=jnp.asarray(np.asarray(n_heavy_list, dtype=np.int32)),
            heavy_atom_inds=jnp.asarray(heavy_atom_inds_np),
            atom_unique_ids=jnp.asarray(atom_unique_ids_np),
            atom_wildcard_ids=jnp.asarray(atom_wildcard_ids_np),
            atom_unique_id_index=atom_unique_id_index,
        )

        try:
            setattr(pbt, "_atom_type_annotation", ann)
        except AttributeError:
            # Slotted object with no dict -- not something we produce
            # from ``packed_block_types.py``, but be defensive anyway.
            pass
        return ann

    def setup_block_type(self, restype: Any) -> Dict[str, Any]:
        """Resolve atom-type indices for a single RefinedResidueType.

        This mirrors the per-block-type setup phase in upstream
        :meth:`AtomTypeDependentTerm.setup_block_type` but *returns* the
        resolved data rather than mutating the restype. Callers can use
        the returned dict directly or stash it on the restype (idempotent
        via ``_atom_type_setup_block_type``).

        Returns
        -------
        dict with keys

          * ``atom_types``        -- ``int32[n_atoms]`` chem-db atom-type idx
          * ``heavy_atom_inds``   -- ``int32[n_heavy]`` atom positions of heavies
          * ``atom_unique_ids``   -- ``list[str]`` per-atom unique tag
          * ``atom_wildcard_ids`` -- ``list[str]`` per-atom wildcard tag
        """
        cached = getattr(restype, "_atom_type_setup_block_type", None)
        if cached is not None:
            return cached

        score_db = self.chem_db_score
        atom_types = np.asarray(
            [score_db.atom_type_index_for(a.atom_type) for a in restype.atoms],
            dtype=np.int32,
        )
        is_hydrogen = score_db.atom_type_is_hydrogen
        heavy_mask = ~is_hydrogen[np.clip(atom_types, 0, len(is_hydrogen) - 1)]
        # Types not in the chem-db score table (atom_types == -1) are
        # treated as non-heavy -- they are almost certainly a bug upstream,
        # and matching them to heavy_atom_inds would skew the atom counts.
        heavy_mask = heavy_mask & (atom_types >= 0)
        heavy_atom_inds = np.nonzero(heavy_mask)[0].astype(np.int32)

        base_name = getattr(restype, "base_name", restype.name)
        atom_unique_ids = [
            self.get_atom_unique_id_name(base_name, a.name) for a in restype.atoms
        ]
        atom_wildcard_ids = [
            self.get_atom_wildcard_id_name(a.name) for a in restype.atoms
        ]

        out = {
            "atom_types": atom_types,
            "heavy_atom_inds": heavy_atom_inds,
            "atom_unique_ids": atom_unique_ids,
            "atom_wildcard_ids": atom_wildcard_ids,
        }
        try:
            setattr(restype, "_atom_type_setup_block_type", out)
        except AttributeError:
            pass
        return out

    # ------------------------------------------------------------------ #
    # Default pose-stack setup                                           #
    # ------------------------------------------------------------------ #
    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> Dict[str, Any]:
        """Return an ``{"atom_type_annotation": AtomTypeAnnotation}``
        dict derived from the pose-stack's PBT.

        Subclasses typically call this and merge the returned dict into
        their own term-state dict; consuming terms can then read the
        annotation via ``term_state["atom_type_annotation"]``.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types
        annotation = self._ensure_atom_type_annotation(pbt)
        return {"atom_type_annotation": annotation}


__all__ = ["AtomTypeAnnotation", "AtomTypeDependentTerm"]
