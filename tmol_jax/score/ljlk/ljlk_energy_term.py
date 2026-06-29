"""LJ/LK energy term implementation — three score types from a single traversal.

Glues the per-pair potentials in :mod:`.potentials` to the full PoseStack
coord tensor. The hot path builds an ``[n_poses, N, N]`` distance matrix,
gathers per-atom LJ/LK parameters via ``LJTypeParams.gather``, and evaluates
``lj_atr_rep`` + ``lk_isotropic_pair`` across the upper triangle. All ops
are pure JAX (jit/vmap/grad-safe).

The inter-atomic bonded-path-length matrix is computed exactly via
:func:`tmol_jax.score.common.count_pair.pairwise_bonded_path_length`,
which threads through inter-residue connection atoms using
``pose_stack.inter_block_bondsep``. This matches upstream's
``CountPair::inter_block_separation`` logic and is essential for
getting ``fa_ljrep`` right on residues connected by peptide bonds (the
coarse "every cross-residue pair has weight 1" approximation inflates
``fa_ljrep`` by orders of magnitude).
"""

from __future__ import annotations

from typing import Any, Optional

import attrs
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial as _partial

from tmol_jax.score.common.count_pair import (
    pairwise_bonded_path_length,
    scatter_per_atom,
)


@_partial(jax.jit, static_argnames=("max_n_atoms_per_pose",))
def _ljlk_pose_palette_jit(
    block_type_ind: jax.Array,  # [P, B] int32
    block_coord_offset: jax.Array,  # [P, B] int32
    n_atoms_per_type: jax.Array,  # [T] int32
    lj_radius_table: jax.Array,
    lj_wdepth_table: jax.Array,
    is_donor_table: jax.Array,
    is_acceptor_table: jax.Array,
    is_hydroxyl_table: jax.Array,
    is_polarh_table: jax.Array,
    lk_volume_table: jax.Array,
    lk_dgfree_table: jax.Array,
    lk_lambda_table: jax.Array,
    max_n_atoms_per_pose: int,
) -> tuple:
    """Fused LJLK palette gather — one jit trace for all 9 per-atom
    scatters + the real-atom mask.

    Replaces 10 separate ``scatter_per_atom`` calls (each ~400us of JAX
    dispatch) with one ~100us cached trace. Bit-exact with the
    individual-call path.
    """
    from tmol_jax.score.common.count_pair import pose_atom_block_index

    block_for_atom, local_for_atom, atom_is_real = pose_atom_block_index(
        block_type_ind,
        block_coord_offset,
        n_atoms_per_type,
        max_n_atoms_per_pose,
    )
    P = block_type_ind.shape[0]
    safe_bti = jnp.where(block_type_ind >= 0, block_type_ind, 0)
    bti_of_atom = safe_bti[jnp.arange(P)[:, None], block_for_atom]  # [P, N]

    def _s_f(table, dtype=jnp.float32):
        return jnp.where(
            atom_is_real,
            table[bti_of_atom, local_for_atom],
            jnp.asarray(0, dtype=dtype),
        )

    def _s_b(table):
        return jnp.where(
            atom_is_real, table[bti_of_atom, local_for_atom], False
        )

    return (
        _s_f(lj_radius_table),
        _s_f(lj_wdepth_table),
        _s_b(is_donor_table),
        _s_b(is_acceptor_table),
        _s_b(is_hydroxyl_table),
        _s_b(is_polarh_table),
        _s_f(lk_volume_table),
        _s_f(lk_dgfree_table),
        _s_f(lk_lambda_table),
        atom_is_real,
    )
from tmol_jax.score.common.neighbor_list import (
    _take_2d,
    _take_3d,
    build_cell_list,
    gather_pair_matrix,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.ljlk.params import LJLKGlobal, LJTypeParams
from tmol_jax.score.ljlk.potentials import lj_atr_rep, lk_isotropic_pair
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass


@pytree_dataclass
class _LJLKState:
    """Per-pose-stack precomputed LJ/LK state."""

    # Per-atom LJ/LK parameters (packed to pose-atom layout):
    # Each is [n_poses, max_n_atoms_per_pose].
    lj_radius: jax.Array
    lj_wdepth: jax.Array
    is_donor: jax.Array
    is_acceptor: jax.Array
    is_hydroxyl: jax.Array
    is_polarh: jax.Array
    lk_volume: jax.Array
    lk_dgfree: jax.Array
    lk_lambda: jax.Array

    # Per-atom real mask: [n_poses, max_n_atoms_per_pose] bool.
    atom_is_real: jax.Array

    # Bonded-path-length matrix per pose: [n_poses, N, N] int32.
    # Values >= 5 mean "ignore count-pair" (weight = 1.0).
    bonded_path_length: jax.Array

    # Global params (scalar arrays).
    global_lj_hbond_dis: jax.Array
    global_lj_hbond_OH_donor_dis: jax.Array
    global_lj_hbond_hdis: jax.Array

    # ---- Optional cell-list state ---------------------------------- #
    # Populated iff the term was configured with ``use_neighbor_list=True``.
    # Shapes:
    #   neighbor_idx        : [P, N, max_nbr]  int32 (half-list; j > i)
    #   neighbor_mask       : [P, N, max_nbr]  bool
    #   bpl_neighbor        : [P, N, max_nbr]  int32 (bonded_path_length
    #                                                  compressed to nbr layout)
    # If the term is in dense mode, these arrays are None so the dense
    # path can skip the gather entirely (pytree_dataclass supports
    # ``None`` leaves as static metadata).
    neighbor_idx: Any = None
    neighbor_mask: Any = None
    bpl_neighbor: Any = None


def _build_pose_atom_type_indices(pose_stack: Any, pbt: Any) -> jax.Array:
    """Return [n_poses, max_n_atoms_per_pose] atom-type indices, -1 for padding.

    JIT-compiled pose scatter: pads with -1 for non-real atom slots.
    """
    atom_types = jnp.asarray(pbt.atom_types, dtype=jnp.int32)
    bti = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
    offsets = jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)
    n_atoms = jnp.asarray(pbt.n_atoms, dtype=jnp.int32)
    return scatter_per_atom(
        atom_types,
        bti,
        offsets,
        n_atoms,
        int(pose_stack.coords.shape[1]),
        default=-1,
    )


def _build_bonded_path_length(pose_stack: Any, pbt: Any) -> jax.Array:
    """Exact ``[n_poses, N, N]`` bonded-path-length matrix.

    Thin wrapper around
    :func:`tmol_jax.score.common.count_pair.pairwise_bonded_path_length`
    -- kept as a local helper so downstream callers that imported this
    symbol continue to work.
    """
    return pairwise_bonded_path_length(pose_stack, pbt)


def _annotate_packed_block_types_for_ljlk(pbt: Any, param_db: Any) -> None:
    """Cache LJLK per-atom-type palette-level parameter tables on ``pbt``.

    Only touches ``pbt`` once (idempotent). After this call ``pbt`` exposes::

        pbt.ljlk_lj_radius_by_bt       [n_types, max_n_atoms] float32
        pbt.ljlk_lj_wdepth_by_bt       [n_types, max_n_atoms] float32
        pbt.ljlk_is_donor_by_bt        [n_types, max_n_atoms] bool
        pbt.ljlk_is_acceptor_by_bt     [n_types, max_n_atoms] bool
        pbt.ljlk_is_hydroxyl_by_bt     [n_types, max_n_atoms] bool
        pbt.ljlk_is_polarh_by_bt       [n_types, max_n_atoms] bool
        pbt.ljlk_lk_volume_by_bt       [n_types, max_n_atoms] float32
        pbt.ljlk_lk_dgfree_by_bt       [n_types, max_n_atoms] float32
        pbt.ljlk_lk_lambda_by_bt       [n_types, max_n_atoms] float32

    These are per-BT tables (not per-pose) so they are a pure function of
    the PBT palette and can be cached across any number of PoseStacks
    that share the same palette.
    """
    if hasattr(pbt, "ljlk_lj_radius_by_bt"):
        return
    ljlk_db = param_db.scoring.ljlk
    type_params = LJTypeParams.from_database(ljlk_db, param_db.chemical)

    atom_types = np.asarray(pbt.atom_types, dtype=np.int32)  # [T, M]
    real = atom_types >= 0
    safe_idx = np.where(real, atom_types, 0)

    def _g_f(arr):
        return np.where(real, np.asarray(arr)[safe_idx], 0.0).astype(np.float32)

    def _g_b(arr):
        return np.where(real, np.asarray(arr)[safe_idx], False)

    setattr(pbt, "ljlk_lj_radius_by_bt", jnp.asarray(_g_f(type_params.lj_radius)))
    setattr(pbt, "ljlk_lj_wdepth_by_bt", jnp.asarray(_g_f(type_params.lj_wdepth)))
    setattr(pbt, "ljlk_is_donor_by_bt", jnp.asarray(_g_b(type_params.is_donor)))
    setattr(pbt, "ljlk_is_acceptor_by_bt", jnp.asarray(_g_b(type_params.is_acceptor)))
    setattr(pbt, "ljlk_is_hydroxyl_by_bt", jnp.asarray(_g_b(type_params.is_hydroxyl)))
    setattr(pbt, "ljlk_is_polarh_by_bt", jnp.asarray(_g_b(type_params.is_polarh)))
    setattr(pbt, "ljlk_lk_volume_by_bt", jnp.asarray(_g_f(type_params.lk_volume)))
    setattr(pbt, "ljlk_lk_dgfree_by_bt", jnp.asarray(_g_f(type_params.lk_dgfree)))
    setattr(pbt, "ljlk_lk_lambda_by_bt", jnp.asarray(_g_f(type_params.lk_lambda)))


class LJLKEnergyTerm(EnergyTerm):
    """beta_nov2016 LJ + LK term — provides ``fa_ljatr``, ``fa_ljrep``, ``fa_lk``.

    The :meth:`setup_for_pose_stack` step looks up LJ/LK parameters for every
    atom in every pose and precomputes an approximate bonded-path-length
    matrix. The :meth:`__call__` step runs entirely inside jit-able pure-JAX.

    Two evaluation paths:

    * Dense (default): materialise ``[P, N, N]`` distance / param /
      bonded-path-length tensors. Bit-exact parity with upstream,
      fast enough for typical proteins (≲3000 atoms).
    * Cell-list (``use_neighbor_list=True``): build a padded
      ``[P, N, max_nbr]`` neighbour list from a 3D spatial hash during
      setup, then sum only over those pairs inside the hot loop.
      Memory scales as ``O(N * k_max)`` rather than ``O(N^2)``.
      Intended for large systems (>~4000 atoms) where the dense
      tensors start to dominate GPU memory.

    The cell-list is built once at setup time with a cutoff of 6.0 A
    (the LJ / LK ``cpoly_dmax``). If coordinates drift significantly
    after minimisation, the caller should rebuild the term state.
    """

    def __init__(
        self,
        param_db: Any = None,
        use_neighbor_list: bool = False,
        # ponytail: matches LJLKGlobalParams.max_dis (= 6.0) post-upstream
        # commit 1d7e18c. Read from db.global_params.max_dis if jtmol ever
        # needs to track a non-default cutoff.
        neighbor_list_cutoff: float = 6.0,
    ) -> None:
        # The base EnergyTerm stores ``param_db`` for us; we also
        # remember the opt-in flag so setup can route to the cell-list
        # builder.
        super().__init__(param_db=param_db)
        self.use_neighbor_list = bool(use_neighbor_list)
        self.neighbor_list_cutoff = float(neighbor_list_cutoff)

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk]

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Optional[Any] = None
    ) -> _LJLKState:
        if pbt is None:
            pbt = pose_stack.packed_block_types

        # Idempotent palette-level caching — once per pbt. After this call
        # ``pbt`` exposes ``ljlk_*_by_bt`` tables keyed by
        # ``[block_type_ind, local_atom]``. All 9 per-atom scatters + the
        # atom_is_real mask fuse into a single jit trace via
        # :func:`_ljlk_pose_palette_jit`, dropping the setup from 3.4 ms
        # (one dispatch per table) to well under a ms.
        _annotate_packed_block_types_for_ljlk(pbt, self.param_db)
        global_params = LJLKGlobal.from_database(self.param_db.scoring.ljlk)

        bti = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
        offsets = jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)
        n_atoms_per_type = jnp.asarray(pbt.n_atoms, dtype=jnp.int32)
        A = int(pose_stack.coords.shape[1])

        (
            lj_radius,
            lj_wdepth,
            is_donor,
            is_acceptor,
            is_hydroxyl,
            is_polarh,
            lk_volume,
            lk_dgfree,
            lk_lambda,
            atom_is_real,
        ) = _ljlk_pose_palette_jit(
            bti,
            offsets,
            n_atoms_per_type,
            pbt.ljlk_lj_radius_by_bt,
            pbt.ljlk_lj_wdepth_by_bt,
            pbt.ljlk_is_donor_by_bt,
            pbt.ljlk_is_acceptor_by_bt,
            pbt.ljlk_is_hydroxyl_by_bt,
            pbt.ljlk_is_polarh_by_bt,
            pbt.ljlk_lk_volume_by_bt,
            pbt.ljlk_lk_dgfree_by_bt,
            pbt.ljlk_lk_lambda_by_bt,
            A,
        )

        bonded_path_length = _build_bonded_path_length(pose_stack, pbt)

        # Optional cell-list construction. We keep the state fields as
        # ``None`` for the dense path so the pytree has no extra leaves
        # (see ``_LJLKState``); for the fast path we build them here once,
        # using the (static) real-atom mask and the pose-stack coords.
        neighbor_idx = None
        neighbor_mask = None
        bpl_neighbor = None
        if self.use_neighbor_list:
            nbr_idx, _n_nbrs, nbr_mask = build_cell_list(
                pose_stack.coords,
                cutoff=self.neighbor_list_cutoff,
                atom_is_real=np.asarray(atom_is_real),
                half_list=True,
            )
            # Compress the [P, N, N] bonded_path_length matrix down to
            # the [P, N, max_nbr] layout used by the hot loop. This is
            # done once, outside jit, so we never materialise a full
            # dense int32 pair tensor inside the traced evaluation.
            bpl_neighbor = gather_pair_matrix(bonded_path_length, nbr_idx)
            neighbor_idx = nbr_idx
            neighbor_mask = nbr_mask

        return _LJLKState(
            lj_radius=lj_radius,
            lj_wdepth=lj_wdepth,
            is_donor=is_donor,
            is_acceptor=is_acceptor,
            is_hydroxyl=is_hydroxyl,
            is_polarh=is_polarh,
            lk_volume=lk_volume,
            lk_dgfree=lk_dgfree,
            lk_lambda=lk_lambda,
            atom_is_real=atom_is_real,
            bonded_path_length=bonded_path_length,
            global_lj_hbond_dis=global_params.lj_hbond_dis,
            global_lj_hbond_OH_donor_dis=global_params.lj_hbond_OH_donor_dis,
            global_lj_hbond_hdis=global_params.lj_hbond_hdis,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bpl_neighbor=bpl_neighbor,
        )

    def __call__(
        self,
        pose_stack: Any,
        term_state: _LJLKState,
        coords_override: Optional[jax.Array] = None,
    ) -> dict:
        coords = coords_override if coords_override is not None else pose_stack.coords
        # coords: [P, N, 3]
        if self.use_neighbor_list and term_state.neighbor_idx is not None:
            return self._call_neighbor_list(coords, term_state)

        # Build pairwise distance matrix [P, N, N].
        # NaN-safe distance computation: the gradient of ``jnp.linalg.norm``
        # is ``diff/|diff|`` which is ``0/0`` at the diagonal. Use the
        # double-where trick so sqrt sees only positive inputs and the
        # downstream branch gets a safe constant.
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1)
        safe = dist_sq > 1e-6
        dist = jnp.where(
            safe,
            jnp.sqrt(jnp.where(safe, dist_sq, 1.0)),
            0.1,
        )

        # Build per-pair parameter views using broadcasting:
        # i_params: [P, N, 1, ...]   j_params: [P, 1, N, ...]
        def _as_i(arr):
            return arr[:, :, None]

        def _as_j(arr):
            return arr[:, None, :]

        @attrs.define
        class _Side:
            lj_radius: jax.Array
            lj_wdepth: jax.Array
            is_donor: jax.Array
            is_acceptor: jax.Array
            is_hydroxyl: jax.Array
            is_polarh: jax.Array
            lk_volume: jax.Array
            lk_dgfree: jax.Array
            lk_lambda: jax.Array

        s = term_state
        i_params = _Side(
            lj_radius=_as_i(s.lj_radius),
            lj_wdepth=_as_i(s.lj_wdepth),
            is_donor=_as_i(s.is_donor),
            is_acceptor=_as_i(s.is_acceptor),
            is_hydroxyl=_as_i(s.is_hydroxyl),
            is_polarh=_as_i(s.is_polarh),
            lk_volume=_as_i(s.lk_volume),
            lk_dgfree=_as_i(s.lk_dgfree),
            lk_lambda=_as_i(s.lk_lambda),
        )
        j_params = _Side(
            lj_radius=_as_j(s.lj_radius),
            lj_wdepth=_as_j(s.lj_wdepth),
            is_donor=_as_j(s.is_donor),
            is_acceptor=_as_j(s.is_acceptor),
            is_hydroxyl=_as_j(s.is_hydroxyl),
            is_polarh=_as_j(s.is_polarh),
            lk_volume=_as_j(s.lk_volume),
            lk_dgfree=_as_j(s.lk_dgfree),
            lk_lambda=_as_j(s.lk_lambda),
        )

        class _G:
            lj_hbond_dis = s.global_lj_hbond_dis
            lj_hbond_OH_donor_dis = s.global_lj_hbond_OH_donor_dis
            lj_hbond_hdis = s.global_lj_hbond_hdis

        Vatr, Vrep = lj_atr_rep(dist, s.bonded_path_length, i_params, j_params, _G)
        Vlk = lk_isotropic_pair(
            dist, s.bonded_path_length, i_params, j_params, _G
        )

        # Mask: real-atom-pairs, upper triangle only (avoid double-count).
        real_pair = s.atom_is_real[:, :, None] & s.atom_is_real[:, None, :]
        iu = jnp.triu(jnp.ones(dist.shape[-2:], dtype=bool), k=1)[None]
        mask = real_pair & iu

        Vatr = jnp.where(mask, Vatr, 0.0).sum(axis=(-1, -2))
        Vrep = jnp.where(mask, Vrep, 0.0).sum(axis=(-1, -2))
        Vlk = jnp.where(mask, Vlk, 0.0).sum(axis=(-1, -2))

        return {
            ScoreType.fa_ljatr: Vatr,
            ScoreType.fa_ljrep: Vrep,
            ScoreType.fa_lk: Vlk,
        }

    # ------------------------------------------------------------------ #
    # Cell-list fast path                                                #
    # ------------------------------------------------------------------ #

    def _call_neighbor_list(
        self, coords: jax.Array, term_state: _LJLKState
    ) -> dict:
        """Evaluate LJ/LK using the precomputed half-list of neighbours.

        Never materialises the ``[P, N, N]`` distance / parameter /
        bonded-path-length tensors that the dense path builds. Instead
        we gather ``[P, N, max_nbr]`` slices of each per-atom quantity
        along the neighbour axis and evaluate the same
        ``lj_atr_rep`` / ``lk_isotropic_pair`` potentials on them.

        The cell list is a *half*-list (``j > i``) built once in
        :meth:`setup_for_pose_stack`, which means the downstream sum
        already avoids double-counting — mirrors the
        ``jnp.triu(..., k=1)`` mask used by the dense path.
        """
        s = term_state
        nbr_idx = s.neighbor_idx  # [P, N, max_nbr] int32
        nbr_mask = s.neighbor_mask  # [P, N, max_nbr] bool

        # Gather coords along the neighbour axis so we can form per-pair
        # distances without a [P, N, N] intermediate. [P, N, max_nbr, 3].
        coords_j = _take_3d(coords, nbr_idx)
        coords_i = coords[:, :, None, :]

        diff = coords_i - coords_j  # [P, N, max_nbr, 3]
        dist_sq = jnp.sum(diff * diff, axis=-1)  # [P, N, max_nbr]
        # NaN-safe sqrt (padding slots may have zero distance because
        # their nbr index is 0). The mask nulls them out of the sum.
        safe = dist_sq > 1e-6
        dist = jnp.where(
            safe,
            jnp.sqrt(jnp.where(safe, dist_sq, 1.0)),
            0.1,
        )

        # Gather per-atom parameters along the neighbour axis. ``_as_i``
        # broadcasts i-side params across the neighbour axis; ``_as_j``
        # picks the j-side parameter *at the neighbour index*. All
        # outputs end up shape [P, N, max_nbr].
        def _as_i(a):
            return a[:, :, None]

        def _as_j(a):
            return _take_2d(a, nbr_idx)

        @attrs.define
        class _Side:
            lj_radius: jax.Array
            lj_wdepth: jax.Array
            is_donor: jax.Array
            is_acceptor: jax.Array
            is_hydroxyl: jax.Array
            is_polarh: jax.Array
            lk_volume: jax.Array
            lk_dgfree: jax.Array
            lk_lambda: jax.Array

        i_params = _Side(
            lj_radius=_as_i(s.lj_radius),
            lj_wdepth=_as_i(s.lj_wdepth),
            is_donor=_as_i(s.is_donor),
            is_acceptor=_as_i(s.is_acceptor),
            is_hydroxyl=_as_i(s.is_hydroxyl),
            is_polarh=_as_i(s.is_polarh),
            lk_volume=_as_i(s.lk_volume),
            lk_dgfree=_as_i(s.lk_dgfree),
            lk_lambda=_as_i(s.lk_lambda),
        )
        j_params = _Side(
            lj_radius=_as_j(s.lj_radius),
            lj_wdepth=_as_j(s.lj_wdepth),
            is_donor=_as_j(s.is_donor),
            is_acceptor=_as_j(s.is_acceptor),
            is_hydroxyl=_as_j(s.is_hydroxyl),
            is_polarh=_as_j(s.is_polarh),
            lk_volume=_as_j(s.lk_volume),
            lk_dgfree=_as_j(s.lk_dgfree),
            lk_lambda=_as_j(s.lk_lambda),
        )

        class _G:
            lj_hbond_dis = s.global_lj_hbond_dis
            lj_hbond_OH_donor_dis = s.global_lj_hbond_OH_donor_dis
            lj_hbond_hdis = s.global_lj_hbond_hdis

        # ``bpl_neighbor`` was gathered host-side in setup so the
        # count-pair weight is already in the neighbour layout.
        Vatr, Vrep = lj_atr_rep(dist, s.bpl_neighbor, i_params, j_params, _G)
        Vlk = lk_isotropic_pair(dist, s.bpl_neighbor, i_params, j_params, _G)

        # Mask out padded neighbour slots and pair-where-either-atom-is-padding.
        # The cell-list builder already excluded padded atoms on the ``i`` side,
        # but we still zero padded ``i`` rows for safety against future callers
        # that build lists over all [P, N] rows regardless of realness.
        real_i = s.atom_is_real[:, :, None]
        mask = nbr_mask & real_i

        Vatr = jnp.where(mask, Vatr, 0.0).sum(axis=(-1, -2))
        Vrep = jnp.where(mask, Vrep, 0.0).sum(axis=(-1, -2))
        Vlk = jnp.where(mask, Vlk, 0.0).sum(axis=(-1, -2))

        return {
            ScoreType.fa_ljatr: Vatr,
            ScoreType.fa_ljrep: Vrep,
            ScoreType.fa_lk: Vlk,
        }


__all__ = ["LJLKEnergyTerm"]
