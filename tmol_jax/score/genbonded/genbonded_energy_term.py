"""Generic-bonded torsion energy term (pure JAX).

Upstream reference: :mod:`tmol.score.genbonded.genbonded_energy_term`.

Provides one score type — :attr:`ScoreType.gen_torsions` — which sums:

1. **Proper torsions**: every 4-atom subgraph ``(a, b, c, d)`` reachable
   in the intra-block bond graph, scored with the bare-cosine 4-period
   form (see :func:`tmol_jax.score.genbonded.potentials.gen_proper_E`).
2. **Improper torsions**: for every atom with exactly 3 neighbors, one
   harmonic ``k * (theta - delta)^2`` penalty.

Parameter lookup is by atom *chemical type* (e.g. ``CS``, ``C*``, ``X``)
via the :class:`GenBondedDatabase` hierarchy walker. Bond-type bin defaults
to ``(SINGLE, not-in-ring)`` because jtmol does not currently track per-bond
type or ring membership in the chemical database. This is fine for the
protein-only port: in the upstream YAML, 239/263 torsion entries use the
``~`` wildcard bond, so bond-type accuracy hardly matters in practice.
The integration-scan-noted skips (DOUBLE/TRIPLE/AROMATIC ring variants)
are deliberate; document with a ``ponytail:`` comment below.

**On 1ubq this term evaluates to 0.0** — matching upstream's gold value
(see ``tmol/tests/score/test_score_function.py``). The reason is that the
upstream ``genbonded.yaml`` only carries an ``atoms:`` hierarchy for
ligand-style generic types (``CS``, ``CD``, ``Nad``, ...), not for the
Rosetta protein types (``Nbb``, ``CAbb``, ``CObb``, ``Hapo``, ...) that
fill canonical protein residues. Every protein atom's hierarchy is the
trivial ``[<self>]`` and no DB entry matches — every torsion is a hash
miss, contributing zero. The term is implemented for ligands and any
future YAML extension; it costs ~nothing on protein-only inputs.

Inter-residue **proper torsions** (4-atom subgraphs spanning a peptide
bond) are emitted alongside the intra-block ones. Enumeration mirrors
the cartbonded ``_inter_residue_subgraphs_for_pair`` helper: walk
``atom_paths_from_conn`` on both residues, join ``reversed(path_A) +
path_B``, keep size-4 tuples. The only swap is the param-lookup function
-- atom-type tuple via :meth:`GenBondedDatabase.find_torsion_params`
instead of atom-name tuple. Impropers are intra-residue by definition
(an improper is rooted at a single atom with exactly three neighbours,
all on the same residue) so no inter-block impropers are emitted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.common.geom import dihedral_angle
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.score.genbonded.potentials import gen_improper_E, gen_proper_E
from tmol_jax.score.cartbonded.cartbonded_energy_term import (
    _CON_PATH_INDICES,
    _bond_atom_map,
    _find_intra_subgraphs,
    _pad_atoms,
    _pad_params,
    _path_atoms,
)
from tmol_jax.types import pytree_dataclass


# --------------------------------------------------------------------------- #
# Rosetta `calculate_offset`: zero the energy floor when DB offset is 0       #
# --------------------------------------------------------------------------- #


def _calculate_offset(k1: float, k2: float, k3: float, k4: float) -> float:
    """Match Rosetta's ``GenTorsionParams::calculate_offset``.

    If two or more of ``(k1, k2, k3)`` are non-zero, scan the energy at
    5-degree intervals and return the negation of the minimum so the
    potential floors at zero. Otherwise return 0 (the energy already has
    a single well at zero by construction).

    A non-zero database ``offset`` overrides this entirely; callers gate
    on ``abs(db_offset) < 1e-6`` before invoking.
    """
    nk = sum(1 for k in (k1, k2, k3) if abs(k) > 1e-6)
    if nk <= 1:
        return 0.0
    t = np.arange(-180.0, 180.0, 5.0) * (np.pi / 180.0)
    e = (
        k1 * (1.0 + np.cos(t))
        + k2 * (1.0 + np.cos(2.0 * t))
        + k3 * (1.0 + np.cos(3.0 * t))
        + k4 * (1.0 + np.cos(4.0 * t))
    )
    for k in (k1, k2, k3, k4):
        if k < 0.0:
            e = e + (-2.0 * k)
    return float(-e.min())


# --------------------------------------------------------------------------- #
# Per-pose-stack precomputed state                                            #
# --------------------------------------------------------------------------- #


@pytree_dataclass
class GenBondedTermState:
    """Padded per-block-type tables, materialized to per-pose.

    Shape legend: ``P`` = n_poses, ``B`` = max_n_blocks,
    ``T`` / ``I`` = per-category pad widths.
    """

    # Proper torsions — 4 atoms + (k1, k2, k3, k4, offset).
    torsion_atoms: jnp.ndarray  # int32[P, B, T, 4]
    torsion_params: jnp.ndarray  # float32[P, B, T, 5]
    torsion_mask: jnp.ndarray  # bool[P, B, T]

    # Impropers — 4 atoms + (k, delta).
    improper_atoms: jnp.ndarray  # int32[P, B, I, 4]
    improper_params: jnp.ndarray  # float32[P, B, I, 2]
    improper_mask: jnp.ndarray  # bool[P, B, I]

    # Inter-residue proper torsions spanning a peptide-bond connection.
    # Shape ``[P, IT, ...]`` -- one entry per pose per cross-block
    # 4-atom subgraph. Emitted only for ``block_i < block_j`` to avoid
    # double counting, mirroring cartbonded's inter-block tables.
    inter_torsion_atoms: jnp.ndarray  # int32[P, IT, 4]
    inter_torsion_params: jnp.ndarray  # float32[P, IT, 5] (k1,k2,k3,k4,offset)
    inter_torsion_mask: jnp.ndarray  # bool[P, IT]


# --------------------------------------------------------------------------- #
# Per-block-type setup                                                        #
# --------------------------------------------------------------------------- #


def _find_improper_subgraphs(
    adj: Dict[int, List[int]],
) -> List[Tuple[int, int, int, int]]:
    """Atoms with exactly 3 neighbors → ``(center, n1, n2, n3)`` sorted.

    Mirrors upstream ``find_improper_subgraphs``.
    """
    out: List[Tuple[int, int, int, int]] = []
    for center, nb in adj.items():
        if len(nb) == 3:
            n1, n2, n3 = sorted(nb)
            out.append((center, n1, n2, n3))
    return out


def _subgraphs_for_block_type(
    rt: Any,
    gen_db: Any,
) -> Tuple[
    List[Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float]]],
    List[Tuple[Tuple[int, int, int, int], Tuple[float, float]]],
]:
    """Enumerate + resolve intra-block proper torsions and impropers.

    Returns ``(propers, impropers)`` where each entry is a
    ``(local_atom_tuple, param_tuple)`` pair. Subgraphs without a DB
    match are dropped (mirrors upstream hash-miss behavior).
    """
    n_atoms = rt.n_atoms
    bond_indices = rt.bond_indices
    if bond_indices is None:
        bond_indices = np.zeros((0, 2), dtype=np.int32)

    adj = _bond_atom_map(bond_indices, n_atoms)

    # Proper torsions: reuse cartbonded's enumeration (a<d canonical,
    # a!=c, b!=d). Pass an empty improper_roots set; we enumerate
    # impropers separately by neighbor count.
    _l, _a, torsions_sg, _imp = _find_intra_subgraphs(adj, [])
    impropers_sg = _find_improper_subgraphs(adj)

    atom_types = [a.atom_type for a in rt.atoms]

    propers_out: List[
        Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float]]
    ] = []
    for atoms in torsions_sg:
        t1, t2, t3, t4 = (atom_types[a] for a in atoms)
        # ponytail: bond_type defaults to (SINGLE, not-in-ring); jtmol's
        # chemical DB has no per-bond order/ring metadata. 239/263 YAML
        # entries use the '~' wildcard bond so this hardly matters.
        # Upgrade path: extend Atom / bond schema to carry BondType +
        # is_ring and pass through here.
        entry = gen_db.find_torsion_params(t1, t2, t3, t4)
        if entry is None:
            continue
        off = entry.offset
        if abs(off) < 1e-6:
            off = _calculate_offset(entry.k1, entry.k2, entry.k3, entry.k4)
        propers_out.append(
            (atoms, (entry.k1, entry.k2, entry.k3, entry.k4, off))
        )

    impropers_out: List[Tuple[Tuple[int, int, int, int], Tuple[float, float]]] = []
    for atoms in impropers_sg:
        tc, t1, t2, t3 = (atom_types[a] for a in atoms)
        entry = gen_db.find_improper_params(tc, t1, t2, t3)
        if entry is None:
            continue
        impropers_out.append((atoms, (entry.k, entry.delta)))

    return propers_out, impropers_out


def _inter_residue_subgraphs_for_pair(
    rt_A: Any,
    rt_B: Any,
    conn_A: int,
    conn_B: int,
    gen_db: Any,
) -> List[
    Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, float, float, float, float]]
]:
    """Enumerate cross-block proper-torsion subgraphs across one connection.

    Mirrors :func:`tmol_jax.score.cartbonded.cartbonded_energy_term._inter_residue_subgraphs_for_pair`
    but specialized to genbonded: only size-4 (proper torsion) subgraphs,
    no length / angle / improper variants, and parameters resolved by
    atom *type* via :meth:`GenBondedDatabase.find_torsion_params` (which
    already walks the hierarchy + bond-bin multiplicity scoring).

    Returns a list of ``(local_atoms, is_on_B, params)`` triples.
    ``is_on_B[i]`` is ``True`` if ``local_atoms[i]`` indexes into residue
    B (else residue A); the caller adds the matching ``block_coord_offset``
    when materializing to global atom indices.

    Subgraphs without a matching DB entry are dropped (hash-miss).
    """
    paths_A = rt_A.atom_paths_from_conn[conn_A]
    paths_B = rt_B.atom_paths_from_conn[conn_B]

    atom_types_A = [a.atom_type for a in rt_A.atoms]
    atom_types_B = [a.atom_type for a in rt_B.atoms]

    out: List[
        Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, float, float, float, float]]
    ] = []
    seen: set = set()

    for cat, pa_i, pb_i in _CON_PATH_INDICES:
        if cat != 2:  # only torsions (cat 0 = length, 1 = angle)
            continue
        path_A = _path_atoms(paths_A[pa_i])
        path_B = _path_atoms(paths_B[pb_i])
        if not path_A or not path_B:
            continue
        rev_A = path_A[::-1]
        atoms = rev_A + path_B
        if len(atoms) != 4:
            continue
        is_on_B = tuple([False] * len(rev_A) + [True] * len(path_B))
        key = (atoms, is_on_B)
        if key in seen:
            continue
        seen.add(key)
        types = tuple(
            atom_types_B[a] if b else atom_types_A[a]
            for a, b in zip(atoms, is_on_B)
        )
        # ponytail: default bond_type=SINGLE, is_ring=False -- same
        # rationale as the intra-block enumeration above.
        entry = gen_db.find_torsion_params(*types)
        if entry is None:
            continue
        off = entry.offset
        if abs(off) < 1e-6:
            off = _calculate_offset(entry.k1, entry.k2, entry.k3, entry.k4)
        out.append(
            (atoms, is_on_B, (entry.k1, entry.k2, entry.k3, entry.k4, off))
        )

    return out


# --------------------------------------------------------------------------- #
# Energy term                                                                 #
# --------------------------------------------------------------------------- #


class GenBondedEnergyTerm(EnergyTerm):
    """Generic-bonded torsions + impropers (one score type: gen_torsions).

    Parameters
    ----------
    param_db
        A :class:`tmol_jax.database.ParameterDatabase` exposing
        ``scoring.genbonded`` as a
        :class:`~tmol_jax.database.scoring.genbonded.GenBondedDatabase`.
    """

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        self.gen_database = param_db.scoring.genbonded
        # Cache keyed by id(active_block_types) — same idiom as cartbonded.
        self._bt_cache: Dict[int, Dict[str, Any]] = {}

    def score_types_provided(self) -> List[ScoreType]:
        return [ScoreType.gen_torsions]

    # --------------------------------------------------------------- #
    def _build_pbt_tables(self, pbt: Any) -> Dict[str, Any]:
        key = id(pbt.active_block_types)
        cached = self._bt_cache.get(key)
        if cached is not None:
            return cached

        per_type_torsion_atoms: List[List[Tuple[int, int, int, int]]] = []
        per_type_torsion_params: List[List[Tuple[float, ...]]] = []
        per_type_improper_atoms: List[List[Tuple[int, int, int, int]]] = []
        per_type_improper_params: List[List[Tuple[float, ...]]] = []

        for rt in pbt.active_block_types:
            propers, impropers = _subgraphs_for_block_type(rt, self.gen_database)
            per_type_torsion_atoms.append([a for a, _ in propers])
            per_type_torsion_params.append([p for _, p in propers])
            per_type_improper_atoms.append([a for a, _ in impropers])
            per_type_improper_params.append([p for _, p in impropers])

        torsion_atoms, torsion_mask, _ = _pad_atoms(per_type_torsion_atoms, 4)
        torsion_params = _pad_params(per_type_torsion_params, 5)
        improper_atoms, improper_mask, _ = _pad_atoms(per_type_improper_atoms, 4)
        improper_params = _pad_params(per_type_improper_params, 2)

        # Inter-residue proper-torsion subgraphs per (bt_A, conn_A, bt_B,
        # conn_B). Empty pairs are omitted so the per-pose materialization
        # in ``setup_for_pose_stack`` can short-circuit on a dict miss.
        inter_torsions_by_pair: Dict[
            Tuple[int, int, int, int],
            List[
                Tuple[
                    Tuple[int, ...],
                    Tuple[bool, ...],
                    Tuple[float, float, float, float, float],
                ]
            ],
        ] = {}
        for bt_A, rt_A in enumerate(pbt.active_block_types):
            n_conn_A = len(rt_A.connections)
            for bt_B, rt_B in enumerate(pbt.active_block_types):
                n_conn_B = len(rt_B.connections)
                for conn_A in range(n_conn_A):
                    for conn_B in range(n_conn_B):
                        torsions = _inter_residue_subgraphs_for_pair(
                            rt_A, rt_B, conn_A, conn_B, self.gen_database
                        )
                        if torsions:
                            inter_torsions_by_pair[
                                (bt_A, conn_A, bt_B, conn_B)
                            ] = torsions

        tables = {
            "torsion_atoms": torsion_atoms,
            "torsion_params": torsion_params,
            "torsion_mask": torsion_mask,
            "improper_atoms": improper_atoms,
            "improper_params": improper_params,
            "improper_mask": improper_mask,
            "inter_torsions_by_pair": inter_torsions_by_pair,
        }
        self._bt_cache[key] = tables
        return tables

    # --------------------------------------------------------------- #
    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> GenBondedTermState:
        if pbt is None:
            pbt = pose_stack.packed_block_types
        tables = self._build_pbt_tables(pbt)

        block_type_ind = np.asarray(pose_stack.block_type_ind)  # [P, B]
        block_coord_offset = np.asarray(pose_stack.block_coord_offset)  # [P, B]
        coords_np = np.asarray(pose_stack.coords)  # [P, A, 3]

        P, _B = block_type_ind.shape

        # Phantom-atom mask: same convention as cartbonded — exclude
        # atoms whose coord norm is < 0.5 Å (sentinel for missing
        # backbone H on N-terminal residues etc.) or non-finite.
        finite = np.all(np.isfinite(coords_np), axis=-1)
        atom_norm = np.linalg.norm(coords_np, axis=-1)
        atom_present = finite & (atom_norm > 0.5)

        def _gather(atoms_bt: np.ndarray, params_bt: np.ndarray, mask_bt: np.ndarray):
            safe_bt = np.where(block_type_ind >= 0, block_type_ind, 0)
            real_block = block_type_ind >= 0
            atoms_pb = atoms_bt[safe_bt]
            params_pb = params_bt[safe_bt]
            mask_pb = mask_bt[safe_bt] & real_block[:, :, None]
            offsets = block_coord_offset[:, :, None, None]
            valid = atoms_pb >= 0
            atoms_global = np.where(valid, atoms_pb + offsets, -1).astype(np.int32)
            safe_global = np.where(atoms_global >= 0, atoms_global, 0)
            pose_idx = np.arange(P)[:, None, None, None]
            present = atom_present[pose_idx, safe_global]
            mask_pb = mask_pb & np.all(present, axis=-1)
            return atoms_global, params_pb.astype(np.float32), mask_pb.astype(bool)

        t_atoms, t_params, t_mask = _gather(
            tables["torsion_atoms"], tables["torsion_params"], tables["torsion_mask"]
        )
        i_atoms, i_params, i_mask = _gather(
            tables["improper_atoms"],
            tables["improper_params"],
            tables["improper_mask"],
        )

        # ---------- Inter-residue materialization ------------------- #
        irc = np.asarray(pose_stack.inter_residue_connections)  # [P, B, C, 2]
        inter_by_pair = tables["inter_torsions_by_pair"]

        inter_T_atoms_per_pose: List[List[Tuple[int, int, int, int]]] = []
        inter_T_params_per_pose: List[List[Tuple[float, ...]]] = []

        B = block_type_ind.shape[1]
        for pose in range(P):
            T_atoms: List[Tuple[int, int, int, int]] = []
            T_params: List[Tuple[float, ...]] = []
            for block_i in range(B):
                bt_i = int(block_type_ind[pose, block_i])
                if bt_i < 0:
                    continue
                n_conn_i = irc.shape[2]
                for conn_i in range(n_conn_i):
                    block_j = int(irc[pose, block_i, conn_i, 0])
                    conn_j = int(irc[pose, block_i, conn_i, 1])
                    if block_j < 0 or block_i >= block_j:
                        continue
                    bt_j = int(block_type_ind[pose, block_j])
                    if bt_j < 0:
                        continue
                    torsions_hit = inter_by_pair.get((bt_i, conn_i, bt_j, conn_j))
                    if not torsions_hit:
                        continue
                    off_i = int(block_coord_offset[pose, block_i])
                    off_j = int(block_coord_offset[pose, block_j])
                    for atoms, is_on_B, params in torsions_hit:
                        g = tuple(
                            (a + off_j) if b else (a + off_i)
                            for a, b in zip(atoms, is_on_B)
                        )
                        T_atoms.append(g)
                        T_params.append(params)
            inter_T_atoms_per_pose.append(T_atoms)
            inter_T_params_per_pose.append(T_params)

        max_len = max((len(x) for x in inter_T_atoms_per_pose), default=0)
        max_len = max(max_len, 1)
        inter_T_atoms = np.full((P, max_len, 4), -1, dtype=np.int32)
        inter_T_params = np.zeros((P, max_len, 5), dtype=np.float32)
        inter_T_mask = np.zeros((P, max_len), dtype=bool)
        for p, (atoms_l, params_l) in enumerate(
            zip(inter_T_atoms_per_pose, inter_T_params_per_pose)
        ):
            for j, (a, pa) in enumerate(zip(atoms_l, params_l)):
                inter_T_atoms[p, j, :] = a
                inter_T_params[p, j, :] = pa
                inter_T_mask[p, j] = True

        # Phantom-atom screen (same convention as the intra path).
        if inter_T_atoms.size > 0:
            safe = np.where(inter_T_atoms >= 0, inter_T_atoms, 0)
            pose_idx = np.arange(P)[:, None, None]
            present = atom_present[pose_idx, safe]
            inter_T_mask = inter_T_mask & np.all(present, axis=-1)

        return GenBondedTermState(
            torsion_atoms=jnp.asarray(t_atoms),
            torsion_params=jnp.asarray(t_params),
            torsion_mask=jnp.asarray(t_mask),
            improper_atoms=jnp.asarray(i_atoms),
            improper_params=jnp.asarray(i_params),
            improper_mask=jnp.asarray(i_mask),
            inter_torsion_atoms=jnp.asarray(inter_T_atoms),
            inter_torsion_params=jnp.asarray(inter_T_params),
            inter_torsion_mask=jnp.asarray(inter_T_mask),
        )

    # --------------------------------------------------------------- #
    def __call__(
        self,
        pose_stack: Any,
        term_state: GenBondedTermState,
        coords_override: Optional[jax.Array] = None,
    ) -> Dict[ScoreType, jax.Array]:
        coords = pose_stack.coords if coords_override is None else coords_override

        def _gather_xyz(global_atoms: jax.Array) -> jax.Array:
            safe = jnp.where(global_atoms >= 0, global_atoms, 0)
            P, B, K, k = safe.shape
            flat = safe.reshape(P, B * K * k)
            idx = flat[:, :, None].repeat(3, axis=-1)
            gathered = jnp.take_along_axis(coords, idx, axis=1)
            return gathered.reshape(P, B, K, k, 3)

        # Proper torsions ------------------------------------------- #
        ta = term_state.torsion_atoms
        txyz = _gather_xyz(ta)
        t_theta = dihedral_angle(
            txyz[..., 0, :], txyz[..., 1, :], txyz[..., 2, :], txyz[..., 3, :]
        )
        tp = term_state.torsion_params
        t_E = gen_proper_E(
            t_theta,
            tp[..., 0], tp[..., 1], tp[..., 2], tp[..., 3], tp[..., 4],
        )
        t_E = jnp.where(term_state.torsion_mask, t_E, 0.0)
        proper_sum = jnp.sum(t_E, axis=(1, 2))  # [P]

        # Impropers ------------------------------------------------- #
        ia = term_state.improper_atoms
        ixyz = _gather_xyz(ia)
        i_theta = dihedral_angle(
            ixyz[..., 0, :], ixyz[..., 1, :], ixyz[..., 2, :], ixyz[..., 3, :]
        )
        ip = term_state.improper_params
        i_E = gen_improper_E(i_theta, ip[..., 0], ip[..., 1])
        i_E = jnp.where(term_state.improper_mask, i_E, 0.0)
        imp_sum = jnp.sum(i_E, axis=(1, 2))  # [P]

        # Inter-residue proper torsions ----------------------------- #
        ita = term_state.inter_torsion_atoms  # [P, E, 4]
        inter_sum = jnp.zeros((coords.shape[0],), dtype=proper_sum.dtype)
        if ita.shape[1] > 0:
            safe = jnp.where(ita >= 0, ita, 0)
            P_, E_, k_ = safe.shape
            flat = safe.reshape(P_, E_ * k_)
            idx = flat[:, :, None].repeat(3, axis=-1)
            ixyz = jnp.take_along_axis(coords, idx, axis=1).reshape(
                P_, E_, k_, 3
            )
            it_theta = dihedral_angle(
                ixyz[..., 0, :], ixyz[..., 1, :],
                ixyz[..., 2, :], ixyz[..., 3, :],
            )
            itp = term_state.inter_torsion_params
            it_E = gen_proper_E(
                it_theta,
                itp[..., 0], itp[..., 1], itp[..., 2], itp[..., 3], itp[..., 4],
            )
            it_E = jnp.where(term_state.inter_torsion_mask, it_E, 0.0)
            inter_sum = jnp.sum(it_E, axis=1)

        return {ScoreType.gen_torsions: proper_sum + imp_sum + inter_sum}


__all__ = ["GenBondedEnergyTerm", "GenBondedTermState"]
