"""Cartesian-bonded energy term (pure JAX).

Upstream reference: :mod:`tmol.score.cartbonded.cartbonded_energy_term`.

The Rosetta ``cart_bonded`` family imposes harmonic (lengths, angles)
and periodic (torsions, impropers, hxltorsions) penalties on bond
lengths, bond angles, proper torsions, improper torsions, and hydroxyl
torsions. Parameters come from ``cartbonded.yaml`` via
:class:`tmol_jax.database.scoring.cartbonded.CartBondedDatabase`.

This JAX port restricts itself to *intra-residue* subgraphs: every bond,
angle, torsion, improper, and hxl-torsion whose atoms all live in the
same block. Inter-residue geometry (the peptide-bond ``C-N`` length,
flanking :math:`\\phi` / :math:`\\psi` torsions, ...) is omitted. The
upstream CUDA kernel handles that by walking ``inter_residue_connections``
and ``atom_paths_from_conn``; porting that full machinery is left as
future work. The omission matters for absolute energies but not for
local minimization of side chains or rigid-backbone CartBonded scoring.

Both impropers and proper torsions share the same sum-of-three-cosines
form in upstream ``cbtorsion_V_dV``; only the score-type bucket differs.
We store impropers with the same 6-wide ``(k1, k2, k3, phi1, phi2, phi3)``
parameter vector and reuse ``hxltorsion_E``.

Setup proceeds in two phases:

1. **Per-block-type enumeration.** For each active residue type we walk
   its bond graph (``RefinedResidueType.bond_indices``) to enumerate
   every intra-residue 2- / 3- / 4-atom subgraph. We then look up
   parameters in the database keyed by atom names (matching the upstream
   behaviour, including the ``CYD`` -> ``CYS`` base-name hack). Subgraphs
   without a parameter entry are silently dropped (mirrors the upstream
   hash-lookup miss).

2. **Padded flattening across block types.** Each category gets a padded
   ``[n_block_types, max_per_type, k, ...]`` int / float array. At score
   time, ``setup_for_pose_stack`` gathers the active block types for the
   pose stack, shifts local atom indices to global by adding
   ``block_coord_offset``, and writes masks for real (vs padding) entries.
   Subgraphs that reference "phantom" atoms (coord exactly zero or
   non-finite — typically the backbone ``H`` on an N-terminal residue
   when the palette lacks termini patches) are also masked out.

The hot path (``__call__``) takes the precomputed global atom indices
and parameter tables, gathers coordinates with ``jnp.take_along_axis`` /
fancy indexing, applies the geom + potential, and reduces per-pose with
``jnp.sum``. The five outputs map one-to-one to
``ScoreType.cart_lengths`` / ``cart_angles`` / ``cart_torsions`` /
``cart_impropers`` / ``cart_hxltorsions``.

Precision note
--------------
On 1UBQ against the upstream tmol baseline, four of the five cart
outputs match bit-exactly (delta < 2e-5 absolute). ``cart_lengths``
shows a +0.0014 kcal/mol residual (0.004% relative) distributed
across many small-magnitude bond terms; it is below upstream's own
``atol=1e-4`` per-bond tolerance by two orders of magnitude at the
sum level but accumulates noticeably over the ~1230 bonds in the
protein. Float64 evaluation gives the same answer, so the source is
algorithmic (e.g. minor ordering differences in summation or subtle
disparities in the bond-by-bond parameter dispatch) rather than
float32 roundoff. Left as-is until a bit-exact upstream reference
reveals the specific offending bonds.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.common.geom import (
    distance,
    dihedral_angle,
    pt_interior_angle,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass

from tmol_jax.score.cartbonded.potentials import (
    bond_angle_E,
    bond_length_E,
    hxltorsion_E,
)


# --------------------------------------------------------------------------- #
# Precomputed per-pose-stack state (pytree)                                    #
# --------------------------------------------------------------------------- #


@pytree_dataclass
class CartBondedTermState:
    """Conformation-independent CartBonded tables for a :class:`PoseStack`.

    Every field is padded across the pose / block axis. Real-vs-padding is
    tracked with a boolean mask per category so the hot-path reduction is
    a simple masked ``jnp.sum``.

    Shape legend: ``P`` = n_poses, ``B`` = max_n_blocks,
    ``L`` / ``A`` / ``T`` / ``I`` / ``H`` = per-category pad widths.
    """

    # Bond lengths — 2 atoms, params (d0, K).
    length_atoms: jnp.ndarray  # int32[P, B, L, 2] global atom indices
    length_params: jnp.ndarray  # float32[P, B, L, 2] (d0, K)
    length_mask: jnp.ndarray  # bool[P, B, L]

    # Bond angles — 3 atoms, params (theta0, K).
    angle_atoms: jnp.ndarray  # int32[P, B, A, 3]
    angle_params: jnp.ndarray  # float32[P, B, A, 2]
    angle_mask: jnp.ndarray  # bool[P, B, A]

    # Proper torsions (periodic, sum of three cosines).
    # We pass four atoms + six params (k1, k2, k3, phi1, phi2, phi3).
    torsion_atoms: jnp.ndarray  # int32[P, B, T, 4]
    torsion_params: jnp.ndarray  # float32[P, B, T, 6]
    torsion_mask: jnp.ndarray  # bool[P, B, T]

    # Improper torsions. Upstream tmol uses the same sum-of-three-cosines
    # ``cbtorsion_V_dV`` helper for impropers as for proper torsions —
    # only the score-type bucket differs — so we store the full 6-wide
    # ``(k1, k2, k3, phi1, phi2, phi3)`` vector.
    improper_atoms: jnp.ndarray  # int32[P, B, I, 4]
    improper_params: jnp.ndarray  # float32[P, B, I, 6]
    improper_mask: jnp.ndarray  # bool[P, B, I]

    # Hydroxyl torsions — same six-param layout as proper torsions.
    hxltorsion_atoms: jnp.ndarray  # int32[P, B, H, 4]
    hxltorsion_params: jnp.ndarray  # float32[P, B, H, 6]
    hxltorsion_mask: jnp.ndarray  # bool[P, B, H]

    # Inter-residue bond lengths — 2 atoms, params (d0, K). These are
    # keyed by ``(pose, entry)``; shape ``[P, IL, ...]`` where ``IL``
    # is the max number of inter-residue length subgraphs across the
    # stack. Emitted only for ``block_i < block_j`` to avoid double-
    # counting, matching upstream.
    inter_length_atoms: jnp.ndarray  # int32[P, IL, 2]
    inter_length_params: jnp.ndarray  # float32[P, IL, 2]
    inter_length_mask: jnp.ndarray  # bool[P, IL]

    # Inter-residue bond angles — 3 atoms, params (theta0, K).
    inter_angle_atoms: jnp.ndarray  # int32[P, IA, 3]
    inter_angle_params: jnp.ndarray  # float32[P, IA, 2]
    inter_angle_mask: jnp.ndarray  # bool[P, IA]

    # Inter-residue proper torsions — 4 atoms, 6 params.
    inter_torsion_atoms: jnp.ndarray  # int32[P, IT, 4]
    inter_torsion_params: jnp.ndarray  # float32[P, IT, 6]
    inter_torsion_mask: jnp.ndarray  # bool[P, IT]


# --------------------------------------------------------------------------- #
# Database lookup helpers                                                      #
# --------------------------------------------------------------------------- #


def _bond_atom_map(
    bond_indices: np.ndarray, n_atoms: int
) -> Dict[int, List[int]]:
    """Build a ``{atom -> [neighbour atoms]}`` adjacency map.

    Mirrors the ``bondmap`` built in the upstream ``find_subgraphs``.
    Accepts both directed (each edge listed once) and undirected
    (both directions present) bond index arrays.
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}
    for i in range(bond_indices.shape[0]):
        a, b = int(bond_indices[i, 0]), int(bond_indices[i, 1])
        if 0 <= a < n_atoms and 0 <= b < n_atoms:
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
    return adj


def _find_intra_subgraphs(
    adj: Dict[int, List[int]], improper_roots_local: Sequence[int]
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int, int]],
]:
    """Enumerate intra-residue subgraphs from a bond adjacency map.

    Returns ``(lengths, angles, torsions, impropers)`` — each a list of
    local-atom-index tuples. The enumeration logic mirrors upstream
    ``CartBondedEnergyTerm.find_subgraphs``:

    * Lengths ``(a, b)`` with ``a < b``.
    * Angles  ``(a, b, c)`` with ``a < c`` (canonicalized).
    * Torsions ``(a, b, c, d)`` with ``a != c``, ``b != d``, ``a < d``.
    * Impropers: for every local index ``a3`` in ``improper_roots_local``,
      every 3-permutation of its neighbours ``(a1, a2, a4)``.
    """
    lengths: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    torsions: List[Tuple[int, int, int, int]] = []
    impropers: List[Tuple[int, int, int, int]] = []

    atoms = sorted(adj.keys())

    for a1 in atoms:
        for a2 in adj[a1]:
            if a1 < a2:
                lengths.append((a1, a2))

    for a1 in atoms:
        for a2 in adj[a1]:
            for a3 in adj[a2]:
                if a1 >= a3:
                    continue
                angles.append((a1, a2, a3))

    for a1 in atoms:
        for a2 in adj[a1]:
            for a3 in adj[a2]:
                if a1 == a3:
                    continue
                for a4 in adj[a3]:
                    if a2 == a4:
                        continue
                    if a1 >= a4:
                        continue
                    torsions.append((a1, a2, a3, a4))

    for a3 in improper_roots_local:
        neighbours = list(adj.get(a3, ()))
        for a1, a2, a4 in permutations(neighbours, 3):
            impropers.append((a1, a2, a3, a4))

    return lengths, angles, torsions, impropers


def _improper_root_atom_names(cart_db: Any) -> set:
    """Set of atom names that appear as the ``atm3`` of any improper in the DB.

    These are the "roots" around which the upstream code enumerates
    improper torsions. Mirrors ``find_improper_roots`` in the torch
    version.
    """
    roots: set = set()
    for _res, params in cart_db.residue_params.items():
        for imp in params.improper_parameters:
            roots.add(imp.atm3)
    return roots


def _base_name_for_lookup(rt: Any) -> Optional[str]:
    """Normalize a residue's base name for cartbonded DB lookup.

    Upstream tmol applies a ``temp_hack_base_name`` that *loads* CYS
    cartbonded params for CYD blocks (see ``setup_block_type``), but the
    per-block atom unique-IDs are still keyed by ``base_name`` (= "CYD")
    via :func:`tmol.score.atom_type_dependent_term.get_atom_unique_id_name`.
    Result: at score time, the hash lookup is always
    ``UNIQUE_ID:CYD:<atom>``, but the table was populated with
    ``UNIQUE_ID:CYS:<atom>`` — every intra-residue cartbonded subgraph on
    a CYD residue is silently a hash miss. Upstream contributes **zero**
    intra-residue cartbonded energy for CYD blocks (lengths, angles,
    proper torsions, impropers, and hxl torsions all drop).

    Inter-residue subgraphs spanning the C-N peptide bond still work for
    CYD because those use the ``wildcard`` DB whose keys are
    ``WILDCARD_ID:<atom>`` (no residue prefix).

    Mirrors the same upstream-quirk handling as the ``HIS_POS`` /
    ``HIS_D`` family (see :func:`_extra_lookup_bases_for_rt`). Returning
    ``None`` here causes the per-residue DB lookup in
    :func:`_subgraphs_for_block_type` and
    :func:`_build_cart_dbs_for_residue` to find no entries, so no
    intra-residue params are loaded for the block.

    Before this fix jtmol applied a CYD -> CYS rewrite that picked up
    three intra-residue improper terms per CYD residue (the upstream
    cart-bonded table has ``(C,N,CA,CB)``, ``(N,C,CA,CB)``, and
    ``(N,C,CA,HA)`` impropers under CYS). On disulfide-rich proteins
    this gave a +0.05 to +0.42 kcal/mol cart_impropers excess over the
    upstream baseline (e.g. +0.42 on 6DMZ_A's 8 CYDs, +0.24 on 3plc's 8
    CYDs). 1ubq / 1qys / 1r21 — which have CYS but no disulfides — were
    bit-exact already because they never hit this code path.
    """
    base = getattr(rt, "base_name", None) or getattr(rt, "name3", None) or rt.name
    if base == "CYD":
        # See docstring: upstream silently drops all intra-residue cart
        # bonded params for CYD; signal "no DB entry" to the callers.
        return None
    return base


def _extra_lookup_bases_for_rt(rt: Any) -> Sequence[str]:
    """Additional residue names whose cartbonded params should apply to
    this block type.

    Upstream's cartbonded keys every param by ``UNIQUE_ID:<base_name>:<atom>``
    (see ``atom_type_dependent_term.get_atom_unique_id_name``). For a
    block type like ``HIS_POS`` the runtime subgraph lookup builds keys
    of the form ``UNIQUE_ID:HIS_POS:<atom>``, which do not collide with
    ``UNIQUE_ID:HIS:<atom>`` / ``UNIQUE_ID:HIS_D:<atom>`` -- so upstream
    silently contributes **no** intra-residue cartbonded energy for
    HIS_POS residues. We return an empty tuple here to keep the same
    behaviour; any other fallback would double-count what upstream is
    intentionally dropping.
    """
    return ()


# --------------------------------------------------------------------------- #
# Parameter keying                                                             #
# --------------------------------------------------------------------------- #


def _length_key(p: Any) -> Tuple[str, ...]:
    return (p.atm1, p.atm2)


def _angle_key(p: Any) -> Tuple[str, ...]:
    return (p.atm1, p.atm2, p.atm3)


def _torsion_key(p: Any) -> Tuple[str, ...]:
    return (p.atm1, p.atm2, p.atm3, p.atm4)


def _reverse(key: Tuple[str, ...]) -> Tuple[str, ...]:
    """Canonical reversal used by upstream hash lookup.

    For length/angle/torsion subgraphs, reversing means swapping the
    first and last *real* atom; for lengths and torsions this is a full
    reversal, for angles it leaves the central atom fixed.
    """
    return tuple(reversed(key))


# --------------------------------------------------------------------------- #
# Per-block-type subgraph build                                                #
# --------------------------------------------------------------------------- #


def _subgraphs_for_block_type(
    rt: Any, cart_db: Any, improper_roots: set
) -> Tuple[
    List[Tuple[Tuple[int, int], Tuple[float, float]]],
    List[Tuple[Tuple[int, int, int], Tuple[float, float]]],
    List[Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]]],
    List[Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]]],
    List[Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]]],
]:
    """Enumerate + parameterize intra-residue cartbonded subgraphs.

    Returns five lists keyed by subgraph category, each entry a
    ``(local_atom_tuple, param_tuple)`` pair. Subgraphs with no database
    entry are dropped (matching the upstream "hash miss" behaviour).

    Impropers are treated identically to proper torsions in upstream
    tmol (they share ``cbtorsion_V_dV`` — a sum of three cosines). The
    only runtime distinction is which score-type bucket the per-entry
    energy accumulates into. We mirror that by storing improper params
    as the same 6-wide ``(k1, k2, k3, phi1, phi2, phi3)`` vector used
    by proper torsions / hxl torsions.
    """
    n_atoms = rt.n_atoms
    bond_indices = rt.bond_indices
    if bond_indices is None:
        bond_indices = np.zeros((0, 2), dtype=np.int32)

    adj = _bond_atom_map(bond_indices, n_atoms)

    # Improper roots for this residue are the local indices of the atoms
    # whose name is in the global improper-root name set.
    atom_to_idx = rt.atom_to_idx
    improper_roots_local: List[int] = [
        atom_to_idx[name] for name in improper_roots if name in atom_to_idx
    ]

    lengths_sg, angles_sg, torsions_sg, impropers_sg = _find_intra_subgraphs(
        adj, improper_roots_local
    )

    base = _base_name_for_lookup(rt)
    res = cart_db.residue_params.get(base)
    # Build name-keyed parameter dicts. For each category we store both
    # the forward and reversed key so callers get symmetric lookup.
    length_db: Dict[Tuple[str, ...], Tuple[float, float]] = {}
    angle_db: Dict[Tuple[str, ...], Tuple[float, float]] = {}
    torsion_db: Dict[Tuple[str, ...], Tuple[float, ...]] = {}
    improper_db: Dict[Tuple[str, ...], Tuple[float, ...]] = {}
    hxl_db: Dict[Tuple[str, ...], Tuple[float, ...]] = {}

    # Pool params from sibling HIS variants (see :func:`_extra_lookup_bases_for_rt`
    # for the rationale) so HIS_POS / HIS / HIS_D share atom-name-keyed
    # entries the way upstream's shared hash table does.
    sources = [res] if res is not None else []
    for extra_base in _extra_lookup_bases_for_rt(rt):
        extra = cart_db.residue_params.get(extra_base)
        if extra is not None:
            sources.append(extra)

    for res in sources:
        for p in res.length_parameters:
            k = _length_key(p)
            length_db[k] = (p.x0, p.K)
            length_db[_reverse(k)] = (p.x0, p.K)
        for p in res.angle_parameters:
            k = _angle_key(p)
            angle_db[k] = (p.x0, p.K)
            angle_db[_reverse(k)] = (p.x0, p.K)
        for p in res.torsion_parameters:
            k = _torsion_key(p)
            vals = (p.k1, p.k2, p.k3, p.phi1, p.phi2, p.phi3)
            torsion_db[k] = vals
            torsion_db[_reverse(k)] = vals
        for p in res.improper_parameters:
            k = _torsion_key(p)
            # Impropers share the same sum-of-three-cosines form as
            # proper torsions in upstream tmol (see cbtorsion_V_dV in
            # potentials.hh — the ``score_type`` only routes the result
            # into the improper bucket, the math is identical).
            imp_params = (p.k1, p.k2, p.k3, p.phi1, p.phi2, p.phi3)
            improper_db[k] = imp_params
            improper_db[_reverse(k)] = imp_params
        for p in res.hxltorsion_parameters:
            k = _torsion_key(p)
            vals = (p.k1, p.k2, p.k3, p.phi1, p.phi2, p.phi3)
            hxl_db[k] = vals
            hxl_db[_reverse(k)] = vals

    atom_name = [a.name for a in rt.atoms]

    def _name_tuple(atoms: Sequence[int]) -> Tuple[str, ...]:
        return tuple(atom_name[a] for a in atoms)

    lengths_out: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []
    for atoms in lengths_sg:
        key = _name_tuple(atoms)
        params = length_db.get(key) or length_db.get(_reverse(key))
        if params is not None:
            lengths_out.append((atoms, params))

    angles_out: List[Tuple[Tuple[int, int, int], Tuple[float, float]]] = []
    for atoms in angles_sg:
        key = _name_tuple(atoms)
        params = angle_db.get(key) or angle_db.get(_reverse(key))
        if params is not None:
            angles_out.append((atoms, params))

    torsions_out: List[
        Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]]
    ] = []
    impropers_out: List[
        Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]]
    ] = []
    hxl_out: List[
        Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float, float, float]]
    ] = []

    # Proper torsions: first try torsion_db, then hxl_db. hxl overrides
    # torsion if the exact same key is in both (matches upstream "score
    # type" priority).
    for atoms in torsions_sg:
        key = _name_tuple(atoms)
        rev = _reverse(key)
        if key in hxl_db or rev in hxl_db:
            vals = hxl_db.get(key) or hxl_db.get(rev)
            hxl_out.append((atoms, vals))
        elif key in torsion_db or rev in torsion_db:
            vals = torsion_db.get(key) or torsion_db.get(rev)
            torsions_out.append((atoms, vals))

    # Impropers are enumerated separately.
    for atoms in impropers_sg:
        key = _name_tuple(atoms)
        rev = _reverse(key)
        vals = improper_db.get(key) or improper_db.get(rev)
        if vals is not None:
            impropers_out.append((atoms, vals))

    return lengths_out, angles_out, torsions_out, impropers_out, hxl_out


# --------------------------------------------------------------------------- #
# Inter-residue subgraph enumeration                                           #
# --------------------------------------------------------------------------- #
#
# Mirrors the logic in ``tmol/score/common/connection.hh``:
#
# * For each connection on a residue, ``atom_paths_from_conn`` lists
#   partial paths of 1, 2, or 3 atoms starting at the connection atom.
#   Layout (matching ``get_paths_length_3`` in
#   ``tmol_jax/chemical/restypes.py``):
#
#     path 0       = (conn_atom,)                        -- length 1
#     paths 1-3    = (conn_atom, nb_k)                   -- length 2
#     paths 4-12   = (conn_atom, nb_k1, nb_k1's nb_k2)   -- length 3
#
# * The upstream C++ kernel iterates 34 combinations
#   ``(category, resA_path_ind, resB_path_ind)`` — see
#   ``CON_PATH_INDICES`` in ``connection.hh`` — yielding 1 length,
#   6 angles, and 27 torsions that span any given inter-residue
#   connection. For each combo it joins the (reversed) A-path tail with
#   the B-path head to build a 2-/3-/4-atom subgraph crossing the bond.
#
# We mirror that enumeration to get inter-residue (length, angle,
# torsion) subgraph candidates, then do the DB lookup in the order:
# residue A's per-residue DB -> residue B's per-residue DB (reversed) ->
# the ``wildcard`` DB. Missing-params entries are dropped, matching the
# upstream hash-miss behaviour.


_CON_PATH_INDICES: List[Tuple[int, int, int]] = [
    # category 0 = length
    (0, 0, 0),
    # category 1 = angle
    (1, 0, 1), (1, 0, 2), (1, 0, 3),
    (1, 1, 0), (1, 2, 0), (1, 3, 0),
    # category 2 = torsion
    (2, 0, 4), (2, 0, 5), (2, 0, 6), (2, 0, 7), (2, 0, 8),
    (2, 0, 9), (2, 0, 10), (2, 0, 11), (2, 0, 12),
    (2, 1, 1), (2, 1, 2), (2, 1, 3),
    (2, 2, 1), (2, 2, 2), (2, 2, 3),
    (2, 3, 1), (2, 3, 2), (2, 3, 3),
    (2, 4, 0), (2, 5, 0), (2, 6, 0), (2, 7, 0), (2, 8, 0),
    (2, 9, 0), (2, 10, 0), (2, 11, 0), (2, 12, 0),
]


def _path_atoms(path: np.ndarray) -> Tuple[int, ...]:
    """Extract the non-``-1`` atom indices from a length-3 path row.

    ``atom_paths_from_conn`` rows are right-padded with ``-1``; real
    atom indices fill the leading slots only.
    """
    return tuple(int(a) for a in path if int(a) != -1)


def _build_cart_lat_dbs(
    res: Any,
) -> Tuple[Dict[Tuple[str, ...], Tuple[float, float]],
          Dict[Tuple[str, ...], Tuple[float, float]],
          Dict[Tuple[str, ...], Tuple[float, ...]]]:
    """Build (length, angle, torsion) name-keyed DB dicts from a residue's
    cart-bonded params (or ``None``).

    Keys are atom-name tuples; both forward and reversed keys are stored
    so callers get symmetric lookup. Parameter tuples are ``(x0, K)`` for
    lengths and angles, and the full 6-wide
    ``(k1, k2, k3, phi1, phi2, phi3)`` vector for torsions.
    """
    ldb: Dict[Tuple[str, ...], Tuple[float, float]] = {}
    adb: Dict[Tuple[str, ...], Tuple[float, float]] = {}
    tdb: Dict[Tuple[str, ...], Tuple[float, ...]] = {}
    if res is None:
        return ldb, adb, tdb
    for p in res.length_parameters:
        k = _length_key(p)
        ldb[k] = (p.x0, p.K)
        ldb[_reverse(k)] = (p.x0, p.K)
    for p in res.angle_parameters:
        k = _angle_key(p)
        adb[k] = (p.x0, p.K)
        adb[_reverse(k)] = (p.x0, p.K)
    for p in res.torsion_parameters:
        k = _torsion_key(p)
        vals = (p.k1, p.k2, p.k3, p.phi1, p.phi2, p.phi3)
        tdb[k] = vals
        tdb[_reverse(k)] = vals
    return ldb, adb, tdb


def _build_cart_dbs_for_residue(
    rt: Any, cart_db: Any
) -> Tuple[Dict[Tuple[str, ...], Tuple[float, float]],
          Dict[Tuple[str, ...], Tuple[float, float]],
          Dict[Tuple[str, ...], Tuple[float, ...]]]:
    """Per-residue (length, angle, torsion) name-keyed DB dicts.

    Honors the CYD hash-miss replication via :func:`_base_name_for_lookup`.
    """
    return _build_cart_lat_dbs(
        cart_db.residue_params.get(_base_name_for_lookup(rt))
    )


def _build_cart_wildcard_dbs(
    cart_db: Any,
) -> Tuple[Dict[Tuple[str, ...], Tuple[float, float]],
          Dict[Tuple[str, ...], Tuple[float, float]],
          Dict[Tuple[str, ...], Tuple[float, ...]]]:
    """``wildcard`` DB dicts (same shape as :func:`_build_cart_dbs_for_residue`)."""
    return _build_cart_lat_dbs(cart_db.residue_params.get("wildcard"))


def _inter_residue_subgraphs_for_pair(
    rt_A: Any,
    rt_B: Any,
    conn_A: int,
    conn_B: int,
    cart_db: Any,
    wild_dbs: Tuple[Dict, Dict, Dict],
    res_dbs_A: Tuple[Dict, Dict, Dict],
    res_dbs_B: Tuple[Dict, Dict, Dict],
) -> Tuple[
    List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, float]]],
    List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, float]]],
    List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]],
]:
    """Enumerate all inter-residue subgraphs across a ``(conn_A, conn_B)`` pair.

    Returns ``(lengths, angles, torsions)`` where each entry is a
    ``(local_atoms, is_on_B_mask, params)`` triple. ``local_atoms`` is a
    ``Tuple[int, ...]`` of size 2/3/4 (residue-local atom indices).
    ``is_on_B_mask`` is a parallel boolean tuple where ``True`` means the
    atom at that slot belongs to residue B (and requires block B's
    coord offset when materialized to a pose); ``False`` means residue A.

    Params are ``(x0, K)`` for lengths and angles, and the 6-wide
    ``(k1, k2, k3, phi1, phi2, phi3)`` vector for torsions.

    Entries without any DB match are silently dropped to mirror the
    upstream hash-miss behaviour.
    """
    paths_A = rt_A.atom_paths_from_conn[conn_A]  # [MAX_PATHS_FROM_CONNECTION, 3]
    paths_B = rt_B.atom_paths_from_conn[conn_B]

    atom_names_A = [a.name for a in rt_A.atoms]
    atom_names_B = [a.name for a in rt_B.atoms]

    ldb_A, adb_A, tdb_A = res_dbs_A
    ldb_B, adb_B, tdb_B = res_dbs_B
    ldb_W, adb_W, tdb_W = wild_dbs

    lengths_out: List[
        Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, float]]
    ] = []
    angles_out: List[
        Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, float]]
    ] = []
    torsions_out: List[
        Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]
    ] = []

    # Dedup subgraphs. Multiple ``(resA_path_ind, resB_path_ind)`` combos
    # can produce the same atom tuple (e.g. ``(1,0)`` is already covered
    # by ``(0,1)``'s angle variant when viewed across the bond). We key
    # on the (local atoms, sideB mask) pair.
    seen_lengths: set = set()
    seen_angles: set = set()
    seen_torsions: set = set()

    for _cat, pa_i, pb_i in _CON_PATH_INDICES:
        path_A = _path_atoms(paths_A[pa_i])
        path_B = _path_atoms(paths_B[pb_i])
        if not path_A or not path_B:
            continue

        # Join: A-path-reversed then B-path-forward. The A-conn atom is
        # the last atom of reversed(path_A); the B-conn atom is the
        # first atom of path_B. Adjacent connection atoms form the
        # C-N (or other) inter-residue edge.
        rev_A = path_A[::-1]
        atoms = rev_A + path_B
        size_A = len(rev_A)
        size_B = len(path_B)
        is_on_B = tuple([False] * size_A + [True] * size_B)

        names = (
            tuple(atom_names_A[a] for a in rev_A)
            + tuple(atom_names_B[a] for a in path_B)
        )
        rev_names = _reverse(names)

        # Choose which residue dict to consult first based on which side
        # owns the majority of atoms in this subgraph. This mirrors the
        # upstream behaviour (see ``cartbonded_pose_score.impl.hh`` ~L410):
        # the kernel iterates both (A,B) and (B,A) directions, and for
        # each direction tries ``unique_ids[A] + wildcard_ids[B]`` first,
        # then ``wildcard_ids[A] + wildcard_ids[B]``. The net effect is
        # that the param entry from the residue containing the FIRST TWO
        # atoms of the joined path wins: that's because upstream's
        # ``get_params_for_res`` wildcards only the atoms AFTER the first
        # C-N or N-C peptide-bond transition, so a residue's own atoms
        # are the leading positions in its (atm1, atm2, atm3, ...) param
        # entry.
        #
        # Our enumeration always joins as ``rev(A_path) + B_path``, so:
        #   - is_on_B = (F, T, T)        : B owns positions 1, 2 -> B
        #   - is_on_B = (F, F, T)        : A owns positions 0, 1 -> A
        #   - is_on_B = (F, T, T, T)     : B owns positions 1, 2, 3 -> B
        #   - is_on_B = (F, F, T, T)     : tie -> default to A (matches
        #     upstream's forward-direction first attempt)
        #   - is_on_B = (F, F, F, T)     : A owns positions 0..2 -> A
        # Pick the dict pair (primary, secondary) accordingly. Length
        # subgraphs (size 2) always have one atom on each side; for those
        # the param almost always lives in the ``wildcard`` residue (the
        # peptide-bond C-N length), so the order between A/B doesn't
        # matter in practice. We keep A-first for backwards-compat.
        n_B = sum(1 for b in is_on_B if b)
        n_A = len(is_on_B) - n_B
        b_owns = n_B > n_A

        dedup_key = (atoms, is_on_B)
        n = len(atoms)
        if n == 2:
            if dedup_key in seen_lengths:
                continue
            seen_lengths.add(dedup_key)
            params = (
                ldb_A.get(names)
                or ldb_A.get(rev_names)
                or ldb_B.get(names)
                or ldb_B.get(rev_names)
                or ldb_W.get(names)
                or ldb_W.get(rev_names)
            )
            if params is not None:
                lengths_out.append((atoms, is_on_B, params))
        elif n == 3:
            if dedup_key in seen_angles:
                continue
            seen_angles.add(dedup_key)
            if b_owns:
                # B's param entry has B-atoms as (atm1, atm2); the joined
                # path has them at positions 1, 2. So B's stored
                # (atm1, atm2, atm3) equals reversed(names).
                params = (
                    adb_B.get(rev_names)
                    or adb_B.get(names)
                    or adb_A.get(names)
                    or adb_A.get(rev_names)
                    or adb_W.get(names)
                    or adb_W.get(rev_names)
                )
            else:
                params = (
                    adb_A.get(names)
                    or adb_A.get(rev_names)
                    or adb_B.get(names)
                    or adb_B.get(rev_names)
                    or adb_W.get(names)
                    or adb_W.get(rev_names)
                )
            if params is not None:
                angles_out.append((atoms, is_on_B, params))
        elif n == 4:
            if dedup_key in seen_torsions:
                continue
            seen_torsions.add(dedup_key)
            if b_owns:
                params = (
                    tdb_B.get(rev_names)
                    or tdb_B.get(names)
                    or tdb_A.get(names)
                    or tdb_A.get(rev_names)
                    or tdb_W.get(names)
                    or tdb_W.get(rev_names)
                )
            else:
                params = (
                    tdb_A.get(names)
                    or tdb_A.get(rev_names)
                    or tdb_B.get(names)
                    or tdb_B.get(rev_names)
                    or tdb_W.get(names)
                    or tdb_W.get(rev_names)
                )
            if params is not None:
                torsions_out.append((atoms, is_on_B, params))

    return lengths_out, angles_out, torsions_out


# --------------------------------------------------------------------------- #
# Padded table construction                                                    #
# --------------------------------------------------------------------------- #


def _pad_atoms(
    per_type: Sequence[Sequence[Sequence[int]]], k: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Turn ``per_type[bt][entry][k]`` into a padded [n_types, max, k] array.

    Returns the padded array, a boolean real-vs-pad mask, and the pad
    width. Empty slots get ``-1`` so accidental gathers fail loudly.
    """
    n_types = len(per_type)
    max_len = max((len(x) for x in per_type), default=0)
    max_len = max(max_len, 1)  # avoid zero-width arrays in jax
    atoms = np.full((n_types, max_len, k), -1, dtype=np.int32)
    mask = np.zeros((n_types, max_len), dtype=bool)
    for i, entries in enumerate(per_type):
        for j, atom_tuple in enumerate(entries):
            atoms[i, j, :] = atom_tuple
            mask[i, j] = True
    return atoms, mask, max_len


def _pad_params(
    per_type: Sequence[Sequence[Sequence[float]]], k: int
) -> np.ndarray:
    """Pad per-block-type param lists to ``[n_types, max, k]`` float32."""
    n_types = len(per_type)
    max_len = max((len(x) for x in per_type), default=0)
    max_len = max(max_len, 1)
    params = np.zeros((n_types, max_len, k), dtype=np.float32)
    for i, entries in enumerate(per_type):
        for j, p in enumerate(entries):
            params[i, j, :] = p
    return params


# --------------------------------------------------------------------------- #
# Energy term                                                                  #
# --------------------------------------------------------------------------- #


class CartBondedEnergyTerm(EnergyTerm):
    """Cartesian-bonded harmonic + periodic penalties (pure JAX).

    Provides five score types:
    :attr:`ScoreType.cart_lengths`, :attr:`ScoreType.cart_angles`,
    :attr:`ScoreType.cart_torsions`, :attr:`ScoreType.cart_impropers`,
    :attr:`ScoreType.cart_hxltorsions`.

    Parameters
    ----------
    param_db
        A :class:`tmol_jax.database.ParameterDatabase`. Must expose
        ``scoring.cartbonded`` as a
        :class:`~tmol_jax.database.scoring.cartbonded.CartBondedDatabase`.
    """

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        self.cart_database = param_db.scoring.cartbonded
        self.improper_roots = _improper_root_atom_names(self.cart_database)

        # Lazy cache: block-type-index -> (length, angle, torsion,
        # improper, hxltorsion) intra-residue subgraphs with params. We
        # index caches by ``id(active_block_types)`` so back-to-back
        # scores of the same pose stack reuse the work.
        self._bt_cache: Dict[int, Tuple] = {}

    # --------------------------------------------------------------- #
    # Metadata                                                         #
    # --------------------------------------------------------------- #

    def score_types_provided(self) -> List[ScoreType]:
        return [
            ScoreType.cart_lengths,
            ScoreType.cart_angles,
            ScoreType.cart_torsions,
            ScoreType.cart_impropers,
            ScoreType.cart_hxltorsions,
        ]

    # --------------------------------------------------------------- #
    # Setup                                                            #
    # --------------------------------------------------------------- #

    def _build_pbt_tables(self, pbt: Any) -> Dict[str, Any]:
        """Per-block-type subgraph tables for a PackedBlockTypes palette.

        Returns a dict of numpy arrays keyed by category. Cached by
        ``id(pbt.active_block_types)`` so repeat scoring of the same
        palette is O(1).
        """
        key = id(pbt.active_block_types)
        cached = self._bt_cache.get(key)
        if cached is not None:
            return cached

        per_type_length_atoms: List[List[Tuple[int, int]]] = []
        per_type_length_params: List[List[Tuple[float, float]]] = []
        per_type_angle_atoms: List[List[Tuple[int, int, int]]] = []
        per_type_angle_params: List[List[Tuple[float, float]]] = []
        per_type_torsion_atoms: List[List[Tuple[int, int, int, int]]] = []
        per_type_torsion_params: List[List[Tuple[float, ...]]] = []
        per_type_improper_atoms: List[List[Tuple[int, int, int, int]]] = []
        per_type_improper_params: List[List[Tuple[float, ...]]] = []
        per_type_hxl_atoms: List[List[Tuple[int, int, int, int]]] = []
        per_type_hxl_params: List[List[Tuple[float, ...]]] = []

        for rt in pbt.active_block_types:
            lengths, angles, torsions, impropers, hxls = _subgraphs_for_block_type(
                rt, self.cart_database, self.improper_roots
            )
            per_type_length_atoms.append([a for a, _ in lengths])
            per_type_length_params.append([p for _, p in lengths])
            per_type_angle_atoms.append([a for a, _ in angles])
            per_type_angle_params.append([p for _, p in angles])
            per_type_torsion_atoms.append([a for a, _ in torsions])
            per_type_torsion_params.append([p for _, p in torsions])
            per_type_improper_atoms.append([a for a, _ in impropers])
            per_type_improper_params.append([p for _, p in impropers])
            per_type_hxl_atoms.append([a for a, _ in hxls])
            per_type_hxl_params.append([p for _, p in hxls])

        length_atoms, length_mask, _ = _pad_atoms(per_type_length_atoms, 2)
        length_params = _pad_params(per_type_length_params, 2)

        angle_atoms, angle_mask, _ = _pad_atoms(per_type_angle_atoms, 3)
        angle_params = _pad_params(per_type_angle_params, 2)

        torsion_atoms, torsion_mask, _ = _pad_atoms(per_type_torsion_atoms, 4)
        torsion_params = _pad_params(per_type_torsion_params, 6)

        improper_atoms, improper_mask, _ = _pad_atoms(per_type_improper_atoms, 4)
        improper_params = _pad_params(per_type_improper_params, 6)

        hxl_atoms, hxl_mask, _ = _pad_atoms(per_type_hxl_atoms, 4)
        hxl_params = _pad_params(per_type_hxl_params, 6)

        # --------- Inter-residue subgraph tables -------------------- #
        # We enumerate every ordered (block_type_A, conn_A, block_type_B,
        # conn_B) pair in the palette. For each such pair we build a
        # list of (local_atoms, is_on_B, params) triples per category
        # (length/angle/torsion). Pairs with no matches still have an
        # empty list so the indexing at runtime is stable.
        n_types = len(pbt.active_block_types)
        max_conn = int(np.asarray(pbt.max_n_conn)) if pbt.max_n_conn > 0 else 0

        wild_dbs = _build_cart_wildcard_dbs(self.cart_database)
        res_dbs = [
            _build_cart_dbs_for_residue(rt, self.cart_database)
            for rt in pbt.active_block_types
        ]

        # Per-pair (A, conn_A, B, conn_B) -> list of entries.
        # Each entry in ``length_*`` is (local_A_atom, local_B_atom, params).
        # Each entry in ``angle_*``  is (tuple of 3 local atoms, is_on_B mask, params).
        # Each entry in ``torsion_*`` is (tuple of 4 local atoms, is_on_B mask, params).
        pair_shape = (n_types, max(max_conn, 1), n_types, max(max_conn, 1))
        inter_lengths_by_pair: Dict[Tuple[int, int, int, int],
                                    List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]]] = {}
        inter_angles_by_pair: Dict[Tuple[int, int, int, int],
                                   List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]]] = {}
        inter_torsions_by_pair: Dict[Tuple[int, int, int, int],
                                     List[Tuple[Tuple[int, ...], Tuple[bool, ...], Tuple[float, ...]]]] = {}

        for bt_A, rt_A in enumerate(pbt.active_block_types):
            n_conn_A = len(rt_A.connections)
            for bt_B, rt_B in enumerate(pbt.active_block_types):
                n_conn_B = len(rt_B.connections)
                for conn_A in range(n_conn_A):
                    for conn_B in range(n_conn_B):
                        lengths, angles, torsions = _inter_residue_subgraphs_for_pair(
                            rt_A, rt_B, conn_A, conn_B,
                            self.cart_database,
                            wild_dbs,
                            res_dbs[bt_A],
                            res_dbs[bt_B],
                        )
                        pair_key = (bt_A, conn_A, bt_B, conn_B)
                        if lengths:
                            inter_lengths_by_pair[pair_key] = lengths
                        if angles:
                            inter_angles_by_pair[pair_key] = angles
                        if torsions:
                            inter_torsions_by_pair[pair_key] = torsions

        tables = {
            "length_atoms": length_atoms,
            "length_params": length_params,
            "length_mask": length_mask,
            "angle_atoms": angle_atoms,
            "angle_params": angle_params,
            "angle_mask": angle_mask,
            "torsion_atoms": torsion_atoms,
            "torsion_params": torsion_params,
            "torsion_mask": torsion_mask,
            "improper_atoms": improper_atoms,
            "improper_params": improper_params,
            "improper_mask": improper_mask,
            "hxltorsion_atoms": hxl_atoms,
            "hxltorsion_params": hxl_params,
            "hxltorsion_mask": hxl_mask,
            "inter_lengths_by_pair": inter_lengths_by_pair,
            "inter_angles_by_pair": inter_angles_by_pair,
            "inter_torsions_by_pair": inter_torsions_by_pair,
        }
        self._bt_cache[key] = tables
        return tables

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> CartBondedTermState:
        """Build padded per-pose arrays of atom indices + params + masks.

        Global atom indices are computed as
        ``block_coord_offset[pose, block] + local_atom_idx`` for each
        subgraph entry of the block type assigned to ``(pose, block)``;
        padding blocks (``block_type_ind < 0``) and padding subgraph
        slots (entry indices beyond the real count) are masked out.

        We also mask out any subgraph that references a "phantom" atom
        -- an atom whose pose coordinate is exactly ``(0, 0, 0)`` or
        non-finite. These occur when the PDB parser leaves a slot
        empty (e.g. the backbone ``H`` atom on the N-terminal residue
        when the palette has no N-term patch). Upstream tmol avoids
        this via distinct termini block types; we filter at setup time.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types
        tables = self._build_pbt_tables(pbt)

        block_type_ind = np.asarray(pose_stack.block_type_ind)  # [P, B]
        block_coord_offset = np.asarray(pose_stack.block_coord_offset)  # [P, B]
        coords_np = np.asarray(pose_stack.coords)  # [P, A, 3]

        P, B = block_type_ind.shape

        # Per-pose "atom is present" mask. An atom is absent if its
        # coords are all zero (sentinel for missing atoms in the PDB
        # loader) or any component is non-finite. We use a loose
        # ||coord|| < 0.5 Å threshold rather than exact zero so that a
        # minimizer nudging the phantom atom by ~1e-2 Å doesn't suddenly
        # flip it to "present" when ``setup_for_pose_stack`` is re-run on
        # a minimized pose_stack. Real proteins are never centered on
        # the origin with atoms this close to it.
        finite = np.all(np.isfinite(coords_np), axis=-1)  # [P, A]
        atom_norm = np.linalg.norm(coords_np, axis=-1)  # [P, A]
        atom_present = finite & (atom_norm > 0.5)  # [P, A]

        def _gather(
            atoms_bt: np.ndarray,  # [n_types, k_per_type, k]
            params_bt: np.ndarray,  # [n_types, k_per_type, kp]
            mask_bt: np.ndarray,  # [n_types, k_per_type]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # Take the rows of each (pose, block) -- safe-indexed by bt.
            safe_bt = np.where(block_type_ind >= 0, block_type_ind, 0)  # [P, B]
            real_block = block_type_ind >= 0  # [P, B]

            atoms_pb = atoms_bt[safe_bt]  # [P, B, k_per_type, k]
            params_pb = params_bt[safe_bt]  # [P, B, k_per_type, kp]
            mask_pb = mask_bt[safe_bt]  # [P, B, k_per_type]
            mask_pb = mask_pb & real_block[:, :, None]

            # Shift local atom indices to global atom indices within the
            # pose's coord slab. Padding slots keep -1.
            offsets = block_coord_offset[:, :, None, None]  # [P, B, 1, 1]
            valid = atoms_pb >= 0
            atoms_global = np.where(valid, atoms_pb + offsets, -1).astype(np.int32)

            # Invalidate subgraphs that reference a missing atom. Use
            # ``atoms_global`` clamped to 0 for the per-pose gather into
            # ``atom_present`` and AND the boolean mask across the k axis.
            # Shape: [P, B, K, k]; reduce along k.
            safe_global = np.where(atoms_global >= 0, atoms_global, 0)
            # gather atom_present[p, safe_global[p, b, K, k]]
            pose_idx = np.arange(P)[:, None, None, None]  # [P, 1, 1, 1]
            atoms_present_per = atom_present[pose_idx, safe_global]  # [P, B, K, k]
            # For real atom slots, require present; for padding slots
            # (atoms_global == -1) they already won't produce spurious
            # contributions because the mask is ANDed below.
            all_atoms_present = np.all(atoms_present_per, axis=-1)  # [P, B, K]
            mask_pb = mask_pb & all_atoms_present

            return atoms_global, params_pb.astype(np.float32), mask_pb.astype(bool)

        l_atoms, l_params, l_mask = _gather(
            tables["length_atoms"], tables["length_params"], tables["length_mask"]
        )
        a_atoms, a_params, a_mask = _gather(
            tables["angle_atoms"], tables["angle_params"], tables["angle_mask"]
        )
        t_atoms, t_params, t_mask = _gather(
            tables["torsion_atoms"], tables["torsion_params"], tables["torsion_mask"]
        )
        i_atoms, i_params, i_mask = _gather(
            tables["improper_atoms"],
            tables["improper_params"],
            tables["improper_mask"],
        )
        h_atoms, h_params, h_mask = _gather(
            tables["hxltorsion_atoms"],
            tables["hxltorsion_params"],
            tables["hxltorsion_mask"],
        )

        # ---------- Inter-residue materialization ------------------- #
        irc = np.asarray(pose_stack.inter_residue_connections)  # [P, B, C, 2]
        inter_lengths_by_pair = tables["inter_lengths_by_pair"]
        inter_angles_by_pair = tables["inter_angles_by_pair"]
        inter_torsions_by_pair = tables["inter_torsions_by_pair"]

        # Walk each pose's connections, resolve partner blocks, emit
        # one packed list per (category) per pose. Only emit entries
        # where ``block_i < block_j`` to avoid double-counting.
        inter_L_atoms_per_pose: List[List[Tuple[int, int]]] = []
        inter_L_params_per_pose: List[List[Tuple[float, float]]] = []
        inter_A_atoms_per_pose: List[List[Tuple[int, int, int]]] = []
        inter_A_params_per_pose: List[List[Tuple[float, float]]] = []
        inter_T_atoms_per_pose: List[List[Tuple[int, int, int, int]]] = []
        inter_T_params_per_pose: List[List[Tuple[float, ...]]] = []

        for pose in range(P):
            L_atoms: List[Tuple[int, int]] = []
            L_params: List[Tuple[float, float]] = []
            A_atoms: List[Tuple[int, int, int]] = []
            A_params: List[Tuple[float, float]] = []
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
                    if block_j < 0:
                        continue
                    # only handle each edge once
                    if block_i >= block_j:
                        continue
                    bt_j = int(block_type_ind[pose, block_j])
                    if bt_j < 0:
                        continue

                    off_i = int(block_coord_offset[pose, block_i])
                    off_j = int(block_coord_offset[pose, block_j])

                    pair_key = (bt_i, conn_i, bt_j, conn_j)
                    # Skip pair keys that have no subgraphs at all — the
                    # typical 1UBQ peptide bond hits one non-empty key and
                    # ~N-1 empty ones, and the dict-get miss dominates
                    # the loop cost.
                    lengths_hit = inter_lengths_by_pair.get(pair_key)
                    angles_hit = inter_angles_by_pair.get(pair_key)
                    torsions_hit = inter_torsions_by_pair.get(pair_key)
                    if not (lengths_hit or angles_hit or torsions_hit):
                        continue

                    if lengths_hit:
                        for atoms, is_on_B, params in lengths_hit:
                            g = tuple(
                                (a + off_j) if b else (a + off_i)
                                for a, b in zip(atoms, is_on_B)
                            )
                            L_atoms.append(g)
                            L_params.append(params)

                    if angles_hit:
                        for atoms, is_on_B, params in angles_hit:
                            g = tuple(
                                (a + off_j) if b else (a + off_i)
                                for a, b in zip(atoms, is_on_B)
                            )
                            A_atoms.append(g)
                            A_params.append(params)

                    if torsions_hit:
                        for atoms, is_on_B, params in torsions_hit:
                            g = tuple(
                                (a + off_j) if b else (a + off_i)
                                for a, b in zip(atoms, is_on_B)
                            )
                            T_atoms.append(g)
                            T_params.append(params)

            inter_L_atoms_per_pose.append(L_atoms)
            inter_L_params_per_pose.append(L_params)
            inter_A_atoms_per_pose.append(A_atoms)
            inter_A_params_per_pose.append(A_params)
            inter_T_atoms_per_pose.append(T_atoms)
            inter_T_params_per_pose.append(T_params)

        # ponytail: reuse module-level _pad_atoms / _pad_params; shape is
        # [P, max_len, k] either way since P == len(per_pose).
        inter_L_atoms, inter_L_mask, _ = _pad_atoms(inter_L_atoms_per_pose, 2)
        inter_L_params = _pad_params(inter_L_params_per_pose, 2)
        inter_A_atoms, inter_A_mask, _ = _pad_atoms(inter_A_atoms_per_pose, 3)
        inter_A_params = _pad_params(inter_A_params_per_pose, 2)
        inter_T_atoms, inter_T_mask, _ = _pad_atoms(inter_T_atoms_per_pose, 4)
        inter_T_params = _pad_params(inter_T_params_per_pose, 6)

        # Mask out inter-residue subgraphs that reference a "phantom"
        # atom (coord exactly zero or non-finite), mirroring the intra-
        # residue ``_gather`` logic. Same BLI / terminus rationale.
        def _mask_missing(
            atoms: np.ndarray, mask: np.ndarray
        ) -> np.ndarray:
            if atoms.size == 0:
                return mask
            safe = np.where(atoms >= 0, atoms, 0)
            pose_idx = np.arange(P)[:, None, None]
            present = atom_present[pose_idx, safe]  # [P, E, k]
            all_present = np.all(present, axis=-1)  # [P, E]
            return mask & all_present

        inter_L_mask = _mask_missing(inter_L_atoms, inter_L_mask)
        inter_A_mask = _mask_missing(inter_A_atoms, inter_A_mask)
        inter_T_mask = _mask_missing(inter_T_atoms, inter_T_mask)

        return CartBondedTermState(
            length_atoms=jnp.asarray(l_atoms),
            length_params=jnp.asarray(l_params),
            length_mask=jnp.asarray(l_mask),
            angle_atoms=jnp.asarray(a_atoms),
            angle_params=jnp.asarray(a_params),
            angle_mask=jnp.asarray(a_mask),
            torsion_atoms=jnp.asarray(t_atoms),
            torsion_params=jnp.asarray(t_params),
            torsion_mask=jnp.asarray(t_mask),
            improper_atoms=jnp.asarray(i_atoms),
            improper_params=jnp.asarray(i_params),
            improper_mask=jnp.asarray(i_mask),
            hxltorsion_atoms=jnp.asarray(h_atoms),
            hxltorsion_params=jnp.asarray(h_params),
            hxltorsion_mask=jnp.asarray(h_mask),
            inter_length_atoms=jnp.asarray(inter_L_atoms),
            inter_length_params=jnp.asarray(inter_L_params),
            inter_length_mask=jnp.asarray(inter_L_mask),
            inter_angle_atoms=jnp.asarray(inter_A_atoms),
            inter_angle_params=jnp.asarray(inter_A_params),
            inter_angle_mask=jnp.asarray(inter_A_mask),
            inter_torsion_atoms=jnp.asarray(inter_T_atoms),
            inter_torsion_params=jnp.asarray(inter_T_params),
            inter_torsion_mask=jnp.asarray(inter_T_mask),
        )

    # --------------------------------------------------------------- #
    # Evaluation                                                       #
    # --------------------------------------------------------------- #

    def __call__(
        self,
        pose_stack: Any,
        term_state: CartBondedTermState,
        coords_override: Optional[jax.Array] = None,
    ) -> Dict[ScoreType, jax.Array]:
        coords = pose_stack.coords if coords_override is None else coords_override
        # coords: [P, A, 3]
        n_poses = coords.shape[0]

        def _gather_xyz(global_atoms: jax.Array) -> jax.Array:
            """Gather coords by (pose, global_atom) indices.

            ``global_atoms`` has shape ``[P, B, K, k]`` with ``-1``
            sentinels in padding slots. We clamp the sentinel to 0 for
            the gather (the results are masked out downstream) and use
            ``jnp.take_along_axis`` along the per-pose atom axis.
            """
            safe = jnp.where(global_atoms >= 0, global_atoms, 0)
            # shape [P, B, K, k] -> flatten B*K*k for a single take per pose
            P, B, K, k = safe.shape
            flat = safe.reshape(P, B * K * k)  # int32[P, M]
            # coords: [P, A, 3]; gather along A.
            idx = flat[:, :, None].repeat(3, axis=-1)  # [P, M, 3]
            gathered = jnp.take_along_axis(coords, idx, axis=1)  # [P, M, 3]
            return gathered.reshape(P, B, K, k, 3)

        # Lengths --------------------------------------------------- #
        la = term_state.length_atoms  # [P, B, L, 2]
        lxyz = _gather_xyz(la)  # [P, B, L, 2, 3]
        l_d0 = term_state.length_params[..., 0]
        l_K = term_state.length_params[..., 1]
        l_d = distance(lxyz[..., 0, :], lxyz[..., 1, :])
        l_E = bond_length_E(l_d, l_d0, l_K)
        l_E = jnp.where(term_state.length_mask, l_E, 0.0)
        cart_lengths = jnp.sum(l_E, axis=(1, 2))  # [P]

        # Angles ---------------------------------------------------- #
        aa = term_state.angle_atoms  # [P, B, A, 3]
        axyz = _gather_xyz(aa)
        a_x0 = term_state.angle_params[..., 0]
        a_K = term_state.angle_params[..., 1]
        a_theta = pt_interior_angle(
            axyz[..., 0, :], axyz[..., 1, :], axyz[..., 2, :]
        )
        a_E = bond_angle_E(a_theta, a_x0, a_K)
        a_E = jnp.where(term_state.angle_mask, a_E, 0.0)
        cart_angles = jnp.sum(a_E, axis=(1, 2))

        # Sum-of-three-cosines torsion: ``hxltorsion_E`` form is shared by
        # proper torsions, impropers, hxl torsions, and inter-residue
        # torsions in upstream ``cbtorsion_V_dV``; only the score-type
        # bucket differs.
        def _cos3_score(atoms_field, params_field, mask_field):
            xyz = _gather_xyz(atoms_field)
            phi = dihedral_angle(
                xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :], xyz[..., 3, :]
            )
            E = hxltorsion_E(
                phi,
                params_field[..., 0], params_field[..., 1], params_field[..., 2],
                params_field[..., 3], params_field[..., 4], params_field[..., 5],
            )
            return jnp.sum(jnp.where(mask_field, E, 0.0), axis=(1, 2))

        cart_torsions = _cos3_score(
            term_state.torsion_atoms,
            term_state.torsion_params,
            term_state.torsion_mask,
        )
        cart_impropers = _cos3_score(
            term_state.improper_atoms,
            term_state.improper_params,
            term_state.improper_mask,
        )
        cart_hxltorsions = _cos3_score(
            term_state.hxltorsion_atoms,
            term_state.hxltorsion_params,
            term_state.hxltorsion_mask,
        )

        # ---------- Inter-residue contributions --------------------- #
        # Inter-residue arrays are stored as ``[P, E, k]`` (one entry
        # per pose per connection-spanning subgraph) rather than
        # ``[P, B, K, k]``. We gather coords with a 3D-flavoured
        # ``jnp.take_along_axis`` and reduce over the E axis.
        def _gather_xyz_3d(global_atoms: jax.Array) -> jax.Array:
            safe = jnp.where(global_atoms >= 0, global_atoms, 0)
            P, E, k = safe.shape
            flat = safe.reshape(P, E * k)
            idx = flat[:, :, None].repeat(3, axis=-1)
            gathered = jnp.take_along_axis(coords, idx, axis=1)
            return gathered.reshape(P, E, k, 3)

        # Inter-residue bond lengths ------------------------------- #
        ila = term_state.inter_length_atoms  # [P, E, 2]
        if ila.shape[1] > 0:
            ilxyz = _gather_xyz_3d(ila)  # [P, E, 2, 3]
            il_d0 = term_state.inter_length_params[..., 0]
            il_K = term_state.inter_length_params[..., 1]
            il_d = distance(ilxyz[..., 0, :], ilxyz[..., 1, :])
            il_E = bond_length_E(il_d, il_d0, il_K)
            il_E = jnp.where(term_state.inter_length_mask, il_E, 0.0)
            cart_lengths = cart_lengths + jnp.sum(il_E, axis=1)

        # Inter-residue bond angles -------------------------------- #
        iaa = term_state.inter_angle_atoms  # [P, E, 3]
        if iaa.shape[1] > 0:
            iaxyz = _gather_xyz_3d(iaa)
            ia_x0 = term_state.inter_angle_params[..., 0]
            ia_K = term_state.inter_angle_params[..., 1]
            ia_theta = pt_interior_angle(
                iaxyz[..., 0, :], iaxyz[..., 1, :], iaxyz[..., 2, :]
            )
            ia_E = bond_angle_E(ia_theta, ia_x0, ia_K)
            ia_E = jnp.where(term_state.inter_angle_mask, ia_E, 0.0)
            cart_angles = cart_angles + jnp.sum(ia_E, axis=1)

        # Inter-residue proper torsions ---------------------------- #
        ita = term_state.inter_torsion_atoms  # [P, E, 4]
        if ita.shape[1] > 0:
            itxyz = _gather_xyz_3d(ita)
            it_phi = dihedral_angle(
                itxyz[..., 0, :], itxyz[..., 1, :],
                itxyz[..., 2, :], itxyz[..., 3, :],
            )
            itp = term_state.inter_torsion_params
            it_E = hxltorsion_E(
                it_phi,
                itp[..., 0], itp[..., 1], itp[..., 2],
                itp[..., 3], itp[..., 4], itp[..., 5],
            )
            it_E = jnp.where(term_state.inter_torsion_mask, it_E, 0.0)
            cart_torsions = cart_torsions + jnp.sum(it_E, axis=1)

        return {
            ScoreType.cart_lengths: cart_lengths,
            ScoreType.cart_angles: cart_angles,
            ScoreType.cart_torsions: cart_torsions,
            ScoreType.cart_impropers: cart_impropers,
            ScoreType.cart_hxltorsions: cart_hxltorsions,
        }


__all__ = ["CartBondedEnergyTerm", "CartBondedTermState"]
