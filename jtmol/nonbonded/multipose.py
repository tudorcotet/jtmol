"""Multi-pose (P>1) nonbonded on the uncapped adaptive Verlet substrate.

Additive extension of the single-pose modules ``jtmol/nonbonded/{verlet,ljlk,
elec,lkball,hbond}.py`` to two batched-scoring workloads. Nothing here modifies
those modules or any jtmol core term; the per-pair physics is reused verbatim
(``lj_atr_rep``,
``lk_isotropic_pair``, ``elec_pair``, ``lk_ball_pair``, ``hbond_pair_V``,
``_place_all_waters``). Only the batching + edge-layout changes.

S2 -- SAME topology, P conformations (e.g. 1ubq x 8 perturbed coords)
----------------------------------------------------------------------
Topology, count-pair matrix and params are SHARED; only coords vary. We build ONE
EXACT-UNION Verlet list: a candidate (i<j) pair is kept if it is within
``cutoff+skin`` for ANY pose. Since each pose's own ``cutoff+skin`` candidate set
is a complete superset of that pose's within-cutoff set (proven in verlet.py), the
union is a complete superset for EVERY pose; per pose the too-far union edges score
exactly 0 (each term is 0 beyond its cutoff). The single-pose edge paths already
broadcast over the pose axis (``coords[:, edge_i, :]`` -> ``[P,E,3]``;
``edge_bpl[None]`` broadcasts), so feeding ``[P,N,3]`` coords + the union edges
evaluates all P poses in one call, returning each channel as ``[P]``.

S3 -- P DISTINCT topologies (e.g. a few different PDBs)
------------------------------------------------------
Each pose has its OWN atoms, count-pair and edges; nothing is shared. We RAGGED-
concatenate: all poses' real atoms into one flat ``[SumN,3]`` coord array, all
edges into one flat ``[E_total]`` list carrying GLOBAL (pose-offset) atom indices
and a per-edge pose id; then ``segment_sum`` per pose. NO maximum-size padding, NO
repeated-topology substitution (each pose is a genuinely distinct structure).

Channel coverage (honest):
  * S2: all 9 channels -- fa_ljatr/ljrep/lk, fa_elec, lk_ball_iso/ball/bridge/
        bridge_uncpl, hbond.
  * S3: all 9 channels -- LJLK+elec+hbond via the flat gather/segment_sum, lk_ball
        via a flat water-placement + segment_sum mirror of ``_score_all_pairs_edges``.

f64 ONLY for the exactness gate: ``jtmol.score.hbond`` forces ``jax_enable_x64`` at
import (degree-10 hbond polynomials need it), so importing this module turns x64 on.

Parity-exact with the single-pose modules (value + gradient); only the batching
and edge layout change.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import attrs
import jax

# hbond (imported below) forces x64; set explicitly so this module is self-sufficient.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jtmol.nonbonded.verlet import (  # noqa: E402
    VerletList,
    build_verlet_list,
    brute_within_cutoff_set,
    within_cutoff_set,
)
from jtmol.nonbonded.ljlk import setup_ljlk_on_verlet, ljlk_energy_on_verlet  # noqa: E402
from jtmol.nonbonded.elec import setup_elec_on_verlet, elec_energy_on_verlet  # noqa: E402
from jtmol.nonbonded.lkball import (  # noqa: E402
    _LKBALL_CUTOFF,
    setup_lkball_on_verlet,
    lkball_energy_on_verlet,
)
from jtmol.nonbonded.hbond import (  # noqa: E402
    HBondVerletState,
    setup_hbond_on_verlet,
    hbond_energy_on_verlet,
)
from jtmol.score.common import distance  # noqa: E402
from jtmol.score.ljlk.potentials import lj_atr_rep, lk_isotropic_pair  # noqa: E402
from jtmol.score.elec.potentials import elec_pair  # noqa: E402
from jtmol.score.hbond.potentials import hbond_pair_V  # noqa: E402
from jtmol.score.lk_ball.potentials import lk_ball_pair  # noqa: E402
from jtmol.score.score_types import ScoreType  # noqa: E402

# Full-atom pairwise cutoff shared by ljlk (max_dis 6.0) and lk_ball (6.0). elec is
# 0 beyond 5.5, so a union at 6.0 is a (harmless, zero-padded) superset for elec.
_FULL_CUTOFF = float(_LKBALL_CUTOFF)  # 6.0

# Per-atom LJLK/elec side params consumed by lj_atr_rep / lk_isotropic_pair.
_SIDE_FIELDS = (
    "lj_radius", "lj_wdepth", "is_donor", "is_acceptor",
    "is_hydroxyl", "is_polarh", "lk_volume", "lk_dgfree", "lk_lambda",
    "is_carbon_lk",
)

# Every lk_ball state field that _eval_full slices per pose (must be [P, ...] to
# vmap in_axes=0). The three neighbour sentinels are broadcast too (never read on
# the edge path, but still mapped).
_LKBALL_PERPOSE = (
    "lj_radius", "lk_dgfree", "lk_lambda", "lk_volume", "is_donor", "is_acceptor",
    "is_hydroxyl", "is_polarh", "is_carbon_lk", "is_polar", "is_occluder",
    "is_real", "B_idx",
    "B0_idx", "acc_hybridization", "attached_H_idx", "attached_H_mask",
    "n_attached_H", "bonded_path_length", "neighbor_idx", "neighbor_mask",
    "bpl_neighbor",
)
# lk_ball per-atom fields concatenated for the S3 ragged path.
_LKBALL_ATOM = _LKBALL_PERPOSE[:18]  # drop bonded_path_length + 3 nbr sentinels (rebuilt flat)
# lk_ball atom fields that hold ATOM INDICES (must be pose-offset in the concat).
_LKBALL_IDX = ("B_idx", "B0_idx", "attached_H_idx")


@attrs.define
class _Side:
    """Minimal per-atom parameter container for ``lj_atr_rep`` / ``lk_isotropic_pair``
    (same shape the single-pose ``_call_edge_list`` builds internally)."""

    lj_radius: Any
    lj_wdepth: Any
    is_donor: Any
    is_acceptor: Any
    is_hydroxyl: Any
    is_polarh: Any
    lk_volume: Any
    lk_dgfree: Any
    lk_lambda: Any
    is_carbon_lk: Any


class _LJLKGlobals:
    """The five ljlk global scalars ``lj_atr_rep`` / ``lk_isotropic_pair`` read."""

    def __init__(self, ls):
        self.lj_hbond_dis = ls.global_lj_hbond_dis
        self.lj_hbond_OH_donor_dis = ls.global_lj_hbond_OH_donor_dis
        self.lj_hbond_hdis = ls.global_lj_hbond_hdis
        self.max_dis = ls.global_max_dis
        self.lj_dlin_sigma_factor = ls.global_lj_dlin_sigma_factor


def _safe_dist(ci, cj):
    """NaN-safe pair distance (double-where), matching the single-pose modules."""
    diff = ci - cj
    d2 = jnp.sum(diff * diff, axis=-1)
    safe = d2 > 1e-6
    return jnp.where(safe, jnp.sqrt(jnp.where(safe, d2, 1.0)), 0.1)


# ========================================================================= #
# S2 -- same topology, P conformations, EXACT UNION Verlet list             #
# ========================================================================= #


def build_union_verlet_list(coords_PN3, cutoff, skin=1.0, atom_is_real=None) -> VerletList:
    """EXACT-UNION Verlet list over P conformations of one shared topology.

    A candidate ``(i<j)`` pair is kept if it lies within ``cutoff+skin`` for ANY
    of the P poses. Reuses :func:`build_verlet_list` per pose (hence the exact
    uncapped cell list) and unions the resulting pair sets. The union is a
    complete superset of every pose's within-cutoff set.

    ``coords_PN3``: ``[P,N,3]``. Returns a :class:`VerletList` whose ``pairs_i/j``
    are the union edges (ref_coords = pose 0's real coords).
    """
    coords = np.asarray(coords_PN3, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords_PN3 must be [P,N,3], got {coords.shape}")
    P = coords.shape[0]
    pair_set: set[tuple[int, int]] = set()
    ref: VerletList | None = None
    for p in range(P):
        vl_p = build_verlet_list(coords[p], cutoff, skin, atom_is_real=atom_is_real)
        pair_set.update(
            zip(np.asarray(vl_p.pairs_i).tolist(), np.asarray(vl_p.pairs_j).tolist())
        )
        ref = vl_p
    pairs = sorted(pair_set)  # deterministic order
    if pairs:
        pi = np.fromiter((a for a, _ in pairs), dtype=np.int32, count=len(pairs))
        pj = np.fromiter((b for _, b in pairs), dtype=np.int32, count=len(pairs))
    else:
        pi = pj = np.empty(0, dtype=np.int32)
    return VerletList(
        pairs_i=jnp.asarray(pi),
        pairs_j=jnp.asarray(pj),
        ref_coords=coords[0][ref.real_idx],
        real_idx=ref.real_idx,
        cutoff=float(cutoff),
        skin=float(skin),
    )


@dataclass
class S2Bundle:
    """Shared-topology multi-pose evaluation state (union edges swapped in)."""

    lt: Any; ls: Any   # ljlk term + state (union half-edges)
    et: Any; es: Any   # elec term + state (union half-edges)
    kt: Any; ks: Any   # lk_ball term + state (union directed edges, params tiled [P,N])
    hs: Any            # hbond state (union donor x acceptor edges)
    union: VerletList
    P: int
    N: int


def _bcast_P(x, P):
    return jnp.broadcast_to(x, (P,) + tuple(x.shape[1:]))


def _hbond_union_state(pose_stack, coords_PN3, param_db, skin) -> HBondVerletState:
    """Union of the per-pose hbond donor x acceptor edge sets (dedup by (H,A)).

    Reuses :func:`setup_hbond_on_verlet` VERBATIM per pose (each builds that pose's
    within-cutoff+skin donor-H x acceptor edges) and unions them. Params are
    topology-fixed, so any occurrence's params are identical -- we keep the first.
    """
    P = coords_PN3.shape[0]
    N = coords_PN3.shape[1]
    states = []
    for p in range(P):
        ps_p = pose_stack.replace(coords=jnp.asarray(coords_PN3[p:p + 1]))
        _, s_p, _, _ = setup_hbond_on_verlet(ps_p, param_db, skin=skin)
        states.append(s_p)

    def cat(attr):
        return np.concatenate([np.asarray(getattr(s, attr)) for s in states], axis=0)

    H = cat("edge_H").astype(np.int64)
    A = cat("edge_A").astype(np.int64)
    key = H * N + A
    _, uniq = np.unique(key, return_index=True)

    def take(attr):
        return jnp.asarray(cat(attr)[uniq])

    return HBondVerletState(
        edge_H=take("edge_H"), edge_D=take("edge_D"), edge_A=take("edge_A"),
        edge_B=take("edge_B"), edge_B0=take("edge_B0"), edge_hyb=take("edge_hyb"),
        edge_weight=take("edge_weight"), edge_ahdist=take("edge_ahdist"),
        edge_ahd=take("edge_ahd"), edge_bah=take("edge_bah"),
        edge_cp_sep=take("edge_cp_sep"),
        edge_pose=jnp.zeros(uniq.shape[0], dtype=jnp.int32),
        gp=states[0].gp, n_poses=1,
    )


def setup_s2(pose_stack, coords_PN3, param_db=None, skin=1.0) -> S2Bundle:
    """Build the shared-topology multi-pose bundle for ``coords_PN3`` ``[P,N,3]``.

    ``pose_stack`` is the single-pose (P==1) base topology; the P conformations
    ride in ``coords_PN3``. The single-pose setups supply params + count-pair
    matrices; we inject the exact-union edges.
    """
    if int(np.asarray(pose_stack.coords).shape[0]) != 1:
        raise ValueError("setup_s2 expects a single-pose base pose_stack (P==1)")
    coords = np.asarray(coords_PN3, dtype=np.float64)
    P, N, _ = coords.shape

    lt, ls, ldense, _ = setup_ljlk_on_verlet(pose_stack, param_db, skin=skin)
    et, es, edense, _ = setup_elec_on_verlet(pose_stack, param_db, skin=skin)
    kt, ks, kdense, _ = setup_lkball_on_verlet(pose_stack, param_db, skin=skin)

    atom_is_real = np.asarray(ldense.atom_is_real[0], dtype=bool)
    union = build_union_verlet_list(coords, cutoff=_FULL_CUTOFF, skin=skin, atom_is_real=atom_is_real)
    uvi = np.asarray(union.pairs_i)
    uvj = np.asarray(union.pairs_j)

    # ljlk / elec: swap the union half-edges (i<j) + re-gather each term's own
    # count-pair matrix at those pairs. Params [1,N] broadcast over the pose axis.
    lj_bpl = np.asarray(ldense.bonded_path_length[0])[uvi, uvj]
    ls = attrs.evolve(
        ls, edge_i=jnp.asarray(uvi.astype(np.int32)),
        edge_j=jnp.asarray(uvj.astype(np.int32)), edge_bpl=jnp.asarray(lj_bpl),
    )
    el_bpl = np.asarray(edense.bonded_path_length[0])[uvi, uvj]
    es = attrs.evolve(
        es, edge_i=jnp.asarray(uvi.astype(np.int32)),
        edge_j=jnp.asarray(uvj.astype(np.int32)), edge_bpl=jnp.asarray(el_bpl),
    )

    # lk_ball: EXPAND the union half-list -> directed i->j list, tile params to [P,N]
    # (_eval_full vmaps per pose in_axes=0; the shared topology's params are the
    # same for all P conformations -- legitimate, not a fake-distinct substitution).
    ei = np.concatenate([uvi, uvj]).astype(np.int32)
    ej = np.concatenate([uvj, uvi]).astype(np.int32)
    k_bpl = np.asarray(kdense.bonded_path_length[0])[ei, ej]
    upd = {f: _bcast_P(getattr(ks, f), P) for f in _LKBALL_PERPOSE}
    upd.update(
        edge_i=jnp.asarray(ei), edge_j=jnp.asarray(ej), edge_bpl=jnp.asarray(k_bpl),
    )
    ks = attrs.evolve(ks, **upd)

    hs = _hbond_union_state(pose_stack, coords, param_db, skin)
    return S2Bundle(lt, ls, et, es, kt, ks, hs, union, P, N)


def _hbond_multipose_energy(state: HBondVerletState, coords_PN3):
    """hbond over shared donor x acceptor edges for ``[P,N,3]`` coords -> ``[P]``.

    Mirrors :func:`hbond_energy_on_verlet` but with a leading pose axis. Flattens
    ``[P,E]`` -> ``[P*E]`` so ``hbond_pair_V`` sees exactly the rank-1 layout the
    single-pose path feeds it, then reshapes + sums per pose.
    """
    c = jnp.asarray(coords_PN3)
    P = c.shape[0]
    E = int(state.edge_H.shape[0])

    def gather(idx):
        return c[:, idx, :]  # [P, E, 3]

    D, H, A = gather(state.edge_D), gather(state.edge_H), gather(state.edge_A)
    B, B0 = gather(state.edge_B), gather(state.edge_B0)

    flat3 = lambda x: x.reshape(P * E, 3)
    tile1 = lambda x: jnp.broadcast_to(x[None], (P, E)).reshape(P * E)
    tile2 = lambda x: jnp.broadcast_to(x[None], (P, E) + x.shape[1:]).reshape((P * E,) + x.shape[1:])

    Ev = hbond_pair_V(
        flat3(D), flat3(H), flat3(A), flat3(B), flat3(B0),
        tile1(state.edge_hyb), tile1(state.edge_weight),
        tile2(state.edge_ahdist), tile2(state.edge_ahd), tile2(state.edge_bah),
        state.gp,
    ).reshape(P, E)
    ah = distance(A, H)  # [P, E]
    mask = (ah < state.gp.max_ah_dist) & (state.edge_cp_sep[None] >= 5)
    Ev = jnp.where(mask, Ev, jnp.zeros_like(Ev))
    return {ScoreType.hbond: Ev.sum(axis=-1)}


def s2_energies(bundle: S2Bundle, coords_PN3) -> dict:
    """All 9 nonbonded channels for ``coords_PN3`` ``[P,N,3]``, each as ``[P]``.

    Reuses the single-pose edge paths with the union edges. Differentiable in
    ``coords_PN3``.
    """
    c = jnp.asarray(coords_PN3)
    out: dict = {}
    out.update(bundle.lt._call_edge_list(c, bundle.ls))    # ljlk 3ch [P]
    out.update(bundle.et._call_edge_list(c, bundle.es))    # elec [P]
    out.update(bundle.kt._eval_full(c, bundle.ks))         # lk_ball 4ch [P]
    out.update(_hbond_multipose_energy(bundle.hs, c))      # hbond [P]
    return out


# ========================================================================= #
# S3 -- P distinct topologies, RAGGED concatenation, segment_sum per pose   #
# ========================================================================= #


@dataclass
class S3Bundle:
    """Ragged multi-topology evaluation state.

    ``per`` holds each pose's single-pose ``(term, state)`` tuples (used both as
    the reference and as the source of per-pose edges/params). The ``*_edge_*``
    arrays are the flat, pose-offset concatenations.
    """

    per: list          # [{lt,ls,et,es,kt,ks,ht,hs, N, offset}]
    P: int
    SumN: int
    offsets: np.ndarray
    coords_flat0: np.ndarray

    # ljlk
    lt: Any; lj: dict; lj_G: _LJLKGlobals
    lj_edge_i: Any; lj_edge_j: Any; lj_edge_bpl: Any; lj_edge_pose: Any
    # elec
    et: Any; charges: Any; el_globals: Any
    el_edge_i: Any; el_edge_j: Any; el_edge_bpl: Any; el_edge_pose: Any
    # hbond (concatenated state -> hbond_energy_on_verlet already segment_sums)
    hs_cat: HBondVerletState
    # lk_ball
    kt: Any; lk: dict
    lk_edge_i: Any; lk_edge_j: Any; lk_edge_bpl: Any; lk_edge_pose: Any


def _perpose_setups(pose_stacks, param_db, skin):
    per = []
    offset = 0
    offsets = []
    coords_list = []
    for ps_p in pose_stacks:
        if int(np.asarray(ps_p.coords).shape[0]) != 1:
            raise ValueError("each S3 pose_stack must be single-pose (P==1)")
        lt, ls, ld, _ = setup_ljlk_on_verlet(ps_p, param_db, skin=skin)
        et, es, ed, _ = setup_elec_on_verlet(ps_p, param_db, skin=skin)
        kt, ks, kd, _ = setup_lkball_on_verlet(ps_p, param_db, skin=skin)
        ht, hs, hd, _ = setup_hbond_on_verlet(ps_p, param_db, skin=skin)
        N_p = int(np.asarray(ps_p.coords).shape[1])
        per.append(dict(ps=ps_p, lt=lt, ls=ls, ld=ld, et=et, es=es, ed=ed,
                        kt=kt, ks=ks, kd=kd, ht=ht, hs=hs, hd=hd,
                        N=N_p, offset=offset))
        offsets.append(offset)
        offset += N_p
        coords_list.append(np.asarray(ps_p.coords[0], dtype=np.float64))
    return per, np.asarray(offsets, dtype=np.int64), offset, np.concatenate(coords_list, axis=0)


def setup_s3(pose_stacks, param_db=None, skin=1.0) -> S3Bundle:
    """Build the ragged bundle for a list of DISTINCT single-pose pose_stacks."""
    per, offsets, SumN, coords_flat0 = _perpose_setups(pose_stacks, param_db, skin)
    P = len(per)

    def cat_atom(states_key, field):
        return np.concatenate(
            [np.asarray(getattr(e[states_key], field))[0] for e in per], axis=0
        )

    def cat_edge(states_key, field, offset=True):
        parts = []
        for e in per:
            a = np.asarray(getattr(e[states_key], field))
            parts.append(a + (e["offset"] if offset else 0))
        return np.concatenate(parts, axis=0)

    def edge_pose(states_key, ref_field="edge_i"):
        return np.concatenate(
            [np.full(int(np.asarray(getattr(e[states_key], ref_field)).shape[0]), p, np.int32)
             for p, e in enumerate(per)], axis=0
        )

    # ---- ljlk ----
    lj = {f: jnp.asarray(cat_atom("ls", f)) for f in _SIDE_FIELDS}
    lj_edge_i = jnp.asarray(cat_edge("ls", "edge_i").astype(np.int32))
    lj_edge_j = jnp.asarray(cat_edge("ls", "edge_j").astype(np.int32))
    lj_edge_bpl = jnp.asarray(cat_edge("ls", "edge_bpl", offset=False))
    lj_edge_pose = jnp.asarray(edge_pose("ls"))
    lj_G = _LJLKGlobals(per[0]["ls"])

    # ---- elec ----
    charges = jnp.asarray(cat_atom("es", "charges"))
    el_edge_i = jnp.asarray(cat_edge("es", "edge_i").astype(np.int32))
    el_edge_j = jnp.asarray(cat_edge("es", "edge_j").astype(np.int32))
    el_edge_bpl = jnp.asarray(cat_edge("es", "edge_bpl", offset=False))
    el_edge_pose = jnp.asarray(edge_pose("es"))

    # ---- hbond: concat per-pose donor x acceptor edges, pose-offset atom idx ----
    def hcat(field, is_idx):
        parts = []
        for e in per:
            a = np.asarray(getattr(e["hs"], field))
            if is_idx:  # pose-offset atom indices; preserve any -1 sentinel (no cross-pose leak)
                a = np.where(a >= 0, a + e["offset"], a)
            parts.append(a)
        return jnp.asarray(np.concatenate(parts, axis=0))

    hs_cat = HBondVerletState(
        edge_H=hcat("edge_H", True), edge_D=hcat("edge_D", True), edge_A=hcat("edge_A", True),
        edge_B=hcat("edge_B", True), edge_B0=hcat("edge_B0", True),
        edge_hyb=hcat("edge_hyb", False), edge_weight=hcat("edge_weight", False),
        edge_ahdist=hcat("edge_ahdist", False), edge_ahd=hcat("edge_ahd", False),
        edge_bah=hcat("edge_bah", False), edge_cp_sep=hcat("edge_cp_sep", False),
        edge_pose=jnp.asarray(edge_pose("hs", ref_field="edge_H")),
        gp=per[0]["hs"].gp, n_poses=P,
    )

    # ---- lk_ball: concat per-atom fields (offset index fields), directed edges ----
    lk = {}
    for f in _LKBALL_ATOM:
        parts = []
        for e in per:
            a = np.asarray(getattr(e["ks"], f))[0]
            if f in _LKBALL_IDX:
                a = np.where(a >= 0, a + e["offset"], a)
            parts.append(a)
        lk[f] = jnp.asarray(np.concatenate(parts, axis=0))
    lk_edge_i = jnp.asarray(cat_edge("ks", "edge_i").astype(np.int32))
    lk_edge_j = jnp.asarray(cat_edge("ks", "edge_j").astype(np.int32))
    lk_edge_bpl = jnp.asarray(cat_edge("ks", "edge_bpl", offset=False))
    lk_edge_pose = jnp.asarray(edge_pose("ks"))

    return S3Bundle(
        per=per, P=P, SumN=SumN, offsets=offsets, coords_flat0=coords_flat0,
        lt=per[0]["lt"], lj=lj, lj_G=lj_G,
        lj_edge_i=lj_edge_i, lj_edge_j=lj_edge_j, lj_edge_bpl=lj_edge_bpl, lj_edge_pose=lj_edge_pose,
        et=per[0]["et"], charges=charges, el_globals=per[0]["es"],
        el_edge_i=el_edge_i, el_edge_j=el_edge_j, el_edge_bpl=el_edge_bpl, el_edge_pose=el_edge_pose,
        hs_cat=hs_cat,
        kt=per[0]["kt"], lk=lk,
        lk_edge_i=lk_edge_i, lk_edge_j=lk_edge_j, lk_edge_bpl=lk_edge_bpl, lk_edge_pose=lk_edge_pose,
    )


def _ljlk_ragged(b: S3Bundle, c):
    ei, ej = b.lj_edge_i, b.lj_edge_j
    dist = _safe_dist(c[ei], c[ej])
    ip = _Side(**{f: b.lj[f][ei] for f in _SIDE_FIELDS})
    jp = _Side(**{f: b.lj[f][ej] for f in _SIDE_FIELDS})
    Vatr, Vrep = lj_atr_rep(dist, b.lj_edge_bpl, ip, jp, b.lj_G, soft_core_cap=b.lt.soft_core_rep_cap)
    Vlk = lk_isotropic_pair(dist, b.lj_edge_bpl, ip, jp, b.lj_G)
    ss = lambda x: jax.ops.segment_sum(x, b.lj_edge_pose, num_segments=b.P)
    return {ScoreType.fa_ljatr: ss(Vatr), ScoreType.fa_ljrep: ss(Vrep), ScoreType.fa_lk: ss(Vlk)}


def _elec_ragged(b: S3Bundle, c):
    ei, ej = b.el_edge_i, b.el_edge_j
    dist = _safe_dist(c[ei], c[ej])
    E = elec_pair(dist, b.charges[ei], b.charges[ej], b.el_edge_bpl, b.el_globals)
    return {ScoreType.fa_elec: jax.ops.segment_sum(E, b.el_edge_pose, num_segments=b.P)}


def _lkball_ragged(b: S3Bundle, c):
    """lk_ball 4 channels ragged: place waters on the flat atom array (per-atom,
    pose-offset base/H indices keep it within-pose), score the flat directed edges
    with ``lk_ball_pair`` VERBATIM, ``segment_sum`` per pose. Mirror of
    :meth:`LKBallEnergyTerm._score_all_pairs_edges` with the sum -> segment_sum."""
    kt = b.kt
    lk = b.lk
    waters = kt._place_all_waters(
        c, lk["is_donor"], lk["is_acceptor"], lk["is_polar"],
        lk["B_idx"], lk["B0_idx"], lk["acc_hybridization"],
        lk["attached_H_idx"], lk["attached_H_mask"], lk["n_attached_H"],
    )  # [SumN, n_waters, 3]
    ei, ej = b.lk_edge_i, b.lk_edge_j
    ci, cj = c[ei], c[ej]
    wi, wj = waters[ei], waters[ej]
    params = {
        "i_lj_radius": lk["lj_radius"][ei], "j_lj_radius": lk["lj_radius"][ej],
        "i_lk_dgfree": lk["lk_dgfree"][ei], "i_lk_lambda": lk["lk_lambda"][ei],
        "j_lk_volume": lk["lk_volume"][ej],
        "i_is_donor": lk["is_donor"][ei], "j_is_donor": lk["is_donor"][ej],
        "i_is_hydroxyl": lk["is_hydroxyl"][ei], "j_is_hydroxyl": lk["is_hydroxyl"][ej],
        "i_is_polarh": lk["is_polarh"][ei], "j_is_polarh": lk["is_polarh"][ej],
        "i_is_acceptor": lk["is_acceptor"][ei], "j_is_acceptor": lk["is_acceptor"][ej],
        "i_is_carbon_lk": lk["is_carbon_lk"][ei], "j_is_carbon_lk": lk["is_carbon_lk"][ej],
    }

    def pair_at(coord_i, waters_i, coord_j, waters_j, prm, bpl):
        res = lk_ball_pair(
            coord_i=coord_i, waters_i=waters_i, coord_j=coord_j, waters_j=waters_j,
            params=prm, defaults=kt.defaults, bonded_path_length=bpl,
        )
        return jnp.stack([res.lk_iso, res.lk_ball, res.lk_bridge, res.lk_bridge_uncpl])

    per_pair = jax.vmap(pair_at)(ci, wi, cj, wj, params, b.lk_edge_bpl)  # [E, 4]
    gate = (
        lk["is_polar"][ei] & lk["is_occluder"][ej] & lk["is_real"][ei] & lk["is_real"][ej]
    ).astype(jnp.float32)[:, None]
    contrib = per_pair * gate  # [E, 4]
    tot = jnp.stack(
        [jax.ops.segment_sum(contrib[:, k], b.lk_edge_pose, num_segments=b.P) for k in range(4)],
        axis=1,
    )  # [P, 4]
    return {
        ScoreType.lk_ball_iso: tot[:, 0], ScoreType.lk_ball: tot[:, 1],
        ScoreType.lk_bridge: tot[:, 2], ScoreType.lk_bridge_uncpl: tot[:, 3],
    }


def s3_energies(bundle: S3Bundle, coords_flat) -> dict:
    """All 9 nonbonded channels for the ragged flat ``[SumN,3]`` coords, each as
    ``[P]``. Differentiable in ``coords_flat``."""
    c = jnp.asarray(coords_flat)
    out: dict = {}
    out.update(_ljlk_ragged(bundle, c))
    out.update(_elec_ragged(bundle, c))
    out.update(hbond_energy_on_verlet(bundle.hs_cat, c))  # already segment_sums by pose
    out.update(_lkball_ragged(bundle, c))
    return out


# ========================================================================= #
# Per-pose reference (loop the single-pose modules, stack) + assert demo     #
# ========================================================================= #

_ALL9 = (
    ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk, ScoreType.fa_elec,
    ScoreType.lk_ball_iso, ScoreType.lk_ball, ScoreType.lk_bridge,
    ScoreType.lk_bridge_uncpl, ScoreType.hbond,
)


def _eval_single_pose(lt, ls, et, es, kt, ks, hs, coords_1N3) -> dict:
    r: dict = {}
    r.update(ljlk_energy_on_verlet(lt, ls, coords_1N3))
    r.update(elec_energy_on_verlet(et, es, coords_1N3))
    r.update(lkball_energy_on_verlet(kt, ks, coords_1N3))
    r.update(hbond_energy_on_verlet(hs, coords_1N3))
    return r


def _eval_dense_pose(lt, ld, et, ed, kt, kd, ht, hd, ps_p, coords_1N3) -> dict:
    """INDEPENDENT dense [N,N] / [D,A] brute path (no edge list at all).

    The non-circular gate: the single-pose edge modules were each verified ==
    this dense path in their own demos, so agreement here proves the union/ragged
    edge construction reproduces brute force, not just another edge sum."""
    r: dict = {}
    r.update(lt(ps_p, ld, coords_override=coords_1N3))
    r.update(et(ps_p, ed, coords_override=coords_1N3))
    r.update(kt(ps_p, kd, coords_override=coords_1N3))
    r.update(ht(ps_p, hd, coords_override=coords_1N3))
    return r


def _s2_reference(pose_stack, coords_PN3, param_db, skin) -> tuple[dict, dict]:
    """Loop the single-pose modules pose-by-pose (each pose's OWN verlet edges).

    Returns (edge_ref, dense_ref): the single-pose EDGE result and the INDEPENDENT
    dense [N,N] brute result, both stacked over poses."""
    P = coords_PN3.shape[0]
    edge = defaultdict(list)
    dense = defaultdict(list)
    for p in range(P):
        cp = jnp.asarray(coords_PN3[p:p + 1])
        ps_p = pose_stack.replace(coords=cp)
        lt, ls, ld, _ = setup_ljlk_on_verlet(ps_p, param_db, skin=skin)
        et, es, ed, _ = setup_elec_on_verlet(ps_p, param_db, skin=skin)
        kt, ks, kd, _ = setup_lkball_on_verlet(ps_p, param_db, skin=skin)
        ht, hs, hd, _ = setup_hbond_on_verlet(ps_p, param_db, skin=skin)
        re = _eval_single_pose(lt, ls, et, es, kt, ks, hs, cp)
        rd = _eval_dense_pose(lt, ld, et, ed, kt, kd, ht, hd, ps_p, cp)
        for st in _ALL9:
            edge[st].append(float(np.asarray(re[st]).reshape(-1)[0]))
            dense[st].append(float(np.asarray(rd[st]).reshape(-1)[0]))
    return ({st: np.asarray(edge[st]) for st in _ALL9},
            {st: np.asarray(dense[st]) for st in _ALL9})


def _s3_reference(bundle: S3Bundle) -> tuple[dict, dict]:
    """Loop the stored per-pose single-pose modules at their own coords, stack.

    Returns (edge_ref, dense_ref)."""
    edge = defaultdict(list)
    dense = defaultdict(list)
    for e in bundle.per:
        cp = jnp.asarray(np.asarray(e["ps"].coords[0], dtype=np.float64))[None]
        re = _eval_single_pose(e["lt"], e["ls"], e["et"], e["es"], e["kt"], e["ks"], e["hs"], cp)
        rd = _eval_dense_pose(e["lt"], e["ld"], e["et"], e["ed"], e["kt"], e["kd"],
                              e["ht"], e["hd"], e["ps"], cp)
        for st in _ALL9:
            edge[st].append(float(np.asarray(re[st]).reshape(-1)[0]))
            dense[st].append(float(np.asarray(rd[st]).reshape(-1)[0]))
    return ({st: np.asarray(edge[st]) for st in _ALL9},
            {st: np.asarray(dense[st]) for st in _ALL9})


def _assert_union_complete(union: VerletList, coords_PN3, real_idx):
    """The union list == each pose's own within-cutoff set (complete, per pose)."""
    coords = np.asarray(coords_PN3, dtype=np.float64)
    for p in range(coords.shape[0]):
        got = within_cutoff_set(union, coords[p])
        ref = brute_within_cutoff_set(coords[p], union.cutoff, real_idx=real_idx)
        assert got == ref, (
            f"union INCOMPLETE for pose {p}: missing {len(ref - got)}, extra {len(got - ref)}"
        )
    return True


# dense [N,N] brute vs edge-list differ only by float reduction order: ~1e-8 for
# the pairwise channels, ~1e-4 for hbond (degree-10 f64 polynomials -- see hbond.py).
_DENSE_TOL = defaultdict(lambda: 1e-6, {ScoreType.hbond: 2e-4})


def _report(tag, mp, ref, show_E=True):
    worst_e = 0.0
    for st in _ALL9:
        m = np.asarray(mp[st], dtype=np.float64)
        r = np.asarray(ref[st], dtype=np.float64)
        rel = np.abs(m - r) / (np.abs(r) + 1e-12)
        worst_e = max(worst_e, float(rel.max()))
        extra = (f"  (per-pose E: {np.array2string(r, precision=3, floatmode='maxprec')})"
                 if show_E else "")
        print(f"    {st.name:16}: max rel {float(rel.max()):.2e}{extra}")
    print(f"  [{tag}] worst channel rel = {worst_e:.2e}")
    return worst_e


def _assert_per_channel(mp, ref, tol_map):
    for st in _ALL9:
        m = np.asarray(mp[st], dtype=np.float64)
        r = np.asarray(ref[st], dtype=np.float64)
        rel = np.abs(m - r) / (np.abs(r) + 1e-12)
        assert float(rel.max()) < tol_map[st], (
            f"{st.name} parity FAILED: rel {float(rel.max()):.2e} >= {tol_map[st]:.2e}"
        )


def demo() -> None:
    """Assert S2 (union) and S3 (ragged) == per-pose single-pose result, f64.

    S2: P=8 perturbed conformations of 1ubq. Union list confirmed COMPLETE per pose
        (== each pose's own within-cutoff set); all 9 channels + gradient match.
    S3: distinct structures (1ubq, 1qys, 2OOB); all 9 channels + gradient match the
        per-pose single-pose modules, no pose dropped/mismatched.
    """
    import os
    jax.config.update("jax_enable_x64", True)
    import jtmol
    from jtmol.database import ParameterDatabase

    data = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data")
    db = ParameterDatabase.get_default()

    # ------------------------------------------------------------------ S2 #
    print("=" * 74)
    print("S2 -- same topology (1ubq), P=8 perturbed conformations, EXACT UNION")
    print("=" * 74)
    ps = jtmol.pose_stack_from_pdb(os.path.join(data, "1ubq.pdb"))
    coords0 = np.asarray(ps.coords[0], dtype=np.float64)
    N = coords0.shape[0]
    P = 8
    rng = np.random.default_rng(0)
    coords_PN3 = np.stack(
        [coords0 + rng.normal(size=(N, 3)) * 0.4 for _ in range(P)], axis=0
    )  # 8 genuinely distinct conformations

    real_idx = np.arange(N, dtype=np.int64)  # 1ubq all-real
    bundle = setup_s2(ps, coords_PN3, db)
    n_union = int(bundle.union.pairs_i.size)
    print(f"[union] {P} poses, N={N} -> {n_union} union candidate pairs "
          f"(cutoff={_FULL_CUTOFF}, skin=1.0)")
    _assert_union_complete(bundle.union, coords_PN3, real_idx)
    print(f"[union] COMPLETE: for every pose, union within-cutoff set == brute force")

    mp = s2_energies(bundle, coords_PN3)
    edge_ref, dense_ref = _s2_reference(ps, coords_PN3, db, skin=1.0)
    print("  --- S2 vs per-pose single-pose EDGE modules (same physics, +0 union edges) ---")
    worst = _report("S2 edge", mp, edge_ref)
    assert worst < 1e-6, f"S2 edge parity FAILED: worst rel {worst}"
    print("  --- S2 vs INDEPENDENT dense [N,N]/[D,A] brute path (non-circular gate) ---")
    _report("S2 dense", mp, dense_ref, show_E=False)
    _assert_per_channel(mp, dense_ref, _DENSE_TOL)

    def _s2_total(c):
        e = s2_energies(bundle, c)
        return sum(jnp.sum(e[st]) for st in _ALL9)

    g_mp = np.asarray(jax.grad(_s2_total)(jnp.asarray(coords_PN3)))  # [P,N,3]
    gmax = 0.0
    for p in range(P):
        cp = jnp.asarray(coords_PN3[p:p + 1])
        ps_p = ps.replace(coords=cp)
        lt, ls, _, _ = setup_ljlk_on_verlet(ps_p, db)
        et, es, _, _ = setup_elec_on_verlet(ps_p, db)
        kt, ks, _, _ = setup_lkball_on_verlet(ps_p, db)
        ht, hs, _, _ = setup_hbond_on_verlet(ps_p, db)
        g_ref = np.asarray(jax.grad(
            lambda c: sum(jnp.sum(v) for v in _eval_single_pose(lt, ls, et, es, kt, ks, hs, c).values())
        )(cp))[0]
        gmax = max(gmax, float(np.abs(g_mp[p] - g_ref).max()))
    print(f"  [S2] gradient max|multipose - perpose| = {gmax:.2e}")
    assert gmax < 1e-5, f"S2 gradient parity FAILED: {gmax}"
    print("  S2 PASS: all 9 channels + gradient == per-pose single-pose, f64.\n")

    # ------------------------------------------------------------------ S3 #
    print("=" * 74)
    print("S3 -- P DISTINCT topologies, RAGGED concat, segment_sum per pose")
    print("=" * 74)
    names = ["1ubq", "1qys", "2OOB"]
    pose_stacks = []
    loaded = []
    for nm in names:
        try:
            pose_stacks.append(jtmol.pose_stack_from_pdb(os.path.join(data, f"{nm}.pdb")))
            loaded.append(nm)
        except Exception as exc:  # noqa: BLE001
            print(f"  (skip {nm}: {type(exc).__name__})")
    assert len(pose_stacks) >= 2, "need >=2 distinct structures for S3"
    sizes = [int(np.asarray(p.coords).shape[1]) for p in pose_stacks]
    print(f"[ragged] {len(loaded)} distinct topologies {loaded}, atoms {sizes} "
          f"-> SumN={sum(sizes)} (no padding)")

    b3 = setup_s3(pose_stacks, db)
    coords_flat = jnp.asarray(b3.coords_flat0)
    mp3 = s3_energies(b3, coords_flat)
    edge_ref3, dense_ref3 = _s3_reference(b3)
    # confirm segment sizes: no pose dropped
    for st in _ALL9:
        assert np.asarray(mp3[st]).shape == (b3.P,), f"{st.name} wrong pose count"
    print("  --- S3 vs per-pose single-pose EDGE modules (stacked) ---")
    worst3 = _report("S3 edge", mp3, edge_ref3)
    assert worst3 < 1e-6, f"S3 edge parity FAILED: worst rel {worst3}"
    print("  --- S3 vs INDEPENDENT dense [N,N]/[D,A] brute path (non-circular gate) ---")
    _report("S3 dense", mp3, dense_ref3, show_E=False)
    _assert_per_channel(mp3, dense_ref3, _DENSE_TOL)

    def _s3_total(c):
        e = s3_energies(b3, c)
        return sum(jnp.sum(e[st]) for st in _ALL9)

    g3 = np.asarray(jax.grad(_s3_total)(coords_flat))  # [SumN,3]
    gmax3 = 0.0
    for e in b3.per:
        off, Np = e["offset"], e["N"]
        cp = jnp.asarray(np.asarray(e["ps"].coords[0], dtype=np.float64))[None]
        g_ref = np.asarray(jax.grad(
            lambda c: sum(jnp.sum(v) for v in _eval_single_pose(
                e["lt"], e["ls"], e["et"], e["es"], e["kt"], e["ks"], e["hs"], c).values())
        )(cp))[0]
        gmax3 = max(gmax3, float(np.abs(g3[off:off + Np] - g_ref).max()))
    print(f"  [S3] gradient max|ragged - perpose| = {gmax3:.2e}")
    assert gmax3 < 1e-5, f"S3 gradient parity FAILED: {gmax3}"
    print("  S3 PASS: all 9 channels + gradient == per-pose single-pose, f64.\n")

    print("ALL CHECKS PASSED -- multi-pose nonbonded (S2 union + S3 ragged) == "
          "per-pose dense single-pose, energy + gradient, f64. Physics reused "
          "verbatim; only batching/edge-layout changed. No timing, no speed claims.")


if __name__ == "__main__":
    demo()
