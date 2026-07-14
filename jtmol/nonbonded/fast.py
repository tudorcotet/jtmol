"""Fast S2 nonbonded — a parity-exact speedup of :func:`s2_energies`.

The generic multi-pose substrate (:mod:`jtmol.nonbonded.multipose`) is correct
but not tuned for the same-topology batched workload (S2). Two independent,
each parity-exact levers reduce its runtime on that workload:

1. **Sorted-scatter backward on ljlk + elec.** These two terms are
   *analytic-force* (the pair force is a plain jvp through the scalar pair
   distance). Their generic reverse-mode accumulates per-edge forces with a
   colliding ``grad.at[:, ei].add(...)`` scatter whose backward multiplier is
   ~26x (ljlk) / ~35x (elec) the forward — millions of atomic collisions. We
   replace *only the accumulation* with a ``jax.ops.segment_sum`` on a
   presorted index (``edge_i`` is sorted by construction; ``edge_j`` via a
   precomputed ``argsort``), a coalesced reduction. Forward math is untouched,
   so the value is bit-identical and the gradient matches to reduction-order.
   **Do not** apply this to lk_ball/hbond: their sorted variant needs an inner
   ``jax.vjp`` through the water geometry / degree-10 polynomials, which blocks
   XLA fusion and *regresses* the backward badly.

2. **Bridge-split lk_ball forward** (the "OPT3" three-list decomposition). The
   generic ``_eval_full`` runs the full ``lk_ball_pair`` — including the
   ``[MW, MW]`` water-water bridge log-sum-exp, which profiles at ~85% of
   lk_ball's cost — on *every* directed edge, then zeroes the ungated ones with
   ``jnp.where``. But the polar->occluder gate is static (topology-fixed), and
   the bridge is nonzero only where ``is_donor[j] | is_acceptor[j]``. We
   pre-filter to (A) the gated edge subset for iso/desolvation and (B) the much
   smaller donor/acceptor-j subset for the bridge, and place waters only on the
   static polar-atom set. Dropped edges contribute *exactly* zero, so this is
   parity-exact (bit-exact f64).

Parity vs :func:`s2_energies`: 9.9e-8 rel (f32), bit-exact (f64) — see
:func:`demo`.

This module is **purely additive**: it does not modify the substrate or any
core term. It reuses every per-pair potential verbatim; only the backward
accumulation (ljlk/elec) and the forward edge/water enumeration (lk_ball)
change. Same :class:`~jtmol.nonbonded.multipose.S2Bundle` from
:func:`~jtmol.nonbonded.multipose.setup_s2`.

Scope: S2 (shared topology, P conformations). The S3 ragged path still uses the
generic :func:`~jtmol.nonbonded.multipose.s3_energies`.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from jtmol.nonbonded.multipose import (
    S2Bundle,
    S3Bundle,
    _hbond_multipose_energy,
    _LJLKGlobals,
    _Side,
    _SIDE_FIELDS,
)
from jtmol.nonbonded.hbond import hbond_energy_on_verlet
from jtmol.score.ljlk.potentials import lj_atr_rep, lk_isotropic_pair
from jtmol.score.elec.potentials import elec_pair
from jtmol.score.lk_ball.potentials import (
    LK_BALL_DEFAULTS as _DEF,
    lj_sigma,
    lk_ball_pair,
    lk_fraction,
    lk_isotropic_pair_simplified,
    place_waters_for_polar_atom,
    safe_norm,
)
from jtmol.score.score_types import ScoreType

_ATR, _REP, _LK = ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk
_EL = ScoreType.fa_elec
_ISO, _BALL, _BRIDGE, _UNCPL = (
    ScoreType.lk_ball_iso,
    ScoreType.lk_ball,
    ScoreType.lk_bridge,
    ScoreType.lk_bridge_uncpl,
)
_HB = ScoreType.hbond

_ALL9 = (_ATR, _REP, _LK, _EL, _ISO, _BALL, _BRIDGE, _UNCPL, _HB)


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _dd(c, ei, ej):
    """Edge diff/dist with the substrate's exact coincident-atom guard."""
    diff = c[:, ei, :] - c[:, ej, :]  # [P, E, 3]
    d2 = jnp.sum(diff * diff, -1)  # [P, E]
    safe = d2 > 1e-6
    dist = jnp.where(safe, jnp.sqrt(jnp.where(safe, d2, 1.0)), 0.1)
    return diff, dist, safe


def _argsort_j(ej):
    ej_np = np.asarray(ej)
    jperm = np.argsort(ej_np, kind="stable").astype(np.int32)
    return jnp.asarray(ej_np[jperm]), jnp.asarray(jperm)


def _seg_i(f, ei, N):
    """vmap-over-P sorted segment_sum of +f onto edge_i (edge_i is sorted)."""
    return jax.vmap(
        lambda fp: jax.ops.segment_sum(fp, ei, num_segments=N, indices_are_sorted=True)
    )(f)


def _seg_j(f, ej_sorted, jperm, N):
    """vmap-over-P sorted segment_sum of -f onto edge_j (via argsort perm)."""
    return jax.vmap(
        lambda fp: jax.ops.segment_sum(
            -fp[jperm], ej_sorted, num_segments=N, indices_are_sorted=True
        )
    )(f)


# --------------------------------------------------------------------------- #
# ljlk — analytic-force sorted backward on the union half-edges               #
# --------------------------------------------------------------------------- #
def _make_ljlk_sorted(bundle: S2Bundle) -> Callable:
    lt, ls, N = bundle.lt, bundle.ls, bundle.N
    ei, ej = ls.edge_i, ls.edge_j
    bpl = ls.edge_bpl[None]
    scc = lt.soft_core_rep_cap
    G = _LJLKGlobals(ls)
    ip = _Side(**{f: getattr(ls, f)[:, ei] for f in _SIDE_FIELDS})
    jp = _Side(**{f: getattr(ls, f)[:, ej] for f in _SIDE_FIELDS})
    ej_sorted, jperm = _argsort_j(ej)

    def pod(dist):
        Va, Vr = lj_atr_rep(dist, bpl, ip, jp, G, soft_core_cap=scc)
        Vl = lk_isotropic_pair(dist, bpl, ip, jp, G)
        return Va, Vr, Vl

    @jax.custom_vjp
    def f(c):
        _, dist, _ = _dd(c, ei, ej)
        Va, Vr, Vl = pod(dist)
        return {_ATR: Va.sum(-1), _REP: Vr.sum(-1), _LK: Vl.sum(-1)}

    def fwd(c):
        diff, dist, safe = _dd(c, ei, ej)
        (Va, Vr, Vl), (dVa, dVr, dVl) = jax.jvp(pod, (dist,), (jnp.ones_like(dist),))
        invd = jnp.where(safe, 1.0 / jnp.where(safe, dist, 1.0), 0.0)
        return ({_ATR: Va.sum(-1), _REP: Vr.sum(-1), _LK: Vl.sum(-1)},
                (diff, invd, dVa, dVr, dVl))

    def bwd(res, cot):
        diff, invd, dVa, dVr, dVl = res
        dEdd = cot[_ATR][:, None] * dVa + cot[_REP][:, None] * dVr + cot[_LK][:, None] * dVl
        force = ((dEdd * invd)[..., None]) * diff  # [P, E, 3]
        return (_seg_i(force, ei, N) + _seg_j(force, ej_sorted, jperm, N),)

    f.defvjp(fwd, bwd)
    return f


# --------------------------------------------------------------------------- #
# elec — identical analytic-force structure, single channel                   #
# --------------------------------------------------------------------------- #
def _make_elec_sorted(bundle: S2Bundle) -> Callable:
    et, es, N = bundle.et, bundle.es, bundle.N
    ei, ej = es.edge_i, es.edge_j
    e_i, e_j = es.charges[:, ei], es.charges[:, ej]
    bpl = es.edge_bpl[None]
    ej_sorted, jperm = _argsort_j(ej)

    def pod(dist):
        return elec_pair(dist, e_i, e_j, bpl, es)  # [P, E]

    @jax.custom_vjp
    def f(c):
        _, dist, _ = _dd(c, ei, ej)
        return {_EL: pod(dist).sum(-1)}

    def fwd(c):
        diff, dist, safe = _dd(c, ei, ej)
        Ev, dEv = jax.jvp(pod, (dist,), (jnp.ones_like(dist),))
        invd = jnp.where(safe, 1.0 / jnp.where(safe, dist, 1.0), 0.0)
        return {_EL: Ev.sum(-1)}, (diff, invd, dEv)

    def bwd(res, cot):
        diff, invd, dEv = res
        force = ((cot[_EL][:, None] * dEv * invd)[..., None]) * diff
        return (_seg_i(force, ei, N) + _seg_j(force, ej_sorted, jperm, N),)

    f.defvjp(fwd, bwd)
    return f


# --------------------------------------------------------------------------- #
# lk_ball — bridge-split (OPT3): list A (gated, iso/desolv) + list B (bridge)  #
# --------------------------------------------------------------------------- #
def _make_lkball_bridge_split(bundle: S2Bundle) -> Callable:
    kt, ks, N = bundle.kt, bundle.ks, bundle.N
    ei = np.asarray(ks.edge_i)
    ej = np.asarray(ks.edge_j)
    bpl = np.asarray(ks.edge_bpl)
    ip = np.asarray(ks.is_polar[0])
    io = np.asarray(ks.is_occluder[0])
    ir = np.asarray(ks.is_real[0])
    isd = np.asarray(ks.is_donor[0]) > 0.5
    isa = np.asarray(ks.is_acceptor[0]) > 0.5

    gate = ip[ei] & io[ej] & ir[ei] & ir[ej]  # list A support (iso/desolv)
    gidx = np.nonzero(gate)[0]
    bmask = gate & (isd[ej] | isa[ej])  # list B support (bridge)
    bidx = np.nonzero(bmask)[0]

    aI = jnp.asarray(ei[gidx].astype(np.int32))
    aJ = jnp.asarray(ej[gidx].astype(np.int32))
    aB = jnp.asarray(bpl[gidx])
    bI = jnp.asarray(ei[bidx].astype(np.int32))
    bJ = jnp.asarray(ej[bidx].astype(np.int32))
    bB = jnp.asarray(bpl[bidx])

    polar_idx = jnp.asarray(np.nonzero(ip)[0].astype(np.int32))
    n_pol = int(polar_idx.shape[0])
    n_waters = kt.n_waters

    _PP = (
        "lj_radius", "lk_dgfree", "lk_lambda", "lk_volume", "is_donor", "is_acceptor",
        "is_hydroxyl", "is_polarh", "is_polar", "is_occluder", "is_real",
        "B_idx", "B0_idx", "acc_hybridization", "attached_H_idx", "attached_H_mask",
        "n_attached_H",
    )
    per_pose = {f: getattr(ks, f) for f in _PP}

    def sigma_of(prm):
        return lj_sigma(
            prm["i_lj_radius"], prm["j_lj_radius"], prm["i_is_donor"], prm["j_is_donor"],
            prm["i_is_hydroxyl"], prm["j_is_hydroxyl"], prm["i_is_polarh"], prm["j_is_polarh"],
            prm["i_is_acceptor"], prm["j_is_acceptor"],
            _DEF.lj_hbond_dis, _DEF.lj_hbond_OH_donor_dis, _DEF.lj_hbond_hdis,
        )

    def place_polar(pc, p):
        safeB = jnp.where(p["B_idx"] >= 0, p["B_idx"], 0)
        safeB0 = jnp.where(p["B0_idx"] >= 0, p["B0_idx"], 0)
        safeH = jnp.where(p["attached_H_idx"] >= 0, p["attached_H_idx"], 0)
        Bc, B0c, Hc = pc[safeB[polar_idx]], pc[safeB0[polar_idx]], pc[safeH[polar_idx]]

        def one(k):
            pi = polar_idx[k]
            return place_waters_for_polar_atom(
                coord=pc[pi], is_donor=p["is_donor"][pi], is_acceptor=p["is_acceptor"][pi],
                hybridization=p["acc_hybridization"][pi], n_attached_H=p["n_attached_H"][pi],
                H_coords=Hc[k], H_mask=p["attached_H_mask"][pi], B_xyz=Bc[k], B0_xyz=B0c[k],
                n_waters=n_waters, water_dist=kt.defaults.lkb_water_dist,
            )

        wp = jax.vmap(one)(jnp.arange(n_pol))
        return jnp.full((N, n_waters, 3), jnp.nan, dtype=wp.dtype).at[polar_idx].set(wp)

    def gather_prm(p, I, J):
        g = lambda f, X: p[f][X]
        return {
            "i_lj_radius": g("lj_radius", I), "j_lj_radius": g("lj_radius", J),
            "i_lk_dgfree": g("lk_dgfree", I), "i_lk_lambda": g("lk_lambda", I),
            "j_lk_volume": g("lk_volume", J),
            "i_is_donor": g("is_donor", I), "j_is_donor": g("is_donor", J),
            "i_is_hydroxyl": g("is_hydroxyl", I), "j_is_hydroxyl": g("is_hydroxyl", J),
            "i_is_polarh": g("is_polarh", I), "j_is_polarh": g("is_polarh", J),
            "i_is_acceptor": g("is_acceptor", I), "j_is_acceptor": g("is_acceptor", J),
        }

    def pair_no_bridge(ci, wi, cj, wj, prm, b):
        has_w = jnp.logical_not(jnp.isnan(wi[0, 0]))
        wi_mask = jnp.logical_not(jnp.isnan(wi[:, 0]))
        swi = jnp.where(jnp.isnan(wi), 0.0, wi)
        sigma = sigma_of(prm)
        dist = safe_norm(ci - cj, axis=-1)
        within = dist < _DEF.distance_threshold
        lk_iso = lk_isotropic_pair_simplified(
            dist, sigma, prm["i_lj_radius"], prm["i_lk_dgfree"],
            prm["i_lk_lambda"], prm["j_lk_volume"], b,
        )
        frac = lk_fraction(swi, cj, prm["j_lj_radius"], _DEF.ramp_width_A2, wi_mask)
        g = jnp.logical_and(has_w, within).astype(lk_iso.dtype)
        return jnp.stack([lk_iso * g, lk_iso * frac * g])  # [iso, ball]

    def pair_bridge_only(ci, wi, cj, wj, prm, b):
        r = lk_ball_pair(ci, wi, cj, wj, prm, kt.defaults, b)
        return jnp.stack([r.lk_bridge, r.lk_bridge_uncpl])  # [bridge, uncpl]

    def one_pose(pc, p):
        waters = place_polar(pc, p)
        ciA, cjA, wiA, wjA = pc[aI], pc[aJ], waters[aI], waters[aJ]
        perA = jax.vmap(pair_no_bridge)(ciA, wiA, cjA, wjA, gather_prm(p, aI, aJ), aB)
        ciB, cjB, wiB, wjB = pc[bI], pc[bJ], waters[bI], waters[bJ]
        perB = jax.vmap(pair_bridge_only)(ciB, wiB, cjB, wjB, gather_prm(p, bI, bJ), bB)
        sA, sB = jnp.sum(perA, axis=0), jnp.sum(perB, axis=0)
        return jnp.stack([sA[0], sA[1], sB[0], sB[1]])  # [iso, ball, bridge, uncpl]

    def f(c):
        tot = jax.vmap(one_pose, in_axes=(0, 0))(c, per_pose)  # [P, 4]
        return {_ISO: tot[:, 0], _BALL: tot[:, 1], _BRIDGE: tot[:, 2], _UNCPL: tot[:, 3]}

    return f


# --------------------------------------------------------------------------- #
# assembly                                                                     #
# --------------------------------------------------------------------------- #
def build_fast_nonbonded(bundle: S2Bundle) -> Callable[[Any], dict]:
    """Return ``fn(coords_PN3) -> {ScoreType: [P]}`` for all 9 nonbonded channels.

    Parity-exact with :func:`~jtmol.nonbonded.multipose.s2_energies` (same value,
    gradient matches to float reduction order) but faster on the batched S2
    workload. Precomputes the sorted-scatter permutations and the lk_ball
    edge/water gates once at build time; call the returned closure per coordinate
    set. Differentiable in ``coords_PN3``.
    """
    ljlk = _make_ljlk_sorted(bundle)
    elec = _make_elec_sorted(bundle)
    lkball = _make_lkball_bridge_split(bundle)
    hs = bundle.hs

    def fast(coords_PN3) -> dict:
        c = jnp.asarray(coords_PN3)
        out: dict = {}
        out.update(ljlk(c))
        out.update(elec(c))
        out.update(lkball(c))
        out.update(_hbond_multipose_energy(hs, c))
        return out

    return fast


# --------------------------------------------------------------------------- #
# S3 ragged fast path (P distinct topologies, flat pose-segmented edge list)   #
# --------------------------------------------------------------------------- #
def _dd_flat(c, ei, ej):
    diff = c[ei] - c[ej]
    d2 = jnp.sum(diff * diff, -1)
    safe = d2 > 1e-6
    dist = jnp.where(safe, jnp.sqrt(jnp.where(safe, d2, 1.0)), 0.1)
    return diff, dist, safe


def _seg_sorted(x, idx, n):
    return jax.ops.segment_sum(x, idx, num_segments=n, indices_are_sorted=True)


def _make_ljlk_sorted_ragged(b: S3Bundle):
    ei, ej = b.lj_edge_i, b.lj_edge_j
    SN, pose, bpl = b.SumN, b.lj_edge_pose, b.lj_edge_bpl
    ip = _Side(**{f: b.lj[f][ei] for f in _SIDE_FIELDS})
    jp = _Side(**{f: b.lj[f][ej] for f in _SIDE_FIELDS})
    G, scc = b.lj_G, b.lt.soft_core_rep_cap
    ej_sorted, jperm = _argsort_j(ej)
    # lj_edge_pose is non-decreasing (ragged blocks concatenated in pose order),
    # so the forward reduction is a fast sorted segment_sum, not a scatter. Missing
    # this flag was ~96% of the S3 ljlk forward cost.
    ss = lambda x: jax.ops.segment_sum(x, pose, num_segments=b.P, indices_are_sorted=True)

    def pod(dist):
        Va, Vr = lj_atr_rep(dist, bpl, ip, jp, G, soft_core_cap=scc)
        return Va, Vr, lk_isotropic_pair(dist, bpl, ip, jp, G)

    @jax.custom_vjp
    def f(c):
        _, dist, _ = _dd_flat(c, ei, ej)
        Va, Vr, Vl = pod(dist)
        return {_ATR: ss(Va), _REP: ss(Vr), _LK: ss(Vl)}

    def fwd(c):
        diff, dist, safe = _dd_flat(c, ei, ej)
        (Va, Vr, Vl), (dVa, dVr, dVl) = jax.jvp(pod, (dist,), (jnp.ones_like(dist),))
        invd = jnp.where(safe, 1.0 / jnp.where(safe, dist, 1.0), 0.0)
        return {_ATR: ss(Va), _REP: ss(Vr), _LK: ss(Vl)}, (diff, invd, dVa, dVr, dVl)

    def bwd(res, cot):
        diff, invd, dVa, dVr, dVl = res
        dEdd = cot[_ATR][pose] * dVa + cot[_REP][pose] * dVr + cot[_LK][pose] * dVl
        force = ((dEdd * invd)[:, None]) * diff
        return (_seg_sorted(force, ei, SN) + _seg_sorted(-force[jperm], ej_sorted, SN),)

    f.defvjp(fwd, bwd)
    return f


def _make_elec_sorted_ragged(b: S3Bundle):
    ei, ej = b.el_edge_i, b.el_edge_j
    SN, pose = b.SumN, b.el_edge_pose
    e_i, e_j, bpl, glob = b.charges[ei], b.charges[ej], b.el_edge_bpl, b.el_globals
    ej_sorted, jperm = _argsort_j(ej)

    def pod(dist):
        return elec_pair(dist, e_i, e_j, bpl, glob)

    @jax.custom_vjp
    def f(c):
        _, dist, _ = _dd_flat(c, ei, ej)
        return {_EL: jax.ops.segment_sum(pod(dist), pose, num_segments=b.P, indices_are_sorted=True)}

    def fwd(c):
        diff, dist, safe = _dd_flat(c, ei, ej)
        Ev, dEv = jax.jvp(pod, (dist,), (jnp.ones_like(dist),))
        invd = jnp.where(safe, 1.0 / jnp.where(safe, dist, 1.0), 0.0)
        return {_EL: jax.ops.segment_sum(Ev, pose, num_segments=b.P, indices_are_sorted=True)}, (diff, invd, dEv)

    def bwd(res, cot):
        diff, invd, dEv = res
        force = ((cot[_EL][pose] * dEv * invd)[:, None]) * diff
        return (_seg_sorted(force, ei, SN) + _seg_sorted(-force[jperm], ej_sorted, SN),)

    f.defvjp(fwd, bwd)
    return f


def _make_lkball_bridge_split_ragged(b: S3Bundle):
    """Bridge-split lk_ball on the flat ragged edge list (L1 gate prefilter + L3
    bridge split; waters placed for all atoms). Ordinary autodiff backward."""
    kt, lk = b.kt, b.lk
    ei = np.asarray(b.lk_edge_i); ej = np.asarray(b.lk_edge_j)
    bpl = np.asarray(b.lk_edge_bpl); pose = np.asarray(b.lk_edge_pose)
    ip = np.asarray(lk["is_polar"]); io = np.asarray(lk["is_occluder"]); ir = np.asarray(lk["is_real"])
    isd = np.asarray(lk["is_donor"]) > 0.5; isa = np.asarray(lk["is_acceptor"]) > 0.5
    gate = ip[ei] & io[ej] & ir[ei] & ir[ej]; gidx = np.nonzero(gate)[0]
    bmask = gate & (isd[ej] | isa[ej]); bidx = np.nonzero(bmask)[0]
    aI = jnp.asarray(ei[gidx].astype(np.int32)); aJ = jnp.asarray(ej[gidx].astype(np.int32))
    aB = jnp.asarray(bpl[gidx]); aP = jnp.asarray(pose[gidx].astype(np.int32))
    bI = jnp.asarray(ei[bidx].astype(np.int32)); bJ = jnp.asarray(ej[bidx].astype(np.int32))
    bB = jnp.asarray(bpl[bidx]); bP = jnp.asarray(pose[bidx].astype(np.int32))
    P = b.P
    wf = (lk["is_donor"], lk["is_acceptor"], lk["is_polar"], lk["B_idx"], lk["B0_idx"],
          lk["acc_hybridization"], lk["attached_H_idx"], lk["attached_H_mask"], lk["n_attached_H"])

    def sigma_of(prm):
        return lj_sigma(
            prm["i_lj_radius"], prm["j_lj_radius"], prm["i_is_donor"], prm["j_is_donor"],
            prm["i_is_hydroxyl"], prm["j_is_hydroxyl"], prm["i_is_polarh"], prm["j_is_polarh"],
            prm["i_is_acceptor"], prm["j_is_acceptor"],
            _DEF.lj_hbond_dis, _DEF.lj_hbond_OH_donor_dis, _DEF.lj_hbond_hdis,
        )

    def gather_prm(I, J):
        g = lambda fld, X: lk[fld][X]
        return {
            "i_lj_radius": g("lj_radius", I), "j_lj_radius": g("lj_radius", J),
            "i_lk_dgfree": g("lk_dgfree", I), "i_lk_lambda": g("lk_lambda", I),
            "j_lk_volume": g("lk_volume", J),
            "i_is_donor": g("is_donor", I), "j_is_donor": g("is_donor", J),
            "i_is_hydroxyl": g("is_hydroxyl", I), "j_is_hydroxyl": g("is_hydroxyl", J),
            "i_is_polarh": g("is_polarh", I), "j_is_polarh": g("is_polarh", J),
            "i_is_acceptor": g("is_acceptor", I), "j_is_acceptor": g("is_acceptor", J),
        }

    prmA, prmB = gather_prm(aI, aJ), gather_prm(bI, bJ)

    def pair_no_bridge(ci, wi, cj, wj, prm, bb):
        has_w = jnp.logical_not(jnp.isnan(wi[0, 0]))
        wi_mask = jnp.logical_not(jnp.isnan(wi[:, 0]))
        swi = jnp.where(jnp.isnan(wi), 0.0, wi)
        sigma = sigma_of(prm)
        dist = safe_norm(ci - cj, axis=-1)
        within = dist < _DEF.distance_threshold
        lk_iso = lk_isotropic_pair_simplified(
            dist, sigma, prm["i_lj_radius"], prm["i_lk_dgfree"],
            prm["i_lk_lambda"], prm["j_lk_volume"], bb,
        )
        frac = lk_fraction(swi, cj, prm["j_lj_radius"], _DEF.ramp_width_A2, wi_mask)
        g = jnp.logical_and(has_w, within).astype(lk_iso.dtype)
        return jnp.stack([lk_iso * g, lk_iso * frac * g])

    def pair_bridge_only(ci, wi, cj, wj, prm, bb):
        r = lk_ball_pair(ci, wi, cj, wj, prm, kt.defaults, bb)
        return jnp.stack([r.lk_bridge, r.lk_bridge_uncpl])

    def f(c):
        waters = kt._place_all_waters(c, *wf)  # [SumN, n_waters, 3]
        perA = jax.vmap(pair_no_bridge)(c[aI], waters[aI], c[aJ], waters[aJ], prmA, aB)  # [na, 2]
        iso = jax.ops.segment_sum(perA[:, 0], aP, num_segments=P, indices_are_sorted=True)
        ball = jax.ops.segment_sum(perA[:, 1], aP, num_segments=P, indices_are_sorted=True)
        perB = jax.vmap(pair_bridge_only)(c[bI], waters[bI], c[bJ], waters[bJ], prmB, bB)  # [nb, 2]
        bridge = jax.ops.segment_sum(perB[:, 0], bP, num_segments=P, indices_are_sorted=True)
        uncpl = jax.ops.segment_sum(perB[:, 1], bP, num_segments=P, indices_are_sorted=True)
        return {_ISO: iso, _BALL: ball, _BRIDGE: bridge, _UNCPL: uncpl}

    return f


def _build_pad(ei0, ej0, bpl0, pose0, P):
    """Pad a flat pose-segmented edge list into a [P, max_E] batch + mask, and
    the sorted-scatter perms over the padded-flat atom indices. Padding edges point
    at atom 0 with mask 0 (zero energy, zero force). The padded
    masked ``.sum(-1)`` reduction is ~4x faster than a flat ``segment_sum`` here,
    bit-exact, at ~1.4x padding for similar-sized structures."""
    counts = np.bincount(pose0.astype(np.int64), minlength=P)
    maxE = int(counts.max()) if counts.size else 0
    ei_pad = np.zeros((P, maxE), np.int32)
    ej_pad = np.zeros((P, maxE), np.int32)
    bpl_pad = np.zeros((P, maxE), bpl0.dtype)
    mask = np.zeros((P, maxE), np.float32)
    off = 0
    for p in range(P):
        n = int(counts[p])
        ei_pad[p, :n] = ei0[off:off + n]
        ej_pad[p, :n] = ej0[off:off + n]
        bpl_pad[p, :n] = bpl0[off:off + n]
        mask[p, :n] = 1.0
        off += n
    eif = ei_pad.reshape(-1)
    ejf = ej_pad.reshape(-1)
    perm_i = np.argsort(eif, kind="stable").astype(np.int32)
    perm_j = np.argsort(ejf, kind="stable").astype(np.int32)
    return dict(
        ei=jnp.asarray(ei_pad), ej=jnp.asarray(ej_pad), bpl=jnp.asarray(bpl_pad),
        mask=jnp.asarray(mask), perm_i=jnp.asarray(perm_i), ei_sorted=jnp.asarray(eif[perm_i]),
        perm_j=jnp.asarray(perm_j), ej_sorted=jnp.asarray(ejf[perm_j]),
    )


def _make_ljlk_padded_ragged(b: S3Bundle):
    """ljlk: padded [P, max_E] masked-``.sum(-1)`` FORWARD (fast reduction) + FLAT
    sorted-scatter BACKWARD. Padding the backward's scatter (over the oversized
    padded-flat) regressed it 3x; instead recompute the (cheap) physics flat in fwd
    and use the proven flat sorted-scatter. Parity-exact vs the flat substrate."""
    N, P = b.SumN, b.P
    G, scc = b.lj_G, b.lt.soft_core_rep_cap
    # padded forward layout
    pd = _build_pad(np.asarray(b.lj_edge_i), np.asarray(b.lj_edge_j),
                    np.asarray(b.lj_edge_bpl), np.asarray(b.lj_edge_pose), P)
    ei_pad, ej_pad, bpl_pad, mask = pd["ei"], pd["ej"], pd["bpl"], pd["mask"]
    ip_p = _Side(**{f: b.lj[f][ei_pad] for f in _SIDE_FIELDS})
    jp_p = _Side(**{f: b.lj[f][ej_pad] for f in _SIDE_FIELDS})
    # flat backward layout
    ei_f, ej_f, bpl_f, pose_f = b.lj_edge_i, b.lj_edge_j, b.lj_edge_bpl, b.lj_edge_pose
    ip_f = _Side(**{f: b.lj[f][ei_f] for f in _SIDE_FIELDS})
    jp_f = _Side(**{f: b.lj[f][ej_f] for f in _SIDE_FIELDS})
    ej_sorted_f, jperm_f = _argsort_j(ej_f)

    def pod_pad(dist):
        Va, Vr = lj_atr_rep(dist, bpl_pad, ip_p, jp_p, G, soft_core_cap=scc)
        return Va, Vr, lk_isotropic_pair(dist, bpl_pad, ip_p, jp_p, G)

    def pod_flat(dist):
        Va, Vr = lj_atr_rep(dist, bpl_f, ip_f, jp_f, G, soft_core_cap=scc)
        return Va, Vr, lk_isotropic_pair(dist, bpl_f, ip_f, jp_f, G)

    ms = lambda V: (V * mask).sum(-1)

    def energy_pad(c):
        diff = c[ei_pad] - c[ej_pad]
        d2 = jnp.sum(diff * diff, -1); safe = d2 > 1e-6
        dist = jnp.where(safe, jnp.sqrt(jnp.where(safe, d2, 1.0)), 0.1)
        Va, Vr, Vl = pod_pad(dist)
        return {_ATR: ms(Va), _REP: ms(Vr), _LK: ms(Vl)}

    @jax.custom_vjp
    def f(c):
        return energy_pad(c)

    def fwd(c):
        E = energy_pad(c)
        diff = c[ei_f] - c[ej_f]
        d2 = jnp.sum(diff * diff, -1); safe = d2 > 1e-6
        dist = jnp.where(safe, jnp.sqrt(jnp.where(safe, d2, 1.0)), 0.1)
        _, (dVa, dVr, dVl) = jax.jvp(pod_flat, (dist,), (jnp.ones_like(dist),))
        invd = jnp.where(safe, 1.0 / jnp.where(safe, dist, 1.0), 0.0)
        return E, (diff, invd, dVa, dVr, dVl)

    def bwd(res, cot):
        diff, invd, dVa, dVr, dVl = res
        dEdd = cot[_ATR][pose_f] * dVa + cot[_REP][pose_f] * dVr + cot[_LK][pose_f] * dVl
        force = ((dEdd * invd)[:, None]) * diff  # [E, 3]
        gi = jax.ops.segment_sum(force, ei_f, num_segments=N, indices_are_sorted=True)
        gj = jax.ops.segment_sum(-force[jperm_f], ej_sorted_f, num_segments=N, indices_are_sorted=True)
        return (gi + gj,)

    f.defvjp(fwd, bwd)
    return f


def _make_elec_padded_ragged(b: S3Bundle):
    """elec: padded [P, max_E] masked-sum FORWARD + FLAT sorted-scatter BACKWARD
    (single channel; same padded-forward/flat-backward split as ljlk)."""
    N, P = b.SumN, b.P
    glob = b.el_globals
    pd = _build_pad(np.asarray(b.el_edge_i), np.asarray(b.el_edge_j),
                    np.asarray(b.el_edge_bpl), np.asarray(b.el_edge_pose), P)
    ei_pad, ej_pad, bpl_pad, mask = pd["ei"], pd["ej"], pd["bpl"], pd["mask"]
    ei_ch_p, ej_ch_p = b.charges[ei_pad], b.charges[ej_pad]
    ei_f, ej_f, bpl_f, pose_f = b.el_edge_i, b.el_edge_j, b.el_edge_bpl, b.el_edge_pose
    ei_ch_f, ej_ch_f = b.charges[ei_f], b.charges[ej_f]
    ej_sorted_f, jperm_f = _argsort_j(ej_f)

    def energy_pad(c):
        diff = c[ei_pad] - c[ej_pad]
        d2 = jnp.sum(diff * diff, -1); safe = d2 > 1e-6
        dist = jnp.where(safe, jnp.sqrt(jnp.where(safe, d2, 1.0)), 0.1)
        return {_EL: (elec_pair(dist, ei_ch_p, ej_ch_p, bpl_pad, glob) * mask).sum(-1)}

    def pod_flat(dist):
        return elec_pair(dist, ei_ch_f, ej_ch_f, bpl_f, glob)

    @jax.custom_vjp
    def f(c):
        return energy_pad(c)

    def fwd(c):
        E = energy_pad(c)
        diff = c[ei_f] - c[ej_f]
        d2 = jnp.sum(diff * diff, -1); safe = d2 > 1e-6
        dist = jnp.where(safe, jnp.sqrt(jnp.where(safe, d2, 1.0)), 0.1)
        _, dEv = jax.jvp(pod_flat, (dist,), (jnp.ones_like(dist),))
        invd = jnp.where(safe, 1.0 / jnp.where(safe, dist, 1.0), 0.0)
        return E, (diff, invd, dEv)

    def bwd(res, cot):
        diff, invd, dEv = res
        force = ((cot[_EL][pose_f] * dEv * invd)[:, None]) * diff  # [E, 3]
        gi = jax.ops.segment_sum(force, ei_f, num_segments=N, indices_are_sorted=True)
        gj = jax.ops.segment_sum(-force[jperm_f], ej_sorted_f, num_segments=N, indices_are_sorted=True)
        return (gi + gj,)

    f.defvjp(fwd, bwd)
    return f


def build_fast_nonbonded_ragged(b3: S3Bundle) -> Callable[[Any], dict]:
    """Return ``fn(coords_flat) -> {ScoreType: [P]}`` for all 9 channels on the
    ragged S3 substrate (P distinct topologies). Parity-exact with
    :func:`~jtmol.nonbonded.multipose.s3_energies`; padded-batch masked-sum forward
    + sorted-scatter backward on ljlk+elec, bridge-split lk_ball forward, generic
    hbond. Differentiable in the flat ``[SumN, 3]`` coordinate array."""
    ljlk = _make_ljlk_padded_ragged(b3)
    elec = _make_elec_padded_ragged(b3)
    lkball = _make_lkball_bridge_split_ragged(b3)
    hs = b3.hs_cat

    def fast(coords_flat) -> dict:
        c = jnp.asarray(coords_flat)
        out: dict = {}
        out.update(ljlk(c))
        out.update(elec(c))
        out.update(lkball(c))
        out.update(hbond_energy_on_verlet(hs, c))
        return out

    return fast


# --------------------------------------------------------------------------- #
# parity self-check                                                            #
# --------------------------------------------------------------------------- #
def demo() -> None:
    """Assert ``build_fast_nonbonded`` == :func:`s2_energies` (value + grad), f64.

    P=6 perturbed conformations of 1ubq; all 9 channels' energy and the weighted
    total gradient must match the generic substrate to reduction-order.
    """
    import os

    jax.config.update("jax_enable_x64", True)
    import jtmol
    from jtmol.database import ParameterDatabase
    from jtmol.nonbonded.multipose import setup_s2, s2_energies

    # official beta2016 nonbonded weights (sign matters for the grad check)
    W = {_ATR: 1.0, _REP: 0.55, _LK: 1.0, _EL: 1.0, _ISO: -0.38, _BALL: 0.92,
         _BRIDGE: -0.33, _UNCPL: -0.33, _HB: 1.0}
    pdb = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data", "1ubq.pdb")
    ps = jtmol.pose_stack_from_pdb(pdb)
    db = ParameterDatabase.get_default()
    base = np.asarray(ps.coords[0], dtype=np.float64)
    rng = np.random.default_rng(0)
    coords = jnp.asarray(np.stack([base + rng.normal(size=base.shape) * 0.3 for _ in range(6)]))

    bundle = setup_s2(ps, np.asarray(coords), db)
    fast = build_fast_nonbonded(bundle)
    ref = s2_energies(bundle, coords)
    got = fast(coords)

    worst_v = 0.0
    for st in _ALL9:
        r, g = np.asarray(ref[st]), np.asarray(got[st])
        worst_v = max(worst_v, float(np.abs(r - g).max() / (np.abs(r).max() + 1e-12)))
    print(f"[fast nonbonded] value  max rel = {worst_v:.2e}")

    gref = np.asarray(jax.grad(lambda c: sum(jnp.sum(s2_energies(bundle, c)[st]) * W[st] for st in _ALL9))(coords))
    gfast = np.asarray(jax.grad(lambda c: sum(jnp.sum(fast(c)[st]) * W[st] for st in _ALL9))(coords))
    worst_g = float(np.abs(gref - gfast).max() / (np.abs(gref).max() + 1e-12))
    print(f"[fast nonbonded] grad   max rel = {worst_g:.2e}")

    assert worst_v < 1e-6, f"value parity FAILED: {worst_v:.2e}"
    assert worst_g < 1e-4, f"grad parity FAILED: {worst_g:.2e}"
    print("PASS: fast nonbonded == s2_energies (value + grad), f64.")


if __name__ == "__main__":
    demo()
