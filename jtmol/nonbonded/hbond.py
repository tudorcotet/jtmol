"""H-bond on the uncapped adaptive Verlet substrate — the hbond term on the exact
nonbonded engine, modelled on the VERIFIED ``jtmol/nonbonded/ljlk.py``.

Reuses jtmol's hbond physics VERBATIM (the per-pair evaluator
:func:`jtmol.score.hbond.potentials.hbond_pair_V`, the polynomial tables, the
count-pair separation matrix); only the EDGE SET changes: instead of the dense
``[P, D, A]`` donor x acceptor grid (17136 pairs on 1ubq, almost all beyond the
A-H cutoff and contributing 0), we evaluate only the donor-H x acceptor pairs the
`jtmol.nonbonded.verlet` list places within the cutoff.

CRITICAL DIFFERENCE from LJLK/elec: the hbond term has NO edge-list path upstream
— its dense kernel enumerates every (donor, acceptor) slot and applies a geometric
cutoff mask (``ah_dist < max_ah_dist``, ~3.11 A on the H..A distance). So we BUILD
the donor x acceptor edge list ourselves from the substrate:

  1. The Verlet candidate list is built over just the hbond-participating atoms
     (each donor's polar H + each acceptor heavy atom — a disjoint set) at
     cutoff = ``max_ah_dist``. Candidate radius = cutoff + skin.
  2. Each candidate atom pair is oriented into (donor_slot, acceptor_slot) using
     atom -> slot maps (one endpoint is a polar H of a donor, the other an
     acceptor). Non-(H,A) candidate pairs (H-H, A-A) are dropped.
  3. Per edge we gather D/H/A/B/B0 coords + the (donor_type, acceptor_type)
     polynomials/weight/hybridization EXACTLY as the dense path gathers them, and
     the count-pair separation is gathered from the dense
     ``donor_acceptor_bond_sep`` matrix (never recomputed — the upstream-bug
     lookup rides along for free).

Correctness: the dense kept set is ``{(d,a): ah_dist < max_ah_dist AND cp_sep>=5}``
(all slots valid at P==1). Since cutoff = max_ah_dist, the candidate radius
max_ah_dist+skin is a strict superset of the within-cutoff set, so applying the
SAME within-cutoff + count-pair mask on the edges reproduces the dense kept set
EXACTLY -> the edge-list sum equals the dense grid sum up to float summation
order. A larger Verlet radius adds only masked-out (zero) edges, so the energy is
invariant to it (completeness sweep in :func:`demo`).

f64 ONLY: ``jtmol.score.hbond`` forces ``jax_enable_x64=True`` at import (the
degree-10 hbond polynomials need f64 to match upstream). This whole module builds
and verifies in f64 — there is no meaningful f32 path here.

Single-topology / single-pose (P==1) here; multi-pose and ragged batching are
handled by :mod:`jtmol.nonbonded.multipose`.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax

# hbond needs f64 (matches jtmol.score.hbond.__init__, which the imports below
# also trigger). Set explicitly so importing this module is self-sufficient.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jtmol.nonbonded.verlet import VerletList, build_verlet_list  # noqa: E402
from jtmol.score.common import distance  # noqa: E402
from jtmol.score.hbond.hbond_energy_term import HBondEnergyTerm  # noqa: E402
from jtmol.score.hbond.params import HBondGlobalParams  # noqa: E402
from jtmol.score.hbond.potentials import hbond_pair_V  # noqa: E402
from jtmol.score.score_types import ScoreType  # noqa: E402


@dataclass(frozen=True)
class HBondVerletState:
    """Per-edge donor x acceptor work list for the Verlet hbond evaluator.

    All arrays are length ``E`` (number of donor-H x acceptor edges within the
    A-H cutoff+skin). ``edge_{H,D,A,B,B0}`` are int32 atom indices into
    ``coords`` (the only differentiable input at eval time); the remaining fields
    are the per-edge params gathered by ``(donor_type, acceptor_type)`` — held
    constant, exactly as the dense path gathers its ``[P, D, A]`` grids.
    """

    edge_H: jax.Array      # int32[E] polar-H atom index
    edge_D: jax.Array      # int32[E] donor heavy atom index
    edge_A: jax.Array      # int32[E] acceptor atom index
    edge_B: jax.Array      # int32[E] acceptor base atom index
    edge_B0: jax.Array     # int32[E] acceptor base0 atom index
    edge_hyb: jax.Array    # int32[E] acceptor hybridization enum
    edge_weight: jax.Array  # f32[E] pair weight (donor_w * acceptor_w)
    edge_ahdist: jax.Array  # f64[E, 15] packed AH-distance polynomial
    edge_ahd: jax.Array    # f64[E, 15] packed AHD-angle polynomial
    edge_bah: jax.Array    # f64[E, 15] packed cos(BAH) polynomial
    edge_cp_sep: jax.Array  # int32[E] donor-acceptor bond separation (count-pair)
    edge_pose: jax.Array   # int32[E] pose id (all 0 at P==1)
    gp: HBondGlobalParams  # scalar knobs incl. max_ah_dist
    n_poses: int           # static, == 1


def setup_hbond_on_verlet(pose_stack, param_db=None, skin: float = 1.0,
                          cutoff: float | None = None):
    """Return ``(term, state, dense, verlet)`` for hbond on the Verlet list.

    The hbond term is extracted from ``beta2016_score_function`` so it carries
    the production config (following the LJLK build; a bare term is avoided on
    principle even though hbond has no soft-core/NL knobs). ``dense`` is the full
    :class:`HBondParams`; ``state`` is the donor x acceptor edge work list; call
    :func:`hbond_energy_on_verlet(state, coords)`.
    """
    import jtmol
    if int(np.asarray(pose_stack.coords).shape[0]) != 1:
        raise ValueError("setup_hbond_on_verlet is single-pose (P==1) for now")

    sfxn = jtmol.beta2016_score_function(param_db=param_db, use_neighbor_list=False)
    term = next(t for t in sfxn.terms if isinstance(t, HBondEnergyTerm))
    dense = term.setup_for_pose_stack(pose_stack)

    ap = dense.atom_params
    dH = np.asarray(ap.donor_H_atom[0])          # [D] polar-H atom idx (-1 pad)
    dHeavy = np.asarray(ap.donor_heavy_atom[0])   # [D]
    dType = np.asarray(ap.donor_type[0])          # [D]
    aA = np.asarray(ap.acceptor_atom[0])          # [A] acceptor atom idx (-1 pad)
    aB = np.asarray(ap.acceptor_base[0])          # [A]
    aB0 = np.asarray(ap.acceptor_base0[0])        # [A]
    aType = np.asarray(ap.acceptor_type[0])       # [A]

    coords0 = np.asarray(pose_stack.coords[0], dtype=np.float64)  # [N, 3]
    N = coords0.shape[0]

    max_ah = float(np.asarray(dense.global_params.max_ah_dist))
    cut = float(cutoff) if cutoff is not None else max_ah
    if cut < max_ah:
        raise ValueError(f"cutoff {cut} < hbond max_ah_dist {max_ah} would drop pairs")

    # atom -> slot maps (donor-H and acceptor atom sets are disjoint: H vs heavy).
    valid_d = dH >= 0
    valid_a = aA >= 0
    part_mask = np.zeros(N, dtype=bool)          # hbond-participating atoms
    part_mask[dH[valid_d]] = True
    part_mask[aA[valid_a]] = True
    atom_to_don = np.full(N, -1, dtype=np.int64)
    atom_to_don[dH[valid_d]] = np.nonzero(valid_d)[0]
    atom_to_acc = np.full(N, -1, dtype=np.int64)
    atom_to_acc[aA[valid_a]] = np.nonzero(valid_a)[0]

    # Verlet candidates over the participating atoms only, at the A-H cutoff.
    vl: VerletList = build_verlet_list(coords0, cut, skin, atom_is_real=part_mask)
    pi = np.asarray(vl.pairs_i)
    pj = np.asarray(vl.pairs_j)

    # Orient each candidate (i<j) pair into (donor_slot, acceptor_slot). ok1/ok2
    # are mutually exclusive (an atom can't be both donor-H and acceptor), so no
    # pair is double-counted.
    d1, a1 = atom_to_don[pi], atom_to_acc[pj]
    ok1 = (d1 >= 0) & (a1 >= 0)
    d2, a2 = atom_to_don[pj], atom_to_acc[pi]
    ok2 = (d2 >= 0) & (a2 >= 0)
    edge_d = np.concatenate([d1[ok1], d2[ok2]]).astype(np.int64)
    edge_a = np.concatenate([a1[ok1], a2[ok2]]).astype(np.int64)

    # Per-edge params, gathered EXACTLY like the dense [P,D,A] grid: by slot for
    # coords/atom-indices, by (donor_type, acceptor_type) for the pair tables.
    dt = dType[edge_d]
    at = aType[edge_a]
    weight = np.asarray(dense.pair_params.weight)                 # [nd, na] f32
    hyb = np.asarray(dense.pair_params.acceptor_hybridization)    # [nd, na] i32
    ahdist = np.asarray(dense.polynomials.AHdist_poly)            # [nd, na, 15] f64
    cosahd = np.asarray(dense.polynomials.cosAHD_poly)
    cosbah = np.asarray(dense.polynomials.cosBAH_poly)
    cp = np.asarray(dense.donor_acceptor_bond_sep[0])             # [D, A] i32

    state = HBondVerletState(
        edge_H=jnp.asarray(dH[edge_d].astype(np.int32)),
        edge_D=jnp.asarray(dHeavy[edge_d].astype(np.int32)),
        edge_A=jnp.asarray(aA[edge_a].astype(np.int32)),
        edge_B=jnp.asarray(aB[edge_a].astype(np.int32)),
        edge_B0=jnp.asarray(aB0[edge_a].astype(np.int32)),
        edge_hyb=jnp.asarray(hyb[dt, at].astype(np.int32)),
        edge_weight=jnp.asarray(weight[dt, at]),
        edge_ahdist=jnp.asarray(ahdist[dt, at]),
        edge_ahd=jnp.asarray(cosahd[dt, at]),
        edge_bah=jnp.asarray(cosbah[dt, at]),
        edge_cp_sep=jnp.asarray(cp[edge_d, edge_a].astype(np.int32)),
        edge_pose=jnp.zeros(edge_d.shape[0], dtype=jnp.int32),
        gp=dense.global_params,
        n_poses=1,
    )
    return term, state, dense, vl


def hbond_energy_on_verlet(state: HBondVerletState, coords):
    """``{ScoreType.hbond: [n_poses]}`` over the donor x acceptor Verlet edges.

    Differentiable in ``coords`` ([P, N, 3] or [N, 3]). Reuses ``hbond_pair_V``
    verbatim; the hard ``ah_dist < max_ah_dist`` mask matches the dense path and
    its derivative is zero a.e. (measure-zero boundary), so the gradient is exact.
    """
    c = coords[0] if coords.ndim == 3 else coords  # single pose
    D = c[state.edge_D]
    H = c[state.edge_H]
    A = c[state.edge_A]
    B = c[state.edge_B]
    B0 = c[state.edge_B0]

    E = hbond_pair_V(
        D, H, A, B, B0,
        state.edge_hyb, state.edge_weight,
        state.edge_ahdist, state.edge_ahd, state.edge_bah, state.gp,
    )
    ah_dist = distance(A, H)
    mask = (ah_dist < state.gp.max_ah_dist) & (state.edge_cp_sep >= 5)
    E = jnp.where(mask, E, jnp.zeros_like(E))

    # ponytail: segment_sum over pose id -- per-pose reduction that already
    # generalises to multi-pose; at P==1 every edge is in segment 0.
    per_pose = jax.ops.segment_sum(E, state.edge_pose, num_segments=state.n_poses)
    return {ScoreType.hbond: per_pose}


def demo() -> None:
    """Assert hbond-on-Verlet == jtmol dense hbond (energy + gradient) on 1ubq, f64.

    Also proves the donor x acceptor edge set is COMPLETE: bumping the Verlet
    radius by +2 A adds only masked-out (zero) edges, so the energy is invariant.
    """
    import os

    import jtmol
    from jtmol.database import ParameterDatabase

    pdb = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data", "1ubq.pdb")
    ps = jtmol.pose_stack_from_pdb(pdb)
    db = ParameterDatabase.get_default()

    term, state, dense, vl = setup_hbond_on_verlet(ps, db)
    n_don = int(np.asarray(dense.atom_params.donor_H_atom[0] >= 0).sum())
    n_acc = int(np.asarray(dense.atom_params.acceptor_atom[0] >= 0).sum())
    E = int(state.edge_H.size)
    print(f"[hbond-on-verlet] 1ubq: {n_don} donors x {n_acc} acceptors "
          f"(dense grid {n_don * n_acc}) -> {E} donor-H x acceptor Verlet edges "
          f"(cutoff=max_ah_dist={float(np.asarray(dense.global_params.max_ah_dist)):.4f} A)")

    # f64 only -- hbond forces x64 at import; there is no meaningful f32 path.
    coords = jnp.asarray(np.asarray(ps.coords, dtype=np.float64))
    ed = float(np.asarray(term(ps, dense, coords_override=coords)[ScoreType.hbond])[0])
    ev = float(np.asarray(hbond_energy_on_verlet(state, coords)[ScoreType.hbond])[0])
    rel = abs(ed - ev) / (abs(ed) + 1e-12)
    print(f"  energy : dense {ed:+.10f}  verlet {ev:+.10f}  rel {rel:.2e}")

    gd = jax.grad(lambda c: term(ps, dense, coords_override=c)[ScoreType.hbond].sum())(coords)
    gv = jax.grad(lambda c: hbond_energy_on_verlet(state, c)[ScoreType.hbond].sum())(coords)
    gmax = float(jnp.abs(gd - gv).max())
    gscale = float(jnp.abs(gd).max())
    print(f"  grad   : max|dense-verlet| {gmax:.2e}  (|grad|_max {gscale:.2e})")

    # Completeness sweep: a strictly larger radius must not change the energy.
    _, state_big, _, vl_big = setup_hbond_on_verlet(ps, db, cutoff=float(
        np.asarray(dense.global_params.max_ah_dist)) + 2.0)
    ev_big = float(np.asarray(hbond_energy_on_verlet(state_big, coords)[ScoreType.hbond])[0])
    sweep_rel = abs(ev - ev_big) / (abs(ev) + 1e-12)
    print(f"  sweep  : +2.0A radius -> {int(state_big.edge_H.size)} edges "
          f"(+{int(state_big.edge_H.size) - E}), energy {ev_big:+.10f}, rel {sweep_rel:.2e}")

    assert rel < 1e-4, f"f64 energy parity FAILED: rel {rel}"
    assert gmax < 1e-4, f"f64 gradient parity FAILED: {gmax}"
    assert sweep_rel < 1e-6, f"edge set INCOMPLETE: +2A radius changed energy by {sweep_rel}"
    print("\nhbond-on-Verlet == dense hbond in f64 (energy rel<1e-4, gradient), and the "
          "donor x acceptor edge set is COMPLETE (radius-invariant). Physics reused "
          "verbatim; only the edge SET changed.")


if __name__ == "__main__":
    demo()
