"""Demo: tune ``k_on``, ``k_off``, ``K_d`` independently using
``KineticsTuningLoss`` on 1ubq split-binder.

Three tests in one script:

1.  Baseline — compute SS99 (kon), BUH (koff), Rosetta dG (Kd) at
    the reference coords. Perturbation-direction sanity check on
    each.

2.  Single-axis tuning — set ``(w_kon, w_koff, w_kd)`` to one-hot
    weights and check that the perturbation actually moves the
    chosen kinetic property in the right direction.

3.  Hot-residue identification per kinetic axis — show that the
    top residues for kon, koff, Kd are DIFFERENT, which is the
    whole point of having three separate gradient channels.

Run:
    PYTHONPATH=. python examples/demo_kinetics_tuning.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import tmol_jax
from tmol_jax.score.extra.ss99 import build_ss99_charges
from tmol_jax.score.extra.kinetics_tuning import (
    KineticsTuningLoss,
    per_residue_kinetic_gradient_norms,
)
from tmol_jax.score.score_function import ScoreFunction
from tmol_jax.score.score_types import ScoreType
from tmol_jax.score import beta2016_score_function


PDB = "tmol_jax/tests/data/1ubq.pdb"
BINDER_RES_END = 40
TOP_N = 8


def _setup_split_pose(pdb, split_res):
    ps = tmol_jax.pose_stack_from_pdb(pdb)
    irc = np.asarray(ps.inter_residue_connections).copy()
    irc[0, split_res - 1, 1] = -1
    irc[0, split_res, 0] = -1
    ibs = np.asarray(ps.inter_block_bondsep).copy()
    ibs[0, split_res - 1, split_res, :, :] = 6
    ibs[0, split_res, split_res - 1, :, :] = 6
    return ps.replace(
        inter_residue_connections=jnp.asarray(irc),
        inter_block_bondsep=jnp.asarray(ibs),
    )


def _name3(bt_idx, pbt):
    if bt_idx < 0:
        return "---"
    bt = pbt.active_block_types[bt_idx]
    return bt.name3 if hasattr(bt, "name3") else "???"


def main():
    print(f"Loading {PDB}, split at residue {BINDER_RES_END}")
    ps = _setup_split_pose(PDB, BINDER_RES_END)
    n_res = int(ps.n_res_per_pose[0])

    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)

    # Build per-atom / per-residue tables.
    res3_per_res = []
    atom_names = []
    atom_to_res = []
    res_ca_idx = np.zeros(n_res, dtype=np.int32)
    res_cb_idx = np.zeros(n_res, dtype=np.int32)
    is_chain_start = np.zeros(n_res, dtype=bool)
    is_chain_end = np.zeros(n_res, dtype=bool)
    chain_id_per_res = np.zeros(n_res, dtype=np.int32)

    for ri, bt in enumerate(bti):
        if bt < 0:
            res3_per_res.append("---")
            continue
        n = int(n_per_bt[bt])
        start = int(offsets[ri])
        names = pbt.atom_names[bt]
        try:
            name3 = pbt.active_block_types[bt].name3
        except Exception:
            name3 = "???"
        res3_per_res.append(name3)
        chain_id_per_res[ri] = 1 if ri >= BINDER_RES_END else 0
        ca_local = cb_local = None
        for local in range(n):
            atom_names.append(names[local])
            atom_to_res.append(ri)
            if names[local] == "CA":
                ca_local = local
            elif names[local] == "CB":
                cb_local = local
        res_ca_idx[ri] = (start + ca_local) if ca_local is not None else start
        res_cb_idx[ri] = (start + cb_local) if cb_local is not None else (start + (ca_local or 0))

    is_chain_start[0] = True
    is_chain_start[BINDER_RES_END] = True
    is_chain_end[BINDER_RES_END - 1] = True
    is_chain_end[n_res - 1] = True

    atom_to_res = np.asarray(atom_to_res, dtype=np.int32)
    coords = jnp.asarray(np.asarray(ps.coords)[0, :len(atom_to_res)])
    chain_id_atoms = jnp.asarray(chain_id_per_res[atom_to_res])

    print(f"  n_res={n_res}, n_atoms={coords.shape[0]}")

    # ---- SS99 inputs ----
    charges = build_ss99_charges(
        res3_per_res, atom_names, atom_to_res, is_chain_start, is_chain_end
    )

    # ---- BUH inputs ----
    polar_atom_idx_list = []
    polar_atom_res_list = []
    polar_capacity_list = []
    polar_donor_list = []
    polar_acceptor_list = []
    for ai, name in enumerate(atom_names):
        ri = int(atom_to_res[ai])
        r3 = res3_per_res[ri]
        cap = d = a = 0.0
        if name == "N":
            cap, d = 1.0, 1.0
        elif name == "O":
            cap, a = 1.0, 1.0
        elif r3 in {"SER", "THR"} and name in {"OG", "OG1"}:
            cap, d, a = 2.0, 1.0, 1.0
        elif r3 == "TYR" and name == "OH":
            cap, d, a = 2.0, 1.0, 1.0
        elif r3 == "ASN" and name == "ND2":
            cap, d = 1.0, 1.0
        elif r3 == "ASN" and name == "OD1":
            cap, a = 1.0, 1.0
        elif r3 == "GLN" and name == "NE2":
            cap, d = 1.0, 1.0
        elif r3 == "GLN" and name == "OE1":
            cap, a = 1.0, 1.0
        if cap > 0:
            polar_atom_idx_list.append(ai)
            polar_atom_res_list.append(ri)
            polar_capacity_list.append(cap)
            polar_donor_list.append(d)
            polar_acceptor_list.append(a)
    polar_atom_idx = jnp.asarray(polar_atom_idx_list, dtype=jnp.int32)
    polar_atom_res = jnp.asarray(polar_atom_res_list, dtype=jnp.int32)
    polar_capacity = jnp.asarray(polar_capacity_list, dtype=jnp.float32)
    polar_is_donor = jnp.asarray(polar_donor_list, dtype=jnp.float32)
    polar_is_acceptor = jnp.asarray(polar_acceptor_list, dtype=jnp.float32)

    # ---- Kd: Rosetta interface dG_separated (bound - separated) ----
    sf_full = beta2016_score_function()
    state_full = sf_full.setup_for_pose_stack(ps)
    # Cross-chain pair flag uses atom_chain_id directly via mask.
    # Simpler: use TmolInterfaceLoss's bound-minus-separated, computed
    # inline here.

    # Mark atoms with chain id so we can translate one side away.
    coords_orig = jnp.asarray(np.asarray(ps.coords))

    def kd_fn(c):
        # c is the flat-atom coords [n_atoms, 3]
        # Build the bound-and-separated full coords.
        n_used = c.shape[0]
        bound = coords_orig.at[0, :n_used].set(c)
        # Separated: translate binder atoms by +500 Å along X.
        binder_mask = chain_id_atoms == 0
        shift = jnp.where(binder_mask[:, None],
                          jnp.asarray([500.0, 0.0, 0.0]),
                          0.0)
        c_sep = c + shift
        separated = coords_orig.at[0, :n_used].set(c_sep)

        per_b = sf_full.score_by_term(
            ps, state=state_full, coords_override=bound,
        )
        per_s = sf_full.score_by_term(
            ps, state=state_full, coords_override=separated,
        )
        e_b = sum(v.reshape(-1)[0] for v in per_b.values())
        e_s = sum(v.reshape(-1)[0] for v in per_s.values())
        return e_b - e_s

    # ---- Baseline ----
    print("\n[1] Baseline at reference coords:")
    loss_obj = KineticsTuningLoss(
        w_kon=1.0, w_koff=1.0, w_kd=1.0,
        charges=charges, chain_id_atoms=chain_id_atoms,
        polar_atom_idx=polar_atom_idx, polar_atom_res=polar_atom_res,
        polar_capacity=polar_capacity,
        polar_is_donor=polar_is_donor, polar_is_acceptor=polar_is_acceptor,
        res_ca_idx=jnp.asarray(res_ca_idx),
        res_cb_idx=jnp.asarray(res_cb_idx),
        atom_to_res=jnp.asarray(atom_to_res),
        kd_fn=kd_fn,
    )
    U, BUH, dG = loss_obj.components(coords)
    print(f"  U_SS99 = {float(U):+.3f} kcal/mol   (lower = faster k_on)")
    print(f"  BUH    = {float(BUH):.3f}            (lower = slower k_off)")
    print(f"  dG_Kd  = {float(dG):+.2f} kcal/mol  (lower = tighter K_d)")

    # ---- Single-axis perturbation tests ----
    print("\n[2] Single-axis tuning — gradient step decreases each property:")
    eps = 1e-3
    for label, w in [
        ("kon",  dict(w_kon=1.0, w_koff=0.0, w_kd=0.0)),
        ("koff", dict(w_kon=0.0, w_koff=1.0, w_kd=0.0)),
        ("kd",   dict(w_kon=0.0, w_koff=0.0, w_kd=1.0)),
    ]:
        L = KineticsTuningLoss(
            **w,
            charges=charges, chain_id_atoms=chain_id_atoms,
            polar_atom_idx=polar_atom_idx, polar_atom_res=polar_atom_res,
            polar_capacity=polar_capacity,
            polar_is_donor=polar_is_donor, polar_is_acceptor=polar_is_acceptor,
            res_ca_idx=jnp.asarray(res_ca_idx),
            res_cb_idx=jnp.asarray(res_cb_idx),
            atom_to_res=jnp.asarray(atom_to_res),
            kd_fn=kd_fn,
        )
        def floss(c):
            v, _ = L(c)
            return v
        v0 = float(floss(coords))
        g = jax.grad(floss)(coords)
        gn = float(jnp.linalg.norm(g))
        v1 = float(floss(coords + eps * (-g / max(gn, 1e-12))))
        ok = v1 < v0 - 1e-6
        print(f"  axis={label:5s}  L={v0:+.4f} → L'={v1:+.4f}  "
              f"|g|={gn:.3e}  useful_gradient={ok}")

    # ---- Hot-residue analysis per kinetic axis ----
    print("\n[3] Per-residue hot-residue ranking on each kinetic axis:")
    norms = per_residue_kinetic_gradient_norms(
        loss_obj, coords, atom_to_res, n_res,
    )

    def top_binder_residues(arr, k=TOP_N):
        b = arr[:BINDER_RES_END]
        top = np.argsort(-b)[:k]
        return [(int(r) + 1, _name3(int(bti[r]), pbt), float(b[r])) for r in top]

    cols_kon = top_binder_residues(norms["kon"])
    cols_koff = top_binder_residues(norms["koff"])
    cols_kd = top_binder_residues(norms["kd"])

    print(f"  {'rank':>4s}   {'kon (res,aa,|g|)':>22s}   "
          f"{'koff (res,aa,|g|)':>22s}   {'kd (res,aa,|g|)':>22s}")
    for i in range(TOP_N):
        a, b, c = cols_kon[i], cols_koff[i], cols_kd[i]
        print(f"  {i+1:>4d}   "
              f"{a[0]:>4d} {a[1]} {a[2]:>10.2f}   "
              f"{b[0]:>4d} {b[1]} {b[2]:>10.2f}   "
              f"{c[0]:>4d} {c[1]} {c[2]:>10.2f}")

    # Overlap analysis — are the same residues hot for different axes?
    set_kon = set(r for r, _, _ in cols_kon)
    set_koff = set(r for r, _, _ in cols_koff)
    set_kd = set(r for r, _, _ in cols_kd)
    print(f"\n  Top-{TOP_N} overlap:")
    print(f"    kon ∩ koff = {sorted(set_kon & set_koff)}")
    print(f"    kon ∩ kd   = {sorted(set_kon & set_kd)}")
    print(f"    koff ∩ kd  = {sorted(set_koff & set_kd)}")
    print(f"    all three  = {sorted(set_kon & set_koff & set_kd)}")
    print(f"\n  Low overlap = mutations targeting one axis don't")
    print(f"  necessarily move the others (the whole point of having")
    print(f"  three separate gradient channels).")


if __name__ == "__main__":
    main()
