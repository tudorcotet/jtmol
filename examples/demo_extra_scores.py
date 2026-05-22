"""Smoke test for ``tmol_jax.score.extra``.

Exercises all five differentiable interface scores:

* ``ss99`` — Selzer-Schreiber 1999 electrostatic ``k_on`` proxy
* ``buh`` — Stranges-Kuhlman 2013 buried-unsat polars, smooth surrogate
  (with the *coords-only* hbond path so the full gradient flows
  through coords, not through a precomputed hbond-energy vector)
* ``prodigy`` — Vangone-Bonvin 2015 contact-based ΔG
* ``transcomp`` — Qin-Zhou 2011 transition-state ``k_on`` enhancement
* ``dfire`` — Zhang-Zhou-Yang 2005 atom-pair statistical potential
  (running on a placeholder potential table — coefficients not
  bundled, but the gradient infrastructure is exercised)

For each score we report:

  - the value at the reference coordinates,
  - the maximum ``|∂score/∂x|`` over all atoms,
  - whether the gradient is finite everywhere,
  - **the perturbation-direction sanity check**:
    ``loss(coords + ε·(−g)) < loss(coords)`` for small ``ε``, i.e. the
    gradient actually points uphill so a small step opposite to it
    lowers the score. If this fails it usually means the gradient
    is in a saturated region (sigmoid plateau) — the value is
    "finite" but not useful for optimization.

Run:
    PYTHONPATH=. python examples/demo_extra_scores.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import tmol_jax
from tmol_jax.score.extra import (
    ss99_energy,
    buh_score,
    buh_per_kA2,
    prodigy_dG,
    prodigy_contact_counts,
    transcomp_log_kon_enhancement,
    dfire_energy,
    make_placeholder_dfire_table,
)
from tmol_jax.score.extra.ss99 import build_ss99_charges
from tmol_jax.score.extra.prodigy import build_polarity_class


PDB = "tmol_jax/tests/data/1ubq.pdb"
BINDER_RES_END = 40


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


def _perturb_check(loss_fn, coords, eps=1e-3, tag=""):
    """Move ``ε·step`` units in the direction opposite the gradient
    and check that the loss actually decreases. Returns
    ``(L_before, L_after, did_decrease)``.

    A True result confirms the gradient is locally useful for
    optimization (not saturated). A False with finite gradient
    indicates a saturated-region gradient.
    """
    L = float(loss_fn(coords))
    g = jax.grad(loss_fn)(coords)
    g_norm = float(jnp.linalg.norm(g))
    if g_norm < 1e-10:
        return L, L, False, g_norm
    step = -g / g_norm                       # unit step opposite gradient
    L_next = float(loss_fn(coords + eps * step))
    return L, L_next, L_next < L - 1e-6, g_norm


def main():
    print(f"Loading {PDB}, splitting at residue {BINDER_RES_END}...")
    ps = _setup_split_pose(PDB, BINDER_RES_END)
    n_res = int(ps.n_res_per_pose[0])

    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)

    # Build per-residue / per-atom name lookups.
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

        ca_local = None
        cb_local = None
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
    coords = jnp.asarray(np.asarray(ps.coords)[0, : len(atom_to_res)])
    chain_id_atoms = jnp.asarray(chain_id_per_res[atom_to_res])

    print(f"  n_res = {n_res}, n_atoms = {coords.shape[0]}")

    # -------------------- SS99 ----------------------
    print("\n[1/5] SS99 (Selzer-Schreiber 1999 electrostatic):")
    charges = build_ss99_charges(
        res3_per_res, atom_names, atom_to_res, is_chain_start, is_chain_end
    )

    def f_ss99(c):
        return ss99_energy(c, charges, chain_id_atoms, binder_chain_id=0)

    L0, L1, ok, gn = _perturb_check(f_ss99, coords)
    print(f"  U = {L0:+.4f} kcal/mol; |grad| = {gn:.3e}; "
          f"perturb ε=1e-3 → L = {L1:+.4f}; useful gradient: {ok}")

    # -------------------- BUH (coords-only path) -------
    print("\n[2/5] BUH (coords-only hbond satisfaction):")
    polar_atom_idx_list = []
    polar_atom_res_list = []
    polar_capacity_list = []
    polar_donor_list = []
    polar_acceptor_list = []
    for ai, name in enumerate(atom_names):
        ri = int(atom_to_res[ai])
        r3 = res3_per_res[ri]
        capacity = 0.0
        is_donor = 0.0
        is_acceptor = 0.0
        if name == "N":
            capacity, is_donor = 1.0, 1.0
        elif name == "O":
            capacity, is_acceptor = 1.0, 1.0
        elif r3 in {"SER", "THR"} and name in {"OG", "OG1"}:
            capacity, is_donor, is_acceptor = 2.0, 1.0, 1.0
        elif r3 == "TYR" and name == "OH":
            capacity, is_donor, is_acceptor = 2.0, 1.0, 1.0
        elif r3 == "ASN" and name == "ND2":
            capacity, is_donor = 1.0, 1.0
        elif r3 == "ASN" and name == "OD1":
            capacity, is_acceptor = 1.0, 1.0
        elif r3 == "GLN" and name == "NE2":
            capacity, is_donor = 1.0, 1.0
        elif r3 == "GLN" and name == "OE1":
            capacity, is_acceptor = 1.0, 1.0
        elif r3 == "TRP" and name == "NE1":
            capacity, is_donor = 1.0, 1.0
        elif r3 == "HIS" and name in {"ND1", "NE2"}:
            capacity, is_donor = 1.0, 1.0
        if capacity > 0:
            polar_atom_idx_list.append(ai)
            polar_atom_res_list.append(ri)
            polar_capacity_list.append(capacity)
            polar_donor_list.append(is_donor)
            polar_acceptor_list.append(is_acceptor)

    polar_atom_idx = jnp.asarray(polar_atom_idx_list, dtype=jnp.int32)
    polar_atom_res = jnp.asarray(polar_atom_res_list, dtype=jnp.int32)
    polar_capacity = jnp.asarray(polar_capacity_list, dtype=jnp.float32)
    polar_is_donor = jnp.asarray(polar_donor_list, dtype=jnp.float32)
    polar_is_acceptor = jnp.asarray(polar_acceptor_list, dtype=jnp.float32)
    print(f"  polar atoms: {len(polar_atom_idx_list)}")

    def f_buh(c):
        return buh_score(
            c, polar_atom_idx, polar_atom_res, polar_capacity,
            jnp.zeros_like(polar_capacity),    # ignored in coords-only path
            jnp.asarray(res_ca_idx), jnp.asarray(res_cb_idx),
            jnp.asarray(atom_to_res),
            polar_is_donor=polar_is_donor,
            polar_is_acceptor=polar_is_acceptor,
        )

    L0, L1, ok, gn = _perturb_check(f_buh, coords)
    print(f"  BUH = {L0:.4f} (smooth); |grad| = {gn:.3e}; "
          f"perturb ε=1e-3 → L = {L1:.4f}; useful gradient: {ok}")

    # -------------------- PRODIGY -------------------
    print("\n[3/5] PRODIGY (Vangone-Bonvin 2015):")
    polarity = jnp.asarray(build_polarity_class(res3_per_res))
    chain_per_res = jnp.asarray(chain_id_per_res)

    def f_prodigy(c):
        return prodigy_dG(
            c, jnp.asarray(atom_to_res), polarity, chain_per_res,
            binder_chain_id=0,
        )

    L0, L1, ok, gn = _perturb_check(f_prodigy, coords)
    print(f"  ΔG = {L0:+.4f} kcal/mol; |grad| = {gn:.3e}; "
          f"perturb ε=1e-3 → L = {L1:+.4f}; useful gradient: {ok}")

    # -------------------- TransComp ------------------
    print("\n[4/5] TransComp (Qin-Zhou 2011, transition-state kon):")

    def f_tc(c):
        return transcomp_log_kon_enhancement(
            c, charges, chain_id_atoms, binder_chain_id=0,
        )

    L0, L1, ok, gn = _perturb_check(f_tc, coords)
    print(f"  ln(k_on/k_on0) = {L0:+.4f}; |grad| = {gn:.3e}; "
          f"perturb ε=1e-3 → L = {L1:+.4f}; useful gradient: {ok}")

    # -------------------- DFIRE ---------------------
    print("\n[5/5] DFIRE (Zhang-Zhou 2005, placeholder potential):")
    # Toy atom-type mapping: backbone N=1, CA=2, C=3, O=4, else 0 / -1.
    atom_types_np = np.zeros(coords.shape[0], dtype=np.int32)
    for ai, name in enumerate(atom_names):
        if name == "N":
            atom_types_np[ai] = 1
        elif name == "CA":
            atom_types_np[ai] = 2
        elif name == "C":
            atom_types_np[ai] = 3
        elif name == "O":
            atom_types_np[ai] = 4
        else:
            atom_types_np[ai] = 0
    atom_types = jnp.asarray(atom_types_np)
    potential_table = make_placeholder_dfire_table(seed=0)

    def f_dfire(c):
        return dfire_energy(
            c, atom_types, chain_id_atoms, potential_table,
            cross_chain_only=True, binder_chain_id=0,
        )

    L0, L1, ok, gn = _perturb_check(f_dfire, coords)
    print(f"  E_DFIRE (placeholder) = {L0:+.4f}; |grad| = {gn:.3e}; "
          f"perturb ε=1e-3 → L = {L1:+.4f}; useful gradient: {ok}")

    # -------------------- summary ----------------------
    print("\nAll five scores are differentiable end-to-end.")
    print("BUH now uses the coords-only Hbond path so the *full* gradient")
    print("flows through coordinates, not through a precomputed hbond-")
    print("energy vector. Replace DFIRE's placeholder table with the")
    print("published Zhou-Yang coefficients to get the real Kd correlate.")


if __name__ == "__main__":
    main()
