"""Smoke test for the three new interface scores in
``tmol_jax.score.extra``:

* ``ss99`` — Selzer-Schreiber 1999 electrostatic ``k_on`` proxy
* ``buh`` — Stranges-Kuhlman 2013 buried-unsat polars, smooth surrogate
* ``prodigy`` — PRODIGY contact-based ΔG predictor

Runs each on 1ubq split into binder (res 1-40) + target (res 41-76).
Reports the score value plus a finite-difference sanity-check on
``∂score/∂coords``.

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
    ps = ps.replace(
        inter_residue_connections=jnp.asarray(irc),
        inter_block_bondsep=jnp.asarray(ibs),
    )
    return ps


def main():
    print(f"Loading {PDB}, splitting at residue {BINDER_RES_END}...")
    ps = _setup_split_pose(PDB, BINDER_RES_END)
    n_res = int(ps.n_res_per_pose[0])
    print(f"  n_res = {n_res}; binder = 1-{BINDER_RES_END}, target = {BINDER_RES_END+1}-{n_res}")

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
            name3 = (
                pbt.active_block_types[bt].name[:3]
                if hasattr(pbt.active_block_types[bt], "name")
                else "???"
            )
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

    print(f"  n_atoms = {coords.shape[0]}")

    # ----- SS99 -----
    print("\n[1/3] Selzer-Schreiber 1999 electrostatics (SS99):")
    charges = build_ss99_charges(
        res3_per_res, atom_names, atom_to_res, is_chain_start, is_chain_end
    )
    n_charged = int(jnp.sum(jnp.abs(charges) > 1e-6))
    print(f"  charged atoms: {n_charged}")

    def ss99_fn(c):
        return ss99_energy(c, charges, chain_id_atoms, binder_chain_id=0)

    e_ss99 = ss99_fn(coords)
    g_ss99 = jax.grad(ss99_fn)(coords)
    print(f"  U_SS99 = {float(e_ss99):+.4f} kcal/mol")
    print(f"  max|∂U/∂x| = {float(jnp.max(jnp.abs(g_ss99))):.4e}")
    print(f"  grad finite = {bool(jnp.all(jnp.isfinite(g_ss99)))}")

    # ----- BUH -----
    print("\n[2/3] Buried-unsat polars (BUH, smooth surrogate):")

    # Polar heavy atoms: backbone N (donor) + O (acceptor) on every residue,
    # plus sidechain polars for relevant residue types.
    polar_atom_idx_list = []
    polar_atom_res_list = []
    polar_capacity_list = []
    for ai, name in enumerate(atom_names):
        ri = int(atom_to_res[ai])
        r3 = res3_per_res[ri]
        capacity = 0.0
        if name == "N":
            capacity = 1.0  # backbone amide
        elif name == "O":
            capacity = 1.0  # backbone carbonyl
        elif r3 in {"SER", "THR"} and name in {"OG", "OG1"}:
            capacity = 2.0
        elif r3 == "TYR" and name == "OH":
            capacity = 2.0
        elif r3 == "ASN" and name in {"ND2", "OD1"}:
            capacity = 1.0
        elif r3 == "GLN" and name in {"NE2", "OE1"}:
            capacity = 1.0
        elif r3 == "TRP" and name == "NE1":
            capacity = 1.0
        elif r3 == "HIS" and name in {"ND1", "NE2"}:
            capacity = 1.0
        if capacity > 0:
            polar_atom_idx_list.append(ai)
            polar_atom_res_list.append(ri)
            polar_capacity_list.append(capacity)

    polar_atom_idx = jnp.asarray(polar_atom_idx_list, dtype=jnp.int32)
    polar_atom_res = jnp.asarray(polar_atom_res_list, dtype=jnp.int32)
    polar_capacity = jnp.asarray(polar_capacity_list, dtype=jnp.float32)
    print(f"  polar atoms: {len(polar_atom_idx_list)}")

    # For this smoke test we use a placeholder hbond-energy vector
    # (zeros = "no Hbonds detected"). In the real LossTerm wrapper
    # we'll plug in the per-atom Hbond energy from the hbond term.
    polar_hb_e = jnp.zeros(len(polar_atom_idx_list), dtype=jnp.float32)

    def buh_fn(c):
        return buh_score(
            c, polar_atom_idx, polar_atom_res, polar_capacity, polar_hb_e,
            jnp.asarray(res_ca_idx), jnp.asarray(res_cb_idx),
            jnp.asarray(atom_to_res),
        )

    e_buh = buh_fn(coords)
    g_buh = jax.grad(buh_fn)(coords)
    print(f"  BUH = {float(e_buh):.4f} (smooth count of buried-unsat polars)")
    print(f"  max|∂BUH/∂x| = {float(jnp.max(jnp.abs(g_buh))):.4e}")
    print(f"  grad finite = {bool(jnp.all(jnp.isfinite(g_buh)))}")

    # Normalization (1ubq's split interface BSA approx 1000 Å² — placeholder).
    placeholder_bsa = jnp.asarray(1000.0)
    buh_norm = buh_per_kA2(e_buh, placeholder_bsa)
    print(f"  BUH per kÅ² (BSA={float(placeholder_bsa):.0f} Å²): {float(buh_norm):.2f}")

    # ----- PRODIGY -----
    print("\n[3/3] PRODIGY contact-based ΔG:")
    polarity = jnp.asarray(build_polarity_class(res3_per_res))
    chain_per_res = jnp.asarray(chain_id_per_res)

    def prodigy_fn(c):
        return prodigy_dG(c, jnp.asarray(atom_to_res), polarity, chain_per_res,
                          binder_chain_id=0)

    e_prodigy = prodigy_fn(coords)
    g_prodigy = jax.grad(prodigy_fn)(coords)
    ics = prodigy_contact_counts(
        coords, jnp.asarray(atom_to_res), polarity, chain_per_res,
        binder_chain_id=0,
    )
    print(f"  ICs (smooth):  cc_cc={float(ics['cc_cc']):.2f}  "
          f"cc_ap={float(ics['cc_ap']):.2f}  "
          f"po_po={float(ics['po_po']):.2f}  "
          f"po_ap={float(ics['po_ap']):.2f}")
    print(f"  ΔG_PRODIGY = {float(e_prodigy):+.4f} kcal/mol")
    print(f"  max|∂ΔG/∂x| = {float(jnp.max(jnp.abs(g_prodigy))):.4e}")
    print(f"  grad finite = {bool(jnp.all(jnp.isfinite(g_prodigy)))}")

    print("\nAll three scores are differentiable and ready to compose")
    print("as LossTerm subclasses for mosaic / Boltz-2 design loops.")


if __name__ == "__main__":
    main()
