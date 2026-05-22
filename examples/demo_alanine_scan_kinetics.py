"""In-silico alanine scan per kinetic axis on the top hot residues.

Builds on ``demo_kinetics_tuning.py``. For each of the top-N hot
residues on each kinetic axis (``k_on``, ``k_off``), simulate
mutating that residue to ALA and report:

* ΔΔU_SS99 (kon proxy) — change in shielded-Coulomb electrostatic
  energy. Positive ΔΔ = mutation made kon *worse*.
* ΔΔBUH (koff proxy) — change in buried-unsat polar count. Positive
  ΔΔ = mutation made koff *worse* (more unsatisfied polars).

We use the static-topology trick: keep atom positions fixed and zero
out the per-atom properties that an alanine would not have (sidechain
charges and sidechain polar capacities). Backbone N and O stay polar.
This captures the **loss-of-function** effect of an ALA mutation but
not gain-of-function (since the sidechain atoms physically aren't
there).

The classical alanine scan does exactly this conceptually — replaces
sidechain beyond Cβ with hydrogen and re-scores. Real ΔΔG comparison
to wet-lab data also needs sidechain re-packing on neighbors, which
we don't do here; consider this an *upper bound* on the kinetic
penalty of each mutation.

Run:
    PYTHONPATH=. python examples/demo_alanine_scan_kinetics.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import tmol_jax
from tmol_jax.score.extra.ss99 import build_ss99_charges, ss99_energy
from tmol_jax.score.extra.buh import buh_score


PDB = "tmol_jax/tests/data/1ubq.pdb"
BINDER_RES_END = 40

# Top hot residues per axis from demo_kinetics_tuning.py output.
HOT_KON  = [27, 39, 24, 1, 6, 34, 16, 11]
HOT_KOFF = [25, 1, 38, 34, 36, 3, 15, 22]


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


def _is_backbone(name: str) -> bool:
    return name in {"N", "CA", "C", "O", "OXT", "H", "HA"}


def alanine_mutate_arrays(
    target_res: int,
    base_charges: np.ndarray,
    base_polar_capacity: np.ndarray,
    base_polar_donor: np.ndarray,
    base_polar_acceptor: np.ndarray,
    polar_atom_idx: np.ndarray,
    polar_atom_res: np.ndarray,
    atom_to_res: np.ndarray,
    atom_names: list,
):
    """Return mutated copies of the property arrays as if ``target_res``
    were an alanine: zero out sidechain charges and sidechain polar
    capacities; keep backbone properties untouched.
    """
    mut_charges = np.asarray(base_charges).copy()
    mut_capacity = np.asarray(base_polar_capacity).copy()
    mut_donor = np.asarray(base_polar_donor).copy()
    mut_acceptor = np.asarray(base_polar_acceptor).copy()

    # Sidechain charges → 0 on this residue.
    for ai in range(len(atom_names)):
        if int(atom_to_res[ai]) == target_res and not _is_backbone(atom_names[ai]):
            mut_charges[ai] = 0.0

    # Sidechain polars at this residue → not polar anymore.
    for pi in range(len(polar_atom_idx)):
        if int(polar_atom_res[pi]) != target_res:
            continue
        ai = int(polar_atom_idx[pi])
        if not _is_backbone(atom_names[ai]):
            mut_capacity[pi] = 0.0
            mut_donor[pi] = 0.0
            mut_acceptor[pi] = 0.0
    return mut_charges, mut_capacity, mut_donor, mut_acceptor


def main():
    print(f"Loading {PDB}, split at residue {BINDER_RES_END}")
    ps = _setup_split_pose(PDB, BINDER_RES_END)
    n_res = int(ps.n_res_per_pose[0])

    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)

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

    atom_to_res_np = np.asarray(atom_to_res, dtype=np.int32)
    coords = jnp.asarray(np.asarray(ps.coords)[0, : len(atom_to_res)])
    chain_id_atoms = jnp.asarray(chain_id_per_res[atom_to_res_np])

    print(f"  n_res={n_res}, n_atoms={coords.shape[0]}")

    # Build base SS99 charges.
    charges = build_ss99_charges(
        res3_per_res, atom_names, atom_to_res_np, is_chain_start, is_chain_end,
    )
    charges_np = np.asarray(charges)

    # Build base BUH polar arrays.
    polar_idx_list, polar_res_list, polar_cap_list = [], [], []
    polar_d_list, polar_a_list = [], []
    for ai, name in enumerate(atom_names):
        ri = int(atom_to_res_np[ai])
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
            polar_idx_list.append(ai)
            polar_res_list.append(ri)
            polar_cap_list.append(cap)
            polar_d_list.append(d)
            polar_a_list.append(a)

    polar_atom_idx_np = np.asarray(polar_idx_list, dtype=np.int32)
    polar_atom_res_np = np.asarray(polar_res_list, dtype=np.int32)
    polar_capacity_np = np.asarray(polar_cap_list, dtype=np.float32)
    polar_donor_np = np.asarray(polar_d_list, dtype=np.float32)
    polar_acceptor_np = np.asarray(polar_a_list, dtype=np.float32)

    polar_atom_idx = jnp.asarray(polar_atom_idx_np)
    polar_atom_res = jnp.asarray(polar_atom_res_np)
    res_ca_idx_j = jnp.asarray(res_ca_idx)
    res_cb_idx_j = jnp.asarray(res_cb_idx)
    atom_to_res_j = jnp.asarray(atom_to_res_np)

    # Compile SS99 and BUH once each with fixed topology — only the
    # per-atom property arrays change between mutants, not shapes.
    @jax.jit
    def ss99_with(coords, charges):
        return ss99_energy(coords, charges, chain_id_atoms, binder_chain_id=0)

    @jax.jit
    def buh_with(coords, polar_capacity, polar_donor, polar_acceptor):
        return buh_score(
            coords, polar_atom_idx, polar_atom_res,
            polar_capacity, jnp.zeros_like(polar_capacity),
            res_ca_idx_j, res_cb_idx_j, atom_to_res_j,
            polar_is_donor=polar_donor,
            polar_is_acceptor=polar_acceptor,
        )

    # Baseline values.
    print("\nBaseline (wild-type 1ubq split-binder):")
    U_wt = float(ss99_with(coords, jnp.asarray(charges_np)))
    BUH_wt = float(buh_with(
        coords,
        jnp.asarray(polar_capacity_np),
        jnp.asarray(polar_donor_np),
        jnp.asarray(polar_acceptor_np),
    ))
    print(f"  U_SS99_wt = {U_wt:+.4f} kcal/mol")
    print(f"  BUH_wt    = {BUH_wt:.4f}")

    # ----- Alanine scan on top hot residues (each axis) -----
    print(f"\nAlanine scan on top kon-axis hot residues:")
    print(f"  {'rank':>4s} {'res':>4s} {'wt':>4s} → ALA   "
          f"{'ΔΔU_SS99':>10s}   {'sign':>6s}")
    for rank, res_0 in enumerate([r - 1 for r in HOT_KON], start=1):
        wt = _name3(int(bti[res_0]), pbt)
        mc, _, _, _ = alanine_mutate_arrays(
            res_0, charges_np, polar_capacity_np,
            polar_donor_np, polar_acceptor_np,
            polar_atom_idx_np, polar_atom_res_np, atom_to_res_np, atom_names,
        )
        U_mut = float(ss99_with(coords, jnp.asarray(mc)))
        ddU = U_mut - U_wt
        # Positive ddU = mutation made kon worse (lost favorable electrostatic).
        sign = "worse" if ddU > 0.05 else ("better" if ddU < -0.05 else "≈")
        print(f"  {rank:>4d} {res_0+1:>4d} {wt:>4s} → ALA   "
              f"{ddU:>+10.4f}   {sign:>6s}")

    print(f"\nAlanine scan on top koff-axis hot residues:")
    print(f"  {'rank':>4s} {'res':>4s} {'wt':>4s} → ALA   "
          f"{'ΔΔBUH':>10s}   {'sign':>6s}")
    for rank, res_0 in enumerate([r - 1 for r in HOT_KOFF], start=1):
        wt = _name3(int(bti[res_0]), pbt)
        _, mcap, md, ma = alanine_mutate_arrays(
            res_0, charges_np, polar_capacity_np,
            polar_donor_np, polar_acceptor_np,
            polar_atom_idx_np, polar_atom_res_np, atom_to_res_np, atom_names,
        )
        BUH_mut = float(buh_with(
            coords, jnp.asarray(mcap), jnp.asarray(md), jnp.asarray(ma),
        ))
        dBUH = BUH_mut - BUH_wt
        # Positive dBUH = mutation increased unsatisfied polar count = worse koff.
        # Negative dBUH = mutation reduced unsat polars (by removing them entirely) = better koff.
        sign = "worse" if dBUH > 0.1 else ("better" if dBUH < -0.1 else "≈")
        print(f"  {rank:>4d} {res_0+1:>4d} {wt:>4s} → ALA   "
              f"{dBUH:>+10.4f}   {sign:>6s}")

    # ----- Joint scan: did the same mutation help both axes? ------
    print(f"\nJoint kinetic effect of ALA at the union of top hot residues:")
    union = sorted(set(HOT_KON) | set(HOT_KOFF))
    print(f"  union of top kon ∪ koff hot residues: {union}")
    print(f"  {'res':>4s} {'wt':>4s} → ALA   {'ΔΔU_SS99':>10s}   {'ΔΔBUH':>10s}   "
          f"{'kon':>6s}   {'koff':>6s}")
    for res_1 in union:
        res_0 = res_1 - 1
        wt = _name3(int(bti[res_0]), pbt)
        mc, mcap, md, ma = alanine_mutate_arrays(
            res_0, charges_np, polar_capacity_np,
            polar_donor_np, polar_acceptor_np,
            polar_atom_idx_np, polar_atom_res_np, atom_to_res_np, atom_names,
        )
        U_mut = float(ss99_with(coords, jnp.asarray(mc)))
        BUH_mut = float(buh_with(
            coords, jnp.asarray(mcap), jnp.asarray(md), jnp.asarray(ma),
        ))
        ddU = U_mut - U_wt
        dBUH = BUH_mut - BUH_wt
        kon_sign = "worse" if ddU > 0.05 else ("better" if ddU < -0.05 else "≈")
        koff_sign = "worse" if dBUH > 0.1 else ("better" if dBUH < -0.1 else "≈")
        print(f"  {res_1:>4d} {wt:>4s} → ALA   {ddU:>+10.4f}   {dBUH:>+10.4f}   "
              f"{kon_sign:>6s}   {koff_sign:>6s}")

    print("\nReading the table:")
    print("  * ΔΔU_SS99 > 0  = lost favorable electrostatic (slower k_on)")
    print("  * ΔΔBUH > 0     = removed a polar that was satisfied (worse k_off)")
    print("  * ΔΔBUH < 0     = removed an unsatisfied polar (better k_off)")
    print("  * Mutations where both ΔΔ go the same direction are coherent")
    print("    in their kinetic effect; opposites are tradeoffs.")


if __name__ == "__main__":
    main()
