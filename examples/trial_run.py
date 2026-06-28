"""Local trial run: measure the physics-only cost of the design loop.

Without Boltz-2 we can still measure:

1. **Per-step cost of the tmol_jax interface loss** — directly
   comparable to the Boltz-2 step time in the real loop. If tmol is
   <1% of Boltz, the paper's big compute table is basically just the
   Boltz-2 number.

2. **Gradient signal-to-noise** of each physics term — do the
   per-term gradients really point in interpretable directions? We
   test this by doing pure-physics gradient descent on the binder
   atom37 coords (with target frozen) starting from the native 1ubq
   split, and measuring whether the optimizer finds a lower-energy
   configuration AND how composition metrics shift under different
   weight schemes.

3. **Scheme → composition** link — this is the main H1 test without
   any ML model involvement. Sanity checks the core claim.

Results printed as a markdown table ready to paste into the paper's
H1 figure.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------- #
# Test system: 1ubq artificially split binder (res 1-40) + target (41-76)
# --------------------------------------------------------------------- #


def build_1ubq_binder_target_stack():
    import tmol_jax

    ps = tmol_jax.pose_stack_from_pdb(
        "tmol_jax/tests/data/1ubq.pdb"
    )

    # Cut the peptide bond between residues 40 and 41 so the binder
    # half and target half are two independent chains for interface
    # analysis.
    split_res = 40
    irc = np.array(ps.inter_residue_connections, copy=True)
    irc[0, split_res - 1, 1] = -1
    irc[0, split_res, 0] = -1
    ibs = np.array(ps.inter_block_bondsep, copy=True)
    ibs[0, split_res - 1, split_res, :, :] = 6
    ibs[0, split_res, split_res - 1, :, :] = 6
    ps = ps.replace(
        inter_residue_connections=jnp.asarray(irc),
        inter_block_bondsep=jnp.asarray(ibs),
    )
    return ps, split_res


# --------------------------------------------------------------------- #
# Weight schemes — the 7 canonical schemes from the paper plan          #
# --------------------------------------------------------------------- #

SCHEMES: Dict[str, Optional[Dict[str, float]]] = {
    "baseline":            None,
    "electrostatics_5x":   {"fa_elec": 5.0},
    "hbond_5x":            {"hbond": 5.0},
    "shape_2x":            {"fa_ljatr": 2.0, "fa_ljrep": 2.0},
    "water_bridged":       {"lk_ball": 3.0, "lk_bridge": 3.0, "lk_bridge_uncpl": 3.0},
    "polar_only":          {"fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0},
    "ml_only_baseline":    {  # zero out all physics — reference for
                             # "no physics gradient" condition
        "fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0, "fa_elec": 0.0,
        "hbond": 0.0, "rama": 0.0, "omega": 0.0,
        "dunbrack_rot": 0.0, "dunbrack_rotdev": 0.0, "dunbrack_semirot": 0.0,
        "cart_lengths": 0.0, "cart_angles": 0.0, "cart_torsions": 0.0,
        "cart_impropers": 0.0, "cart_hxltorsions": 0.0,
        "lk_ball_iso": 0.0, "lk_ball": 0.0,
        "lk_bridge": 0.0, "lk_bridge_uncpl": 0.0,
        "ref": 0.0, "disulfide": 0.0,
    },
}


# --------------------------------------------------------------------- #
# Compose interface loss                                                 #
# --------------------------------------------------------------------- #


def make_interface_loss(ps, term_weights):
    import tmol_jax
    from tmol_jax.score.score_types import ScoreType

    sfxn = tmol_jax.beta2016_score_function()
    if term_weights is not None:
        for name, w in term_weights.items():
            sfxn.set_weight(getattr(ScoreType, name), float(w))
    state = sfxn.setup_for_pose_stack(ps)

    @jax.jit
    def bound_energy(coords):
        return sfxn(ps, state=state, coords_override=coords).sum()

    return sfxn, state, bound_energy


# --------------------------------------------------------------------- #
# Composition metrics (the H1 "what changes" table)                      #
# --------------------------------------------------------------------- #


def composition_metrics(ps, split_res: int) -> Dict[str, float]:
    """Inspect the current PoseStack and return per-interface features.

    Counts at the atom level:
      - n_hbond_capable_pairs: donor↔acceptor pairs < 3.5 Å across the
        interface (structural H-bond proxy)
      - n_salt_bridge_pairs: polar-H ↔ acceptor pairs < 4.0 Å (charge
        proxy)
      - n_hydrophobic_contacts: C-C pairs < 4.5 Å across the interface
      - interface_area_proxy: count of atom pairs < 5.0 Å
      - min_cross_chain_distance: tightest atom-atom distance
      - max_cross_chain_clash: largest repulsive-pair penalty
    """
    import tmol_jax

    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)
    coords = np.asarray(ps.coords[0])  # [max_n_atoms, 3]

    # chain assignment: atoms belonging to residues < split_res are
    # binder, else target.
    atom_chain = np.full(coords.shape[0], -1, dtype=np.int32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        start = int(offsets[r])
        atom_chain[start : start + n] = 0 if r < split_res else 1

    # flags per atom (donor/acceptor/polar-H/carbon)
    n_atoms_total = coords.shape[0]
    is_donor = np.zeros(n_atoms_total, dtype=bool)
    is_acceptor = np.zeros(n_atoms_total, dtype=bool)
    is_polarh = np.zeros(n_atoms_total, dtype=bool)
    is_carbon = np.zeros(n_atoms_total, dtype=bool)
    chem_atom_types = pbt.chem_db.atom_types
    atom_types_pbt = np.asarray(pbt.atom_types)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        start = int(offsets[r])
        for local in range(n):
            at = int(atom_types_pbt[bt, local])
            if at < 0:
                continue
            cat = chem_atom_types[at]
            is_donor[start + local] = cat.is_donor
            is_acceptor[start + local] = cat.is_acceptor
            is_polarh[start + local] = cat.is_polarh
            is_carbon[start + local] = cat.element == "C"

    # Pairwise cross-chain distances
    mask_binder = atom_chain == 0
    mask_target = atom_chain == 1
    bi = np.where(mask_binder)[0]
    ti = np.where(mask_target)[0]
    if bi.size == 0 or ti.size == 0:
        return {}
    xb = coords[bi]
    xt = coords[ti]
    D = np.linalg.norm(xb[:, None, :] - xt[None, :, :], axis=-1)

    # Hbond-capable pair: (donor in binder AND acceptor in target) OR vice versa, and distance < 3.5
    don_b = is_donor[bi][:, None] & is_acceptor[ti][None, :]
    don_t = is_acceptor[bi][:, None] & is_donor[ti][None, :]
    hbond_mask = (don_b | don_t) & (D < 3.5)

    polar_b = is_polarh[bi][:, None] & is_acceptor[ti][None, :]
    polar_t = is_acceptor[bi][:, None] & is_polarh[ti][None, :]
    salt_mask = (polar_b | polar_t) & (D < 4.0)

    cc_mask = is_carbon[bi][:, None] & is_carbon[ti][None, :]
    hydrophobic_mask = cc_mask & (D < 4.5)

    any_contact_mask = D < 5.0

    return {
        "n_hbond_pairs":         int(hbond_mask.sum()),
        "n_salt_bridge_pairs":   int(salt_mask.sum()),
        "n_hydrophobic_contacts": int(hydrophobic_mask.sum()),
        "n_contacts_5A":         int(any_contact_mask.sum()),
        "min_dist":              float(D.min()),
        "max_close_contact":     float((D < 3.0).sum()),
    }


# --------------------------------------------------------------------- #
# Trial 1: timing                                                        #
# --------------------------------------------------------------------- #


def trial_timing():
    """Measure per-step tmol_jax cost on the 1ubq binder+target system."""
    ps, split_res = build_1ubq_binder_target_stack()
    sfxn, state, bound = make_interface_loss(ps, None)

    # Warm up
    coords = ps.coords
    bound(coords).block_until_ready()

    # Energy timing
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        bound(coords).block_until_ready()
    t_energy = (time.perf_counter() - t0) / N

    # Gradient timing
    grad_fn = jax.jit(jax.grad(lambda c: bound(c)))
    grad_fn(coords).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N):
        grad_fn(coords).block_until_ready()
    t_grad = (time.perf_counter() - t0) / N

    print("## Timing on 1ubq (76 res, 1231 atoms)")
    print(f"| metric | ms/call |")
    print(f"| --- | ---: |")
    print(f"| tmol_jax energy (all 20 terms, dense) | {t_energy*1000:.2f} |")
    print(f"| tmol_jax gradient w.r.t. all coords    | {t_grad*1000:.2f} |")
    print(f"| per step in a 165-step APGM trajectory | {t_grad*165*1000:.1f} ms total |")
    print()
    return t_energy, t_grad


# --------------------------------------------------------------------- #
# Trial 2: scheme → composition                                           #
# --------------------------------------------------------------------- #


def trial_schemes_on_native():
    """Evaluate the 7 weight schemes on the NATIVE 1ubq structure.

    Because no optimization happens here, per-scheme numbers differ
    *only* via the reweighting, not via different coords. This gives
    the floor of what reweighting can do — in real design, optimizing
    different schemes would produce different coords, amplifying the
    differences we see here.
    """
    ps, split_res = build_1ubq_binder_target_stack()

    # Composition of native interface (scheme-independent)
    comp = composition_metrics(ps, split_res)

    # Per-scheme interface energy (E_bound - E_separated)
    print("## Per-scheme interface energy on native 1ubq (kcal/mol)")
    print(f"| scheme | bound_E | separated_E | dG_interface |")
    print(f"| --- | ---: | ---: | ---: |")
    results = {}
    for name, w in SCHEMES.items():
        sfxn, state, bound = make_interface_loss(ps, w)
        E_bound = float(bound(ps.coords))

        # translate binder +500 Å in x
        atom_chain = np.zeros(ps.coords.shape[1], dtype=np.int32)
        bti = np.asarray(ps.block_type_ind[0])
        offsets = np.asarray(ps.block_coord_offset[0])
        n_per_bt = np.asarray(ps.packed_block_types.n_atoms)
        for r, bt in enumerate(bti):
            if bt < 0:
                continue
            n = int(n_per_bt[bt])
            atom_chain[int(offsets[r]) : int(offsets[r]) + n] = (
                1 if r >= split_res else 0
            )
        is_binder = atom_chain == 0
        shift = jnp.where(is_binder[:, None], jnp.asarray([500.0, 0, 0]), 0.0)
        coords_sep = ps.coords.at[0].set(ps.coords[0] + shift)
        E_sep = float(bound(coords_sep))
        dG = E_bound - E_sep
        results[name] = dG
        print(f"| {name} | {E_bound:+.2f} | {E_sep:+.2f} | {dG:+.2f} |")
    print()

    print("## Native 1ubq interface composition (coord-only)")
    print("| metric | value |")
    print(f"| --- | ---: |")
    for k, v in comp.items():
        print(f"| {k} | {v} |")
    print()
    return results, comp


# --------------------------------------------------------------------- #
# Trial 3: steepest-descent design from native, one scheme at a time     #
# --------------------------------------------------------------------- #


def trial_sd_optimization(n_steps: int = 40, lr: float = 0.01):
    """Run pure-physics steepest descent on the binder atom37 coords,
    target frozen, for each scheme. Shows:

    - whether the gradient from each scheme really *lowers* the energy
    - how the composition of the relaxed structure differs by scheme
    - per-step wall-clock (upper bound for the design loop overhead)
    """
    ps, split_res = build_1ubq_binder_target_stack()
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(ps.packed_block_types.n_atoms)
    atom_chain = np.zeros(ps.coords.shape[1], dtype=np.int32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        atom_chain[int(offsets[r]) : int(offsets[r]) + n] = (
            1 if r >= split_res else 0
        )
    is_binder = jnp.asarray(atom_chain == 0)

    print(f"## Pure-physics SD minimization ({n_steps} steps, lr={lr})")
    print(f"| scheme | E₀ | Eₙ | ΔE | hbonds₀ | hbondsₙ | t/step ms |")
    print(f"| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    starting_coords = ps.coords

    for name, w in SCHEMES.items():
        sfxn, state, bound = make_interface_loss(ps, w)
        grad_fn = jax.jit(jax.grad(lambda c: bound(c)))

        coords = starting_coords
        E0 = float(bound(coords))
        comp0 = composition_metrics(ps.replace(coords=coords), split_res)

        # Warm-up compilation
        grad_fn(coords).block_until_ready()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            g = grad_fn(coords)
            # freeze the target: mask binder-only atoms
            g_masked = jnp.where(is_binder[None, :, None], g, 0.0)
            coords = coords - lr * g_masked
        coords.block_until_ready()
        t_per_step = (time.perf_counter() - t0) / n_steps * 1000

        En = float(bound(coords))
        compN = composition_metrics(ps.replace(coords=coords), split_res)

        print(
            f"| {name} | {E0:+.2f} | {En:+.2f} | {En-E0:+.2f} | "
            f"{comp0['n_hbond_pairs']} | {compN['n_hbond_pairs']} | "
            f"{t_per_step:.2f} |"
        )
    print()


# --------------------------------------------------------------------- #
# Main                                                                    #
# --------------------------------------------------------------------- #


if __name__ == "__main__":
    print("# tmol_jax trial run\n")
    print(f"Platform: {jax.devices()}\n")

    print("----")
    trial_timing()
    print("----")
    trial_schemes_on_native()
    print("----")
    trial_sd_optimization(n_steps=40, lr=0.005)
    print("----")
    print("Done.")
