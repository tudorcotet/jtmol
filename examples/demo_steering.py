"""Per-term reweighting steers the binding-affinity gradient in
*different directions*.

The mosaic / Boltz-2 design loop already gives you a gradient on the
sequence (or PSSM) from iPTM / PAE / pLDDT. Adding `TmolInterfaceLoss`
gives you a *second*, physics-grounded gradient on the coords. The
interesting bit is that we can reweight `beta_nov2016`'s 20 channels
to bias the gradient toward different binding hypotheses
(electrostatics-driven, shape complementarity, water-bridged, etc.).

This script measures, on a real PDB (1ubq split into a "binder" +
"target" surrogate), how much the gradient direction changes under
different reweightings, using cosine similarity to the default
beta_nov2016 gradient. Schemes that change the gradient direction
are the ones that actually do something interesting inside the design
loop; schemes that don't are just rescaling.

Run:
    PYTHONPATH=. python examples/demo_steering.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import tmol_jax
from examples.integration_mosaic_hallucinator import (
    _ATOM37_NAMES,
    _build_atom37_to_flat_mapping,
    TmolInterfaceLoss,
)
import equinox as eqx
from jaxtyping import Array


def _setup_split_pose_stack(pdb: str = "tmol_jax/tests/data/1ubq.pdb",
                            split_res: int = 40):
    """Load 1ubq, mark res 0..split-1 as chain 0 (binder), rest chain 1."""
    ps = tmol_jax.pose_stack_from_pdb(pdb)

    # Cut the peptide bond between (split_res-1) and split_res.
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

    # atom37 mapping
    flat_from_a37, is_mapped = _build_atom37_to_flat_mapping(ps)
    object.__setattr__(ps, "_atom37_map", jnp.asarray(flat_from_a37))
    object.__setattr__(ps, "_atom37_is_mapped", jnp.asarray(is_mapped))

    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)

    # Per-atom chain id (0 for binder, 1 for target).
    atom_chain = np.zeros(ps.coords.shape[1], dtype=np.int32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        atom_chain[int(offsets[r]):int(offsets[r]) + n] = 1 if r >= split_res else 0
    object.__setattr__(ps, "_atom_chain_id", jnp.asarray(atom_chain))

    # Build atom37 from real PoseStack coords.
    n_res = int(ps.n_res_per_pose[0])
    atom37 = np.zeros((n_res, 37, 3), dtype=np.float32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        start = int(offsets[r])
        names = pbt.atom_names[bt]
        for local in range(n):
            if names[local] in _ATOM37_NAMES:
                slot = _ATOM37_NAMES.index(names[local])
                atom37[r, slot] = np.asarray(ps.coords[0, start + local])
    return ps, jnp.asarray(atom37), n_res


class _StructOut(eqx.Module):
    structure_coordinates: Array
    backbone_coordinates: Array
    full_sequence: Array


def _grad_under(ps, atom37, full_seq, weights):
    iface = TmolInterfaceLoss(ps, term_weights=weights, scale=1.0,
                              binder_chain_id=0)

    def loss_of_coords(c):
        out = _StructOut(
            structure_coordinates=c,
            backbone_coordinates=c[:, [0, 1, 2, 4]],
            full_sequence=full_seq,
        )
        v, _ = iface(full_seq, out, key=jax.random.PRNGKey(0))
        return v

    v = loss_of_coords(atom37)
    g = jax.grad(loss_of_coords)(atom37)
    return float(v), g


def _cos_sim(g_ref, g):
    a = g_ref.reshape(-1)
    b = g.reshape(-1)
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    return float(jnp.dot(a, b) / (na * nb + 1e-30))


def main():
    print("Loading 1ubq split into binder (res 1-40) + target (res 41-76)...")
    ps, atom37, n_res = _setup_split_pose_stack()
    print(f"  n_res={n_res}, atom37 shape={atom37.shape}")

    full_seq = jnp.zeros((n_res, 20), dtype=jnp.float32).at[:, 0].set(1.0)

    schemes = [
        ("baseline (beta_nov2016)",      None),
        ("electrostatics + Hbond",       {"fa_elec": 5.0, "hbond": 5.0}),
        ("shape complementarity (LJ)",   {"fa_ljatr": 5.0, "fa_ljrep": 5.0}),
        ("polar interface (zero LJ)",    {"fa_ljatr": 0.0, "fa_ljrep": 0.0,
                                          "fa_lk": 0.0}),
        ("water-mediated (LK-ball)",     {"lk_ball": 5.0, "lk_bridge": 5.0,
                                          "lk_bridge_uncpl": 5.0}),
        ("clash-only (fa_ljrep)",        {"fa_ljrep": 1.0, "fa_ljatr": 0.0,
                                          "fa_lk": 0.0, "fa_elec": 0.0,
                                          "hbond": 0.0}),
    ]

    print("\nComputing dG_interface and ∂dG/∂atom37 under each scheme...\n")
    results = []
    for label, w in schemes:
        v, g = _grad_under(ps, atom37, full_seq, w)
        results.append((label, v, g))
        print(f"  {label:32s}  dG = {v:+9.2f} kcal/mol   "
              f"max|g|={float(jnp.max(jnp.abs(g))):.3e}")

    # Cosine similarity table: how much does each scheme rotate the
    # gradient direction relative to baseline beta_nov2016?
    print("\nGradient cosine-similarity vs baseline (1.0 = same direction):")
    g_ref = results[0][2]
    for label, _, g in results[1:]:
        cs = _cos_sim(g_ref, g)
        flag = "  rescaling" if cs > 0.95 else "  ← different direction"
        print(f"  {label:32s}  cos = {cs:+.3f}{flag}")

    print("\nWhat this means inside a design loop:")
    print("  • Schemes with cos ≈ 1.0 just rescale the baseline gradient,")
    print("    so they steer the optimizer along the same axis as default")
    print("    beta_nov2016 — only the step size effectively changes.")
    print("  • Schemes with cos far from 1.0 push the optimizer toward a")
    print("    *different* family of binders (polar vs hydrophobic interface,")
    print("    water-bridged vs direct contact, etc.). These are the ones")
    print("    worth sweeping as a design hyperparameter.")


if __name__ == "__main__":
    main()
