"""Affinity-tuning demo: hot-residue identification from
``∂dG_interface/∂atom_coords``.

This is the jtmol-only artifact for affinity tuning. No mosaic, no
Boltz-2, no torch. Loads a binder+target PDB, computes
``dG_interface`` and its gradient w.r.t. atom-37 coords, then ranks
binder residues by per-residue gradient norm. Residues with high
``|∂dG/∂x|`` are the thermodynamically sensitive sites at the current
pose — those are the residues to mutate in an affinity-tuning
campaign.

We also report the per-term gradient breakdown for each hot residue,
which tells you *why* the residue is hot (clashing? missing an
Hbond? bad electrostatics?). That feeds directly into picking the
right amino-acid substitution.

The full saturation-mutagenesis ΔΔG scan would also live in this
script, but it needs rotamer packing on each mutant to give meaningful
numbers and that turns the demo from "10 seconds on CPU" into "30
minutes". The rotamer packer (``tmol_jax.pack``) does the right
thing; we just don't run it here.

Run:
    PYTHONPATH=. python examples/demo_affinity_tuning.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import tmol_jax
from tmol_jax.score.score_types import ScoreType
from tmol_jax.score import beta2016_score_function
from examples.integration_mosaic_hallucinator import (
    _ATOM37_NAMES,
    _build_atom37_to_flat_mapping,
    TmolInterfaceLoss,
)
import equinox as eqx
from jaxtyping import Array


PDB = "tmol_jax/tests/data/1ubq.pdb"
BINDER_RES_END = 40
TOP_N_HOT = 10


_PER_TERM_NAMES = (
    "fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec", "hbond",
    "lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl",
)


def _setup_split_pose_stack(pdb: str, split_res: int):
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

    flat_from_a37, is_mapped = _build_atom37_to_flat_mapping(ps)
    object.__setattr__(ps, "_atom37_map", jnp.asarray(flat_from_a37))
    object.__setattr__(ps, "_atom37_is_mapped", jnp.asarray(is_mapped))

    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)

    atom_chain = np.zeros(ps.coords.shape[1], dtype=np.int32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        atom_chain[int(offsets[r]):int(offsets[r]) + n] = 1 if r >= split_res else 0
    object.__setattr__(ps, "_atom_chain_id", jnp.asarray(atom_chain))

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
    return ps, jnp.asarray(atom37), n_res, bti, pbt, offsets, n_per_bt


class _StructOut(eqx.Module):
    structure_coordinates: Array
    backbone_coordinates: Array
    full_sequence: Array


def _per_term_grad(ps, atom37, full_seq, term_name):
    """Compute ∂E_term/∂atom37 in isolation (only one channel weighted)."""
    sfxn = beta2016_score_function()
    # Zero all weights, then set just this term.
    for st in ScoreType:
        if st is ScoreType.n_score_types:
            continue
        try:
            sfxn.set_weight(st, 0.0)
        except Exception:
            pass
    try:
        sfxn.set_weight(getattr(ScoreType, term_name), 1.0)
    except Exception:
        return None

    iface = TmolInterfaceLoss(ps, scale=1.0, binder_chain_id=0)
    # Swap in our single-term sfxn directly.
    object.__setattr__(iface, "sfxn", sfxn)
    object.__setattr__(iface, "state", sfxn.setup_for_pose_stack(ps))

    def loss(c):
        out = _StructOut(
            structure_coordinates=c,
            backbone_coordinates=c[:, [0, 1, 2, 4]],
            full_sequence=full_seq,
        )
        v, _ = iface(full_seq, out, key=jax.random.PRNGKey(0))
        return v

    return jax.grad(loss)(atom37)


def _name3(bt_idx, pbt):
    if bt_idx < 0:
        return "---"
    bt = pbt.active_block_types[bt_idx]
    return bt.name3 if hasattr(bt, "name3") else getattr(bt, "name", "???")[:3]


def main():
    print(f"Loading {PDB}, splitting at residue {BINDER_RES_END}...")
    ps, atom37, n_res, bti, pbt, offsets, n_per_bt = _setup_split_pose_stack(
        PDB, BINDER_RES_END
    )
    print(f"  n_res = {n_res}; binder = res 1-{BINDER_RES_END}, "
          f"target = res {BINDER_RES_END+1}-{n_res}")

    full_seq = jnp.zeros((n_res, 20), dtype=jnp.float32).at[:, 0].set(1.0)

    iface = TmolInterfaceLoss(ps, scale=1.0, binder_chain_id=0)

    def loss(c):
        out = _StructOut(
            structure_coordinates=c,
            backbone_coordinates=c[:, [0, 1, 2, 4]],
            full_sequence=full_seq,
        )
        v, _ = iface(full_seq, out, key=jax.random.PRNGKey(0))
        return v

    print("\nComputing dG_interface and the full-loss gradient...")
    dg = float(loss(atom37))
    g_total = jax.grad(loss)(atom37)
    print(f"  dG_interface = {dg:+.2f} kcal/mol")
    print(f"  max|g|       = {float(jnp.max(jnp.abs(g_total))):.2f}")

    # Per-residue gradient norm. Residues with high norm are
    # thermodynamically sensitive at the current pose.
    per_res = jnp.linalg.norm(g_total.reshape(n_res, -1), axis=-1)
    per_res = np.asarray(per_res)

    binder = per_res[:BINDER_RES_END]
    hot = np.argsort(-binder)[:TOP_N_HOT]

    print(f"\nTop {TOP_N_HOT} hot residues on the binder (highest |∂dG/∂x|):")
    print(f"  {'rank':>4s} {'res':>4s} {'aa':>4s} {'|grad|':>10s}")
    for rank, ri in enumerate(hot, start=1):
        aa3 = _name3(int(bti[ri]), pbt)
        print(f"  {rank:>4d} {ri+1:>4d} {aa3:>4s} {binder[ri]:>10.2f}")

    # Per-term gradient breakdown — for each hot residue, what's the
    # dominant contribution to |grad|? That tells you *what kind* of
    # mutation to try (clashing → bulkier vs smaller; missing Hbond →
    # introduce donor/acceptor; bad elec → flip charge; etc.).
    print(f"\nPer-term gradient norm at each hot residue (units: kcal/mol/Å):")
    print(f"  Building per-term gradients (jit-compiles {len(_PER_TERM_NAMES)} times)...")

    per_term_per_res = {}
    for term in _PER_TERM_NAMES:
        g_t = _per_term_grad(ps, atom37, full_seq, term)
        if g_t is None:
            continue
        norms = jnp.linalg.norm(g_t.reshape(n_res, -1), axis=-1)
        per_term_per_res[term] = np.asarray(norms)
        print(f"    {term:18s} max|.| on binder = {float(np.max(norms[:BINDER_RES_END])):.2f}")

    print(f"\nPer-residue per-term breakdown:")
    header = "  res aa   " + "".join(f"{t.replace('fa_',''):>10s}" for t in _PER_TERM_NAMES)
    print(header)
    for rank, ri in enumerate(hot, start=1):
        aa3 = _name3(int(bti[ri]), pbt)
        row = f"  {ri+1:>3d} {aa3:>3s}  "
        for term in _PER_TERM_NAMES:
            v = per_term_per_res.get(term)
            if v is None:
                row += f"{'-':>10s}"
            else:
                row += f"{v[ri]:>10.2f}"
        print(row)

    print(f"\nReading the table:")
    print(f"  * Residues with large fa_ljrep entries are clashing — mutate to a")
    print(f"    smaller side chain (e.g. → ALA, GLY) to relieve.")
    print(f"  * Residues with large hbond / fa_elec entries have unfulfilled")
    print(f"    polar contacts — mutate to introduce donor / acceptor / charge.")
    print(f"  * Residues with large fa_ljatr / fa_lk entries are in good shape")
    print(f"    contact already — mutate carefully, you'll likely break it.")
    print(f"\nNext step in a real campaign: feed top-3 hot residues into the")
    print(f"rotamer packer (tmol_jax.pack) for a saturation-mutagenesis ΔΔG")
    print(f"scan, then rank by predicted ΔΔG and send top mutants to wet lab.")


if __name__ == "__main__":
    main()
