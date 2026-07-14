"""Full 22-channel beta2016 on the Verlet substrate = the 9 nonbonded channels
(jtmol/nonbonded/{ljlk,elec,lkball,hbond}.py, uncapped adaptive Verlet edges) +
the 13 non-pairwise channels (jtmol's per-residue terms, unchanged: dunbrack, rama,
omega, cart_*, disulfide, ref, gen_torsions).

Correctness only, single-pose (P==1). The non-pairwise terms are per-residue O(N) and
need no neighbor list, so they are taken verbatim from jtmol's dense score function; only
the *pairwise* half rides the exact uncapped substrate. Verified == jtmol dense full score.

ponytail: non-pairwise via score_by_term recomputes the (discarded) dense nonbonded too --
wasteful but correctness-clean; a fused build would compute the non-pairwise alone.
"""
from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jtmol.score.score_types import ScoreType as JST
from jtmol.nonbonded.ljlk import setup_ljlk_on_verlet, ljlk_energy_on_verlet
from jtmol.nonbonded.elec import setup_elec_on_verlet, elec_energy_on_verlet
from jtmol.nonbonded.lkball import setup_lkball_on_verlet, lkball_energy_on_verlet
from jtmol.nonbonded.hbond import setup_hbond_on_verlet, hbond_energy_on_verlet

OFFICIAL_WEIGHTS = {
    "fa_ljatr": 1.0, "fa_ljrep": 0.55, "fa_lk": 1.0, "fa_elec": 1.0, "hbond": 1.0,
    "cart_lengths": 0.5, "cart_angles": 0.5, "cart_torsions": 0.5, "cart_impropers": 0.5,
    "cart_hxltorsions": 0.5, "disulfide": 1.25, "omega": 0.48, "rama": 0.50,
    "dunbrack_rot": 0.76, "dunbrack_rotdev": 0.69, "dunbrack_semirot": 0.78,
    "lk_ball_iso": -0.38, "lk_ball": 0.92, "lk_bridge": -0.33, "lk_bridge_uncpl": -0.33,
    "ref": 1.0, "gen_torsions": 1.0,
}
_LJLK = ("fa_ljatr", "fa_ljrep", "fa_lk")
_LKBALL = ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl")
NONBONDED = _LJLK + ("fa_elec",) + _LKBALL + ("hbond",)
NONPAIRWISE = tuple(n for n in OFFICIAL_WEIGHTS if n not in NONBONDED)


@dataclass
class FullScoreState:
    sfxn: object
    pose_stack: object
    lt: object; ls: object
    et: object; es: object
    kt: object; ks: object
    ht: object; hs: object


def setup_full_on_verlet(pose_stack, param_db=None) -> FullScoreState:
    import jtmol
    sfxn = jtmol.beta2016_score_function(param_db=param_db, use_neighbor_list=False)
    # gen_bond_length/angle are jtmol extensions, not upstream beta2016 (harness disables them)
    sfxn.set_weight(JST.gen_bond_length, 0.0)
    sfxn.set_weight(JST.gen_bond_angle, 0.0)
    lt, ls, _, _ = setup_ljlk_on_verlet(pose_stack, param_db)
    et, es, _, _ = setup_elec_on_verlet(pose_stack, param_db)
    kt, ks, _, _ = setup_lkball_on_verlet(pose_stack, param_db)
    ht, hs, _, _ = setup_hbond_on_verlet(pose_stack, param_db)
    return FullScoreState(sfxn, pose_stack, lt, ls, et, es, kt, ks, ht, hs)


def full_energy(fs: FullScoreState, coords) -> jax.Array:
    """Total weighted beta2016 energy (per pose, summed). Differentiable in coords."""
    W = OFFICIAL_WEIGHTS
    e = 0.0
    ljlk = ljlk_energy_on_verlet(fs.lt, fs.ls, coords)
    for ch in _LJLK:
        e = e + jnp.sum(ljlk[getattr(JST, ch)]) * W[ch]
    e = e + jnp.sum(elec_energy_on_verlet(fs.et, fs.es, coords)[JST.fa_elec]) * W["fa_elec"]
    lkb = lkball_energy_on_verlet(fs.kt, fs.ks, coords)
    for ch in _LKBALL:
        e = e + jnp.sum(lkb[getattr(JST, ch)]) * W[ch]
    e = e + jnp.sum(hbond_energy_on_verlet(fs.hs, coords)[JST.hbond]) * W["hbond"]
    # non-pairwise: jtmol per-residue terms, weighted (score_by_term applies weights)
    v = fs.sfxn.score_by_term(fs.pose_stack, coords_override=coords)
    for ch in NONPAIRWISE:
        e = e + jnp.sum(v[getattr(JST, ch)])
    return e


def demo() -> None:
    """Assert full-beta2016-on-Verlet == jtmol dense full beta2016 (energy + gradient), 1ubq."""
    import os
    jax.config.update("jax_enable_x64", True)
    import jtmol
    from jtmol.database import ParameterDatabase

    pdb = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data", "1ubq.pdb")
    ps = jtmol.pose_stack_from_pdb(pdb)
    db = ParameterDatabase.get_default()
    coords = jnp.asarray(np.asarray(ps.coords, dtype=np.float64))

    fs = setup_full_on_verlet(ps, db)
    # dense reference: the SAME beta2016 score function's total (gen_bond disabled to match)
    dense_sfxn = jtmol.beta2016_score_function(param_db=db, use_neighbor_list=False)
    dense_sfxn.set_weight(JST.gen_bond_length, 0.0)
    dense_sfxn.set_weight(JST.gen_bond_angle, 0.0)

    e_full = float(full_energy(fs, coords))
    e_dense = float(jnp.sum(dense_sfxn(ps, coords_override=coords)))
    rel = abs(e_full - e_dense) / (abs(e_dense) + 1e-12)
    print(f"[full-on-verlet] 1ubq total beta2016 (22ch, gen_bond off):")
    print(f"  verlet-substrate : {e_full:+.6f}")
    print(f"  jtmol dense      : {e_dense:+.6f}")
    print(f"  rel = {rel:.2e}")
    assert rel < 1e-6, f"full-score parity FAILED: {rel}"

    g_full = np.asarray(jax.grad(lambda c: full_energy(fs, c))(coords))
    g_dense = np.asarray(jax.grad(lambda c: jnp.sum(dense_sfxn(ps, coords_override=c)))(coords))
    gmax = float(np.abs(g_full - g_dense).max())
    print(f"  gradient max|full - dense| = {gmax:.2e}")
    assert gmax < 1e-4, f"full gradient parity FAILED: {gmax}"
    print("\nFull 22-channel beta2016 on the Verlet substrate == jtmol dense full score "
          "(energy + gradient). Nonbonded on the uncapped substrate + non-pairwise per-residue: DONE.")


if __name__ == "__main__":
    demo()
