"""Confirm the key finding — gradient cosine similarity across schemes —
on more PDBs to rule out 4-case artifact.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
import warnings
from dataclasses import dataclass
from typing import Optional, List

import jax
import jax.numpy as jnp
import numpy as np

warnings.filterwarnings("ignore")

SCHEMES = {
    "baseline":            None,
    "electrostatics_5x":   {"fa_elec": 5.0},
    "hbond_5x":            {"hbond": 5.0},
    "shape_2x":            {"fa_ljatr": 2.0, "fa_ljrep": 2.0},
    "water_bridged":       {"lk_ball": 3.0, "lk_bridge": 3.0, "lk_bridge_uncpl": 3.0},
    "polar_only":          {"fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0},
    # New schemes to probe whether direction change requires *zeroing* terms
    "lj_only":             {"fa_elec": 0.0, "hbond": 0.0, "lk_ball": 0.0,
                            "lk_bridge": 0.0, "lk_bridge_uncpl": 0.0},
    "hbond_only":          {"fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0,
                            "fa_elec": 0.0, "lk_ball": 0.0, "lk_bridge": 0.0,
                            "lk_bridge_uncpl": 0.0},
    "electrostatics_only": {"fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0,
                            "hbond": 0.0, "lk_ball": 0.0, "lk_bridge": 0.0,
                            "lk_bridge_uncpl": 0.0},
    "elec_heavy_10x":      {"fa_elec": 10.0},  # extreme rescale — does it change direction?
    "elec_heavy_100x":     {"fa_elec": 100.0},
}


@dataclass
class Case:
    name: str
    pdb_path: str
    split_at: Optional[int]


CASES: List[Case] = [
    Case("1ubq", "tmol_jax/tests/data/1ubq.pdb", 40),
    Case("1qys", "tmol_jax/tests/data/1qys.pdb", 45),
    Case("1r21", "tmol/tmol/tests/data/pdb/1r21.pdb", 50),
    Case("3plc", "tmol/tmol/tests/data/pdb/3plc.pdb", 27),
    Case("6DMZ_A", "tmol/tmol/tests/data/pdb/6DMZ_A.pdb", 23),
    # 1BL8 omitted (tetramer edge case); add more monomers we split
    Case("1ubq_split_30", "tmol_jax/tests/data/1ubq.pdb", 30),
    Case("1ubq_split_50", "tmol_jax/tests/data/1ubq.pdb", 50),
    Case("1qys_split_30", "tmol_jax/tests/data/1qys.pdb", 30),
    Case("1qys_split_60", "tmol_jax/tests/data/1qys.pdb", 60),
]


def clear_caches():
    import gc
    from tmol_jax.score.common import count_pair as _cp
    for attr in ("_BPL_CACHE", "_BPL_DIRECT_CACHE", "_POSE_INDEX_CACHE"):
        c = getattr(_cp, attr, None)
        if isinstance(c, dict):
            c.clear()
    try:
        from tmol_jax.score.hbond import params as _hbp
        for attr in ("_HBOND_PAIR_TABLES_CACHE", "_hbond_per_bt_cache"):
            c = getattr(_hbp, attr, None)
            if isinstance(c, dict):
                c.clear()
    except Exception:
        pass
    gc.collect()
    jax.clear_caches()


def load_case(case: Case):
    import tmol_jax
    clear_caches()
    ps = tmol_jax.pose_stack_from_pdb(case.pdb_path)
    n_res = int(ps.n_res_per_pose[0])
    split = min(case.split_at, n_res - 4)
    irc = np.array(ps.inter_residue_connections, copy=True)
    ibs = np.array(ps.inter_block_bondsep, copy=True)
    if split >= 1 and split < n_res and irc.shape[2] >= 2:
        irc[0, split - 1, 1] = -1
        irc[0, split, 0] = -1
        ibs[0, split - 1, split, :, :] = 6
        ibs[0, split, split - 1, :, :] = 6
    return ps.replace(
        inter_residue_connections=jnp.asarray(irc),
        inter_block_bondsep=jnp.asarray(ibs),
    ), split


def binder_mask(ps, split_res):
    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)
    mask = np.zeros(ps.coords.shape[1], dtype=bool)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        if r < split_res:
            mask[int(offsets[r]) : int(offsets[r]) + int(n_per_bt[bt])] = True
    return jnp.asarray(mask)


def scheme_grad(ps, split_res, weights):
    import tmol_jax
    from tmol_jax.score.score_types import ScoreType
    sfxn = tmol_jax.beta2016_score_function()
    if weights is not None:
        for name, w in weights.items():
            st = getattr(ScoreType, name, None)
            if st is not None:
                sfxn.set_weight(st, float(w))
    state = sfxn.setup_for_pose_stack(ps)
    bmask = binder_mask(ps, split_res)
    shift = jnp.where(bmask[:, None], jnp.asarray([500.0, 0.0, 0.0]), 0.0)

    @jax.jit
    def dG(coords):
        return (sfxn(ps, state=state, coords_override=coords).sum()
                - sfxn(ps, state=state,
                       coords_override=coords.at[0].set(coords[0] + shift)).sum())

    g = jax.grad(dG)(ps.coords)
    g_binder = jnp.where(bmask[None, :, None], g, 0.0)
    return g_binder


def cosine(a, b):
    af = a.reshape(-1)
    bf = b.reshape(-1)
    na = float(jnp.linalg.norm(af))
    nb = float(jnp.linalg.norm(bf))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(jnp.dot(af, bf) / (na * nb))


def main():
    import tmol_jax
    print(f"# Cosine similarity sweep ({len(CASES)} cases × {len(SCHEMES)} schemes)\n")

    all_grads = {}
    for case in CASES:
        try:
            ps, split = load_case(case)
        except Exception as e:
            print(f"skip {case.name}: {e}")
            continue
        for scheme_name, weights in SCHEMES.items():
            try:
                g = scheme_grad(ps, split, weights)
                all_grads[(case.name, scheme_name)] = g
            except Exception as e:
                print(f"  skip {case.name}/{scheme_name}: {type(e).__name__}")
        clear_caches()

    # Cosine matrix averaged across cases, with std
    names = list(SCHEMES.keys())
    n = len(names)
    mean = np.zeros((n, n))
    std = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                mean[i, j] = 1.0
                continue
            cs = []
            for case in CASES:
                ga = all_grads.get((case.name, a))
                gb = all_grads.get((case.name, b))
                if ga is None or gb is None:
                    continue
                c = cosine(ga, gb)
                if np.isfinite(c):
                    cs.append(c)
            if cs:
                mean[i, j] = np.mean(cs)
                std[i, j] = np.std(cs)

    print("\n## Mean cosine similarity (all cases)\n")
    print("| " + " | ".join(["scheme"] + names) + " |")
    print("| --- | " + " | ".join(["---:"] * n) + " |")
    for i, a in enumerate(names):
        row = [a]
        for j in range(n):
            row.append(f"{mean[i,j]:.2f}")
        print("| " + " | ".join(row) + " |")

    print("\n## Std across cases (tells us if the cosine is stable)\n")
    print("| scheme | max std vs others |")
    print("| --- | ---: |")
    for i, a in enumerate(names):
        max_std = max(std[i, j] for j in range(n) if j != i)
        print(f"| {a} | {max_std:.3f} |")

    # Key finding: does zeroing terms always produce low cos to baseline?
    print("\n## Cos similarity of each scheme to baseline")
    print("(schemes that zero terms diverge; schemes that rescale don't)\n")
    print("| scheme | cos-sim to baseline | mean |∇| |")
    print("| --- | ---: | ---: |")
    i_base = names.index("baseline")
    for i, name in enumerate(names):
        if i == i_base:
            continue
        cs = []
        gnorms = []
        for case in CASES:
            ga = all_grads.get((case.name, "baseline"))
            gb = all_grads.get((case.name, name))
            if ga is None or gb is None:
                continue
            c = cosine(ga, gb)
            if np.isfinite(c):
                cs.append(c)
                gnorms.append(float(jnp.linalg.norm(gb)))
        if cs:
            print(f"| {name} | {np.mean(cs):.3f}±{np.std(cs):.3f} "
                  f"| {np.mean(gnorms):.0f} |")


if __name__ == "__main__":
    main()
