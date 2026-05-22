"""H1 sweep — static analysis (no confounded relaxation).

Two tests:

1. **Per-scheme dG on the native complex.** Does each scheme assign a
   measurably different interface energy to the same structure? This
   is the basic "reweighting changes the objective" sanity check.

2. **Gradient direction analysis.** Compute ∇E_scheme w.r.t. binder
   atom coords. Measure pairwise cosine similarity of scheme
   gradients. Schemes with dissimilar gradients push designs in
   different directions — that's the mechanism by which reweighting
   shifts what the hallucinator finds.

3. **Per-term native breakdown per case.** Fixed-structure decomposition
   — shows which physical channel dominates the interface, which weights
   would matter for reweighting.

No relaxation; no clash dynamics. Pure analysis on native interfaces.
Runs in under a minute locally.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SCHEMES = {
    "baseline":            None,
    "electrostatics_5x":   {"fa_elec": 5.0},
    "hbond_5x":            {"hbond": 5.0},
    "shape_2x":            {"fa_ljatr": 2.0, "fa_ljrep": 2.0},
    "water_bridged":       {"lk_ball": 3.0, "lk_bridge": 3.0,
                            "lk_bridge_uncpl": 3.0},
    "polar_only":          {"fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0},
}


@dataclass
class Case:
    name: str
    pdb_path: str
    split_at: Optional[int]
    description: str


CASES: List[Case] = [
    Case("1ubq", "tmol_jax/tests/data/1ubq.pdb", 40, "ubiquitin, split at domain"),
    Case("1qys", "tmol_jax/tests/data/1qys.pdb", 45, "top7 design, split mid-chain"),
    Case("1r21", "tmol/tmol/tests/data/pdb/1r21.pdb", 50, "NMR, 3 Cys"),
    Case("3plc", "tmol/tmol/tests/data/pdb/3plc.pdb", 27, "disulfide-rich"),
    Case("1BL8", "tmol/tmol/tests/data/pdb/1BL8.pdb", None, "KcsA tetramer, true multi-chain"),
]


# Score types we care about for per-term decomposition
INTERFACE_ST_NAMES = [
    "fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec", "hbond",
    "lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl",
]


def clear_all_caches():
    import gc
    from tmol_jax.score.common import count_pair as _cp
    for attr in ("_BPL_CACHE", "_BPL_DIRECT_CACHE", "_POSE_INDEX_CACHE"):
        cache = getattr(_cp, attr, None)
        if isinstance(cache, dict):
            cache.clear()
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
    clear_all_caches()
    ps = tmol_jax.pose_stack_from_pdb(case.pdb_path)
    n_res = int(ps.n_res_per_pose[0])

    if case.split_at is None:
        pdb_info = getattr(ps, "pdb_info", None)
        split = None
        if pdb_info is not None and pdb_info.chain_id is not None:
            chain_labels = pdb_info.chain_id[0]
            first_label = None
            for r, lbl in enumerate(chain_labels):
                if int(ps.block_type_ind[0, r]) < 0:
                    continue
                if first_label is None:
                    first_label = lbl
                elif lbl != first_label:
                    split = r
                    break
        if split is None:
            split = n_res // 2
        return ps, split, n_res

    split = min(case.split_at, n_res - 4)
    irc = np.array(ps.inter_residue_connections, copy=True)
    ibs = np.array(ps.inter_block_bondsep, copy=True)
    if split >= 1 and split < n_res and irc.shape[2] >= 2:
        irc[0, split - 1, 1] = -1
        irc[0, split, 0] = -1
        ibs[0, split - 1, split, :, :] = 6
        ibs[0, split, split - 1, :, :] = 6
    ps = ps.replace(
        inter_residue_connections=jnp.asarray(irc),
        inter_block_bondsep=jnp.asarray(ibs),
    )
    return ps, split, n_res


def make_sfxn(ps, term_weights):
    import tmol_jax
    from tmol_jax.score.score_types import ScoreType
    sfxn = tmol_jax.beta2016_score_function()
    if term_weights is not None:
        for name, w in term_weights.items():
            st = getattr(ScoreType, name, None)
            if st is not None:
                sfxn.set_weight(st, float(w))
    state = sfxn.setup_for_pose_stack(ps)
    return sfxn, state


def binder_mask(ps, split_res: int) -> jax.Array:
    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)
    mask = np.zeros(ps.coords.shape[1], dtype=bool)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        start = int(offsets[r])
        if r < split_res:
            mask[start:start + n] = True
    return jnp.asarray(mask)


def per_term_interface(ps, split_res: int, term_names: List[str]) -> Dict[str, float]:
    """Compute interface per-term energy = E_bound - E_separated, per term."""
    import tmol_jax
    from tmol_jax.score.score_types import ScoreType
    from tmol_jax.score.score_function import ScoreFunction

    # Build a score function with all terms at weight 1.0 (so
    # score_by_term returns raw per-term energies)
    sfxn = ScoreFunction(tmol_jax.ParameterDatabase.get_default())
    for name in term_names:
        st = getattr(ScoreType, name, None)
        if st is not None:
            sfxn.set_weight(st, 1.0)
    state = sfxn.setup_for_pose_stack(ps)

    bmask = binder_mask(ps, split_res)
    shift = jnp.where(bmask[:, None], jnp.asarray([500.0, 0.0, 0.0]), 0.0)
    coords_bound = ps.coords
    coords_sep = ps.coords.at[0].set(ps.coords[0] + shift)

    per_bound = sfxn.score_by_term(ps, state=state, coords_override=coords_bound)
    per_sep = sfxn.score_by_term(ps, state=state, coords_override=coords_sep)

    out = {}
    for name in term_names:
        st = getattr(ScoreType, name, None)
        if st is None:
            continue
        eb = float(jnp.asarray(per_bound.get(st, jnp.zeros(1))).reshape(-1)[0])
        es = float(jnp.asarray(per_sep.get(st, jnp.zeros(1))).reshape(-1)[0])
        out[name] = eb - es
    return out


def scheme_energy_and_grad(ps, split_res: int, weights):
    """Return (dG_interface, grad_wrt_binder_coords) for one scheme."""
    sfxn, state = make_sfxn(ps, weights)
    bmask = binder_mask(ps, split_res)
    shift = jnp.where(bmask[:, None], jnp.asarray([500.0, 0.0, 0.0]), 0.0)
    coords_bound = ps.coords
    coords_sep = ps.coords.at[0].set(ps.coords[0] + shift)

    @jax.jit
    def dG(coords):
        Eb = sfxn(ps, state=state, coords_override=coords).sum()
        Es = sfxn(ps, state=state,
                  coords_override=coords.at[0].set(coords[0] + shift)).sum()
        return Eb - Es

    dg_val = float(dG(coords_bound))
    g = jax.grad(dG)(coords_bound)   # [1, N, 3]
    # Restrict gradient to binder atoms
    g_binder = jnp.where(bmask[None, :, None], g, 0.0)
    return dg_val, g_binder


def cosine(a: jax.Array, b: jax.Array) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    na = float(jnp.linalg.norm(a_flat))
    nb = float(jnp.linalg.norm(b_flat))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(jnp.dot(a_flat, b_flat) / (na * nb))


def main():
    import tmol_jax
    print("# tmol_jax H1 analysis: static interface + gradient-direction\n")
    print(f"- Platform: {jax.devices()}")
    print(f"- Cases: {[c.name for c in CASES]}")
    print(f"- Schemes: {list(SCHEMES.keys())}\n")

    all_rows = []
    all_grads = {}   # (case, scheme) -> [1, N, 3]
    per_term_tables = {}

    t_total = time.perf_counter()
    for case in CASES:
        print(f"---- case: {case.name} ({case.description}) ----")
        try:
            ps, split, n_res = load_case(case)
        except Exception as e:
            print(f"  load failed: {e}")
            continue

        # Per-term decomposition (scheme-independent; just the interface
        # decomposition on this complex with unit weights)
        try:
            pt = per_term_interface(ps, split, INTERFACE_ST_NAMES)
            per_term_tables[case.name] = pt
            dominant = sorted(pt.items(), key=lambda x: x[1])[:3]
            print(f"  per-term interface (kcal/mol):")
            for n, v in pt.items():
                print(f"    {n:20s} {v:+8.2f}")
            print(f"  (top 3 stabilising: {', '.join(n for n,_ in dominant)})")
        except Exception as e:
            print(f"  per-term failed: {e}")

        # Per-scheme dG and gradient
        print(f"  per-scheme dG (scheme reweights of same structure):")
        for scheme_name, weights in SCHEMES.items():
            try:
                dg, g = scheme_energy_and_grad(ps, split, weights)
                all_rows.append({
                    "case": case.name, "scheme": scheme_name,
                    "dG": dg,
                    "grad_norm": float(jnp.linalg.norm(g)),
                })
                all_grads[(case.name, scheme_name)] = g
                print(f"    {scheme_name:20s} dG={dg:+9.2f}  |∇|={float(jnp.linalg.norm(g)):.1f}")
            except Exception as e:
                print(f"    {scheme_name:20s} FAILED ({type(e).__name__}: {e})")

        clear_all_caches()

    wall = time.perf_counter() - t_total
    print(f"\nWall-clock: {wall:.1f}s\n")

    # -------- Paper Table 1: Per-scheme dG across cases --------
    print("## Paper Table 1 — Per-scheme interface dG (kcal/mol)")
    print("| case | " + " | ".join(SCHEMES.keys()) + " |")
    print("| --- | " + " | ".join(["---:"] * len(SCHEMES)) + " |")
    for case in CASES:
        cells = [case.name]
        for scheme_name in SCHEMES:
            r = next((x for x in all_rows if x["case"] == case.name
                      and x["scheme"] == scheme_name), None)
            cells.append(f"{r['dG']:+8.1f}" if r else "—")
        print("| " + " | ".join(cells) + " |")

    # -------- Paper Table 2: Gradient-direction cosine similarity --------
    print("\n## Paper Table 2 — Cosine similarity of ∇E_scheme w.r.t. binder coords")
    print("(mean across cases, pairwise; 1.0 = same direction, 0 = orthogonal)\n")
    scheme_names = list(SCHEMES.keys())
    print("| | " + " | ".join(scheme_names) + " |")
    print("| --- | " + " | ".join(["---:"] * len(scheme_names)) + " |")
    for i, a in enumerate(scheme_names):
        row = [a]
        for j, b in enumerate(scheme_names):
            if i == j:
                row.append("1.000")
                continue
            cosines = []
            for case in CASES:
                ga = all_grads.get((case.name, a))
                gb = all_grads.get((case.name, b))
                if ga is None or gb is None:
                    continue
                c = cosine(ga, gb)
                if np.isfinite(c):
                    cosines.append(c)
            if cosines:
                row.append(f"{np.mean(cosines):.3f}")
            else:
                row.append("—")
        print("| " + " | ".join(row) + " |")

    # -------- Paper Table 3: Per-term decomposition (where does affinity come from?) --------
    print("\n## Paper Table 3 — Per-term interface dG on native (kcal/mol, unit weights)")
    header = ["case"] + INTERFACE_ST_NAMES
    print("| " + " | ".join(header) + " |")
    print("| --- | " + " | ".join(["---:"] * len(INTERFACE_ST_NAMES)) + " |")
    for case in CASES:
        pt = per_term_tables.get(case.name)
        if pt is None:
            continue
        cells = [case.name] + [f"{pt[n]:+6.1f}" for n in INTERFACE_ST_NAMES]
        print("| " + " | ".join(cells) + " |")

    # -------- Persist CSV --------
    import csv, os
    os.makedirs("docs", exist_ok=True)
    with open("docs/h1_analysis_dg.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["case", "scheme", "dG", "grad_norm"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open("docs/h1_analysis_perterm.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["case"] + INTERFACE_ST_NAMES)
        w.writeheader()
        for case_name, pt in per_term_tables.items():
            w.writerow({"case": case_name, **pt})
    print(f"\nCSVs: docs/h1_analysis_dg.csv  docs/h1_analysis_perterm.csv")


if __name__ == "__main__":
    main()
