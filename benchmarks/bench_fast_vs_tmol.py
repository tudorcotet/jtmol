"""Speed benchmark: jtmol's fast nonbonded engine vs upstream tmol, full-22
beta2016 score, forward and value+grad, across three batched-scoring regimes.

Regimes
-------
S1  one structure, one conformation (K=1)            single-pose latency
S2  one topology, K conformations (K=128)            same-topology ensemble
    (e.g. an MD / relax trajectory; one shared neighbor list)
S3  K distinct structures (K=128)                     heterogeneous batch
    (e.g. scoring many designs; a ragged neighbor list)

jtmol uses ``jtmol.nonbonded.fast`` (parity-exact vs the dense substrate; see
``jtmol/tests/test_nonbonded_fast.py``). Timing is warm, CUDA-synchronized,
ABBA-interleaved per rep (cancels drift), setup excluded. Value+grad is the
gradient of the summed energy w.r.t. coordinates -- what an optimizer needs.

Requires a CUDA GPU with jax[cuda12] and the tmol wheel installed. S1/S2 use the
bundled 1UBQ; S3 needs a directory of distinct PDBs (``--pdb-dir``), e.g. the 128
structures listed in ``benchmarks/upstream_exact_panel_v2.json``.

    python benchmarks/bench_fast_vs_tmol.py --pdb-dir /path/to/pdbs
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

NB9 = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec", "lk_ball_iso", "lk_ball",
       "lk_bridge", "lk_bridge_uncpl", "hbond")
WEIGHTS = {
    "fa_ljatr": 1.0, "fa_ljrep": 0.55, "fa_lk": 1.0, "fa_elec": 1.0, "hbond": 1.0,
    "cart_lengths": 0.5, "cart_angles": 0.5, "cart_torsions": 0.5, "cart_impropers": 0.5,
    "cart_hxltorsions": 0.5, "disulfide": 1.25, "omega": 0.48, "rama": 0.50,
    "dunbrack_rot": 0.76, "dunbrack_rotdev": 0.69, "dunbrack_semirot": 0.78,
    "lk_ball_iso": -0.38, "lk_ball": 0.92, "lk_bridge": -0.33, "lk_bridge_uncpl": -0.33,
    "ref": 1.0, "gen_torsions": 1.0,
}
NP13 = tuple(n for n in WEIGHTS if n not in NB9)


def _abba(jt_fn, tm_fn, k, reps, jax, torch):
    """Interleaved warm timing; returns (jtmol median ms/pose, tmol median ms/pose)."""
    for _ in range(8):
        jax.block_until_ready(jt_fn()); tm_fn(); torch.cuda.synchronize()
    tj, tt = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(jt_fn()); tj.append((time.perf_counter() - t0) * 1000 / k)
        t0 = time.perf_counter(); tm_fn(); torch.cuda.synchronize(); tt.append((time.perf_counter() - t0) * 1000 / k)
    return float(statistics.median(tj)), float(statistics.median(tt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", type=str, default=None, help="directory of distinct PDBs for S3")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n-s3", type=int, default=64,
                    help="how many distinct structures for S3 (128 needs the S3 remat work)")
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    import jax; jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp
    import torch
    import jtmol
    from jtmol.database import ParameterDatabase
    from jtmol.score.score_types import ScoreType as JST
    from jtmol.score.score_function import ScoreFunction
    from jtmol.score.batched import batched_energy
    from jtmol.nonbonded.multipose import setup_s2, setup_s3
    from jtmol.nonbonded.fast import build_fast_nonbonded, build_fast_nonbonded_ragged
    from tmol.io import pose_stack_from_pdb as tload
    from tmol.pose.pose_stack_builder import PoseStackBuilder as TPSB
    from tmol.score import beta2016_score_function

    dev = torch.device("cuda"); db = ParameterDatabase.get_default()
    data = Path(jtmol.__file__).parent / "tests" / "data"
    UBQ = str(data / "1ubq.pdb")
    rng = np.random.default_rng(0)
    rows = []

    def tmol_full(pdb_list, coords_override=None):
        tps = TPSB.from_poses([tload(p, dev) for p in pdb_list], dev)
        sc = beta2016_score_function(dev).render_whole_pose_scoring_module(tps)
        coords = tps.coords if coords_override is None else coords_override
        cg = coords.detach().clone().requires_grad_(True)
        return (lambda: sc(coords).sum().detach(),
                lambda: torch.autograd.grad(sc(cg).sum(), cg))

    # ---- S1 / S2: one topology, K in {1, args.k} ----
    ps = jtmol.pose_stack_from_pdb(UBQ)
    base = np.asarray(ps.coords, dtype=np.float32)[0]
    npair = ScoreFunction(db)
    for n in NP13:
        npair.set_weight(getattr(JST, n), WEIGHTS[n])
    npstate = npair.setup_for_pose_stack(ps)

    for tag, K in (("S1 (K=1)", 1), (f"S2 (K={args.k} same-topology)", args.k)):
        cPN3 = np.stack([base + (0 if K == 1 else rng.normal(scale=0.1, size=base.shape).astype(np.float32)) for _ in range(K)])
        coords = jnp.asarray(cPN3)
        fast = build_fast_nonbonded(setup_s2(ps, cPN3, db))
        np_fn = batched_energy(npair, ps, state=npstate, chunk_size=min(16, K))
        def full(c, fast=fast, np_fn=np_fn):
            d = fast(c); nb = sum(jnp.sum(d[getattr(JST, n)]) * WEIGHTS[n] for n in NB9)
            return nb + jnp.sum(np_fn(c[:, None, :, :]))
        jt_v = jax.jit(jax.value_and_grad(full)); jt_f = jax.jit(full)
        jax.block_until_ready(jt_v(coords))
        tps = TPSB.from_poses([tload(UBQ, dev) for _ in range(K)], dev)
        sc = beta2016_score_function(dev).render_whole_pose_scoring_module(tps)
        # Fairness: both backends must score K DISTINCT conformations, not K
        # identical native copies. Perturb tmol's own native coords by the same
        # magnitude jtmol uses (independent draw; atom orders differ between
        # backends, so we match the workload distribution, not the exact bytes).
        if K == 1:
            tco = tps.coords.detach().clone()
        else:
            base_t = tps.coords[0]
            pert = rng.normal(scale=0.1, size=(K,) + tuple(base_t.shape)).astype(np.float32)
            tco = base_t[None] + torch.as_tensor(pert, device=dev)
        cg = tco.detach().clone().requires_grad_(True)
        fjt, ftm = _abba(lambda: jt_f(coords), lambda: sc(tco).sum().detach(), K, args.reps, jax, torch)
        vjt, vtm = _abba(lambda: jt_v(coords), lambda: torch.autograd.grad(sc(cg).sum(), cg), K, args.reps, jax, torch)
        rows.append((tag, ftm / fjt, vtm / vjt))

    # ---- S3: distinct topologies (fused full-22 on a shared coord variable) ----
    if args.pdb_dir:
        pdbs = sorted(str(p) for p in Path(args.pdb_dir).glob("*.pdb"))
        jt_ps, tm_ps = [], []
        for p in pdbs:
            if len(jt_ps) >= args.n_s3:
                break
            try:
                j = jtmol.pose_stack_from_pdb(p); t = tload(p, dev)
            except Exception:
                continue
            jt_ps.append(j); tm_ps.append(t)
        K = len(jt_ps)
        b3 = setup_s3(jt_ps, db)
        fast3 = build_fast_nonbonded_ragged(b3)
        Ns = [int(np.asarray(x.coords).shape[1]) for x in jt_ps]; maxA = max(Ns)
        padded = np.zeros((K, maxA, 3), np.float32); flat_map = np.zeros(int(b3.SumN), np.int32)
        for i in range(K):
            off = int(b3.offsets[i]); n = Ns[i]
            padded[i, :n] = np.asarray(jt_ps[i].coords[0], dtype=np.float32)
            flat_map[off:off + n] = i * maxA + np.arange(n, dtype=np.int32)
        # non-pairwise on the padded native stack; nonbonded on ragged flat coords
        # gathered from the same padded variable -> one gradient flows to both.
        from benchmarks.exact_upstream_matrix import _build_jtmol_native_stack
        native = _build_jtmol_native_stack(jt_ps, padded, jnp=jnp)
        npair3 = ScoreFunction(db)
        for n in NP13:
            npair3.set_weight(getattr(JST, n), WEIGHTS[n])
        st3 = npair3.setup_for_pose_stack(native)
        fm = jnp.asarray(flat_map); pc = jnp.asarray(padded)
        def full3(padded_c):
            flat_c = padded_c.reshape(K * maxA, 3)[fm]
            d = fast3(flat_c); nb = sum(jnp.sum(d[getattr(JST, n)]) * WEIGHTS[n] for n in NB9)
            return nb + jnp.sum(npair3(native, state=st3, coords_override=padded_c).astype(jnp.float32))
        jt_v3 = jax.jit(jax.value_and_grad(full3)); jax.block_until_ready(jt_v3(pc))
        jt_f3 = jax.jit(full3); jax.block_until_ready(jt_f3(pc))
        tps3 = TPSB.from_poses(tm_ps, dev)
        sc3 = beta2016_score_function(dev).render_whole_pose_scoring_module(tps3)
        cg3 = tps3.coords.detach().clone().requires_grad_(True)
        fjt3, ftm3 = _abba(lambda: jt_f3(pc), lambda: sc3(tps3.coords).sum().detach(), K, args.reps, jax, torch)
        vjt, vtm = _abba(lambda: jt_v3(pc), lambda: torch.autograd.grad(sc3(cg3).sum(), cg3), K, args.reps, jax, torch)
        rows.append((f"S3 (K={K} distinct)", ftm3 / fjt3, vtm / vjt))
    else:
        print("(skipping S3 -- pass --pdb-dir with distinct PDBs to run it)")

    print("\nfull-22 beta2016, jtmol fast engine vs tmol (ratio > 1 = jtmol faster):")
    print(f"  {'regime':<28} {'forward':>10} {'value+grad':>12}")
    for tag, fwd, vg in rows:
        fs = "  n/a" if fwd != fwd else f"{fwd:.2f}x"
        print(f"  {tag:<28} {fs:>10} {vg:>11.2f}x")


if __name__ == "__main__":
    main()
