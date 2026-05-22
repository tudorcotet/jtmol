"""Dense vs cell-list LJ/LK benchmark on 1UBQ.

Compares the two ``LJLKEnergyTerm`` evaluation paths:

* ``use_neighbor_list=False`` -- the default. Materialises a
  ``[P, N, N]`` distance / parameter / bonded-path-length tensor.
* ``use_neighbor_list=True``  -- builds a half-list of neighbours
  within ``cutoff`` A during ``setup_for_pose_stack`` and evaluates
  LJ/LK only on those pairs.

For 1UBQ (1228 real atoms / 1231 slots, 6 A cutoff) the two paths
should agree within float32 roundoff; the cell-list pays back its
setup cost inside the pair loop by avoiding the ``O(N^2)`` pair
tensors. Larger systems (~5000 atoms) benefit more — the dense tensors
dominate GPU memory long before arithmetic becomes the bottleneck.

Run it:

    python examples/neighbor_list_benchmark.py
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp


DEFAULT_PDB = (
    Path(__file__).resolve().parent.parent
    / "tmol_jax"
    / "tests"
    / "data"
    / "1ubq.pdb"
)
INPUT_PDB = Path(os.environ.get("TMOL_JAX_PDB", str(DEFAULT_PDB)))
CUTOFF = float(os.environ.get("TMOL_JAX_CUTOFF", "6.0"))
N_WARM = int(os.environ.get("TMOL_JAX_WARM", "3"))
N_BENCH = int(os.environ.get("TMOL_JAX_BENCH", "20"))


def _banner(title: str) -> None:
    line = "-" * 72
    print(f"\n{line}\n{title}\n{line}")


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.2f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.2f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def _time_call(fn, n_warm: int, n_bench: int) -> tuple[float, float]:
    """Return ``(median_ms, min_ms)`` across ``n_bench`` runs."""
    for _ in range(n_warm):
        out = fn()
        jax.block_until_ready(out)
    samples = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def _state_bytes(state) -> int:
    """Rough upper bound on a term state's memory footprint."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(state):
        if hasattr(leaf, "nbytes"):
            total += int(leaf.nbytes)
    return total


def main() -> int:
    _banner("tmol_jax: LJ/LK dense vs cell-list benchmark")
    print(f"Input PDB: {INPUT_PDB}")
    print(f"Cutoff:    {CUTOFF} A")
    print(f"Warmup:    {N_WARM}   Bench: {N_BENCH}")

    import tmol_jax
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.score.ljlk.ljlk_energy_term import LJLKEnergyTerm

    ps = tmol_jax.pose_stack_from_pdb(str(INPUT_PDB))
    P, N, _ = ps.coords.shape
    print(f"Pose stack: {P} pose(s), {N} atoms/pose")
    db = ParameterDatabase.get_default()

    # --- Dense path ----------------------------------------------------
    _banner("Dense path  (use_neighbor_list=False)")
    t_dense = LJLKEnergyTerm(param_db=db, use_neighbor_list=False)
    t0 = time.perf_counter()
    state_dense = t_dense.setup_for_pose_stack(ps)
    setup_dense_ms = (time.perf_counter() - t0) * 1e3
    print(f"setup_for_pose_stack: {setup_dense_ms:8.2f} ms")
    print(f"term state memory:    {_fmt_bytes(_state_bytes(state_dense))}")

    fn_dense = jax.jit(lambda c: t_dense(ps, state_dense, coords_override=c))
    med, mn = _time_call(lambda: fn_dense(ps.coords), N_WARM, N_BENCH)
    print(f"score call (median):  {med:8.2f} ms  (min {mn:6.2f} ms)")

    grad_dense = jax.jit(
        jax.grad(lambda c: sum(
            v.sum() for v in t_dense(ps, state_dense, coords_override=c).values()
        ))
    )
    med_g, mn_g = _time_call(lambda: grad_dense(ps.coords), N_WARM, N_BENCH)
    print(f"grad call  (median):  {med_g:8.2f} ms  (min {mn_g:6.2f} ms)")

    out_dense = t_dense(ps, state_dense)
    print("unweighted energies:", {str(k): float(v[0]) for k, v in out_dense.items()})

    del fn_dense, grad_dense
    gc.collect()

    # --- Cell-list path ------------------------------------------------
    _banner(f"Cell-list path  (use_neighbor_list=True, cutoff={CUTOFF})")
    t_nbr = LJLKEnergyTerm(
        param_db=db, use_neighbor_list=True, neighbor_list_cutoff=CUTOFF
    )
    t0 = time.perf_counter()
    state_nbr = t_nbr.setup_for_pose_stack(ps)
    setup_nbr_ms = (time.perf_counter() - t0) * 1e3
    max_nbr = int(state_nbr.neighbor_idx.shape[-1])
    valid_nbrs = int(jnp.sum(state_nbr.neighbor_mask))
    print(f"setup_for_pose_stack: {setup_nbr_ms:8.2f} ms")
    print(f"max neighbours/atom:  {max_nbr}")
    print(f"true neighbour pairs: {valid_nbrs:,}  "
          f"(dense half-list would have {N*(N-1)//2:,})")
    print(f"term state memory:    {_fmt_bytes(_state_bytes(state_nbr))}")

    fn_nbr = jax.jit(lambda c: t_nbr(ps, state_nbr, coords_override=c))
    med, mn = _time_call(lambda: fn_nbr(ps.coords), N_WARM, N_BENCH)
    print(f"score call (median):  {med:8.2f} ms  (min {mn:6.2f} ms)")

    grad_nbr = jax.jit(
        jax.grad(lambda c: sum(
            v.sum() for v in t_nbr(ps, state_nbr, coords_override=c).values()
        ))
    )
    med_g, mn_g = _time_call(lambda: grad_nbr(ps.coords), N_WARM, N_BENCH)
    print(f"grad call  (median):  {med_g:8.2f} ms  (min {mn_g:6.2f} ms)")

    out_nbr = t_nbr(ps, state_nbr)
    print("unweighted energies:", {str(k): float(v[0]) for k, v in out_nbr.items()})

    # --- Numerical parity ---------------------------------------------
    _banner("Numerical parity (dense vs cell-list)")
    for key in out_dense:
        d = out_dense[key]
        n = out_nbr[key]
        abs_diff = float(jnp.abs(d - n).max())
        denom = max(float(jnp.abs(d).max()), 1e-6)
        rel_diff = abs_diff / denom
        print(f"  {key!s:<22}  dense={float(d[0]):+12.4f}  "
              f"cell-list={float(n[0]):+12.4f}  "
              f"|d|={abs_diff:9.2e}  rel={rel_diff:9.2e}")

    # --- Intermediate tensor footprint --------------------------------
    _banner("Intermediate pair-tensor footprint (per score call)")
    dense_pair_bytes = P * N * N * 4  # one float32 [P, N, N] tensor
    nbr_pair_bytes = P * N * max_nbr * 4
    print(f"dense  [P, N, N]  float32 = {_fmt_bytes(dense_pair_bytes)}")
    print(f"nbrlst [P, N, k]  float32 = {_fmt_bytes(nbr_pair_bytes)} "
          f"({dense_pair_bytes / max(nbr_pair_bytes, 1):.1f}x smaller)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
