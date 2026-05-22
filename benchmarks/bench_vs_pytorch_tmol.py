"""Benchmark ``tmol_jax`` against the upstream PyTorch ``tmol`` on 1ubq.

Goal
----

Give an apples-to-apples-ish comparison between this JAX port and the
upstream ``tmol`` package (PyTorch + custom C++/CUDA kernels) on a single
small protein (1ubq, 76 residues). The script:

    1. Attempts to ``import tmol`` from the upstream source tree
       vendored at ``./tmol/`` (sibling of ``./tmol_jax/``). This is
       expected to fail on a CPU-only machine because the upstream
       package requires pre-built C++/CUDA extensions that ship in
       ``linux_x86_64`` wheels. We capture and print the exact failure
       mode so the benchmark document can quote it.

    2. Runs a sequence of timing micro-benchmarks against ``tmol_jax``:

         * first call to the energy function (trace + XLA compile),
         * amortised steady-state scoring (100 calls, JIT-cached),
         * gradient evaluation (100 calls, ``jax.jit(jax.grad(...))``),
         * ``vmap`` over 100 perturbed poses (``jit(vmap(score))``),
         * ``vmap(grad(...))`` over 100 poses (ML-training shape),
         * 50 L-BFGS steps of ``run_cart_min``,
         * dense vs neighbor-list LJ/LK path (LJLK term in isolation),
         * full setup-cost breakdown (parse PDB, build sfxn, setup state).

    3. Prints a small JSON block at the end so a wrapper script can
       scrape the numbers for the generated ``BENCHMARKS.md``.

    Upstream tmol is expected to be inaccessible on CPU-only macOS (it
    links custom CUDA/C++ kernels distributed in ``linux_x86_64`` wheels).
    Where we cannot run upstream ourselves we quote the numbers from the
    upstream tmol manuscript / repo docs and flag them as "projected" in
    the generated benchmark markdown.

Running
-------

From the repo root::

    PYTHONPATH=. /opt/homebrew/bin/python3.13 benchmarks/bench_vs_pytorch_tmol.py

Environment overrides
---------------------

    TMOL_JAX_PDB      Path to an alternative PDB (default: 1ubq fixture).
    BENCH_BATCH       Batch size for vmap test (default: 100).
    BENCH_REPEATS     Steady-state repeats per measurement (default: 100).
    BENCH_MIN_STEPS   L-BFGS iterations for run_cart_min (default: 50).
    BENCH_NOISE       Std of Gaussian coord noise for vmap batch (default: 0.1 Å).
    BENCH_SEED        PRNG seed (default: 0).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------- #
# Paths. The script must be runnable from the repo root. We resolve     #
# everything relative to this file so ``cd`` accidents don't break it.  #
# --------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
_TMOL_JAX = _REPO_ROOT / "tmol_jax"
_TMOL_UPSTREAM = _REPO_ROOT / "tmol"
_DEFAULT_PDB = _TMOL_JAX / "tests" / "data" / "1ubq.pdb"

# Ensure we import the vendored tmol_jax package (not a pip-installed one).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


PDB_PATH = Path(os.environ.get("TMOL_JAX_PDB", str(_DEFAULT_PDB)))
BATCH = int(os.environ.get("BENCH_BATCH", "100"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "100"))
MIN_STEPS = int(os.environ.get("BENCH_MIN_STEPS", "50"))
NOISE_STD = float(os.environ.get("BENCH_NOISE", "0.1"))
SEED = int(os.environ.get("BENCH_SEED", "0"))


# --------------------------------------------------------------------- #
# Tiny timing helpers.                                                  #
# --------------------------------------------------------------------- #
def _banner(title: str) -> None:
    line = "=" * 68
    print(f"\n{line}\n{title}\n{line}")


def _block_until_ready(pytree: Any) -> None:
    """Walk a pytree and block on every array leaf.

    JAX dispatches are async, so without this ``perf_counter`` would
    only measure the time to queue the work, not to finish it.
    """
    import jax

    for leaf in jax.tree_util.tree_leaves(pytree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _time(fn: Callable[..., Any], *args, repeats: int = 1, warmup: int = 1) -> float:
    """Return mean wall-time per ``fn(*args)`` call over ``repeats``.

    We run ``warmup`` unmeasured calls first so trace / compile / paging
    work doesn't pollute the first measured iteration.
    """
    for _ in range(warmup):
        _block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
        _block_until_ready(out)
    return (time.perf_counter() - t0) / repeats


# --------------------------------------------------------------------- #
# Host/environment probe — gets quoted in the generated markdown table. #
# --------------------------------------------------------------------- #
def _collect_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_path": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }

    # CPU name / core count via sysctl on Darwin, /proc/cpuinfo on Linux.
    try:
        if sys.platform == "darwin":
            info["cpu_name"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["cpu_physical"] = int(
                subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True)
            )
            info["cpu_logical"] = int(
                subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True)
            )
            info["mem_bytes"] = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            )
        else:
            # Linux fallback. /proc/cpuinfo is noisy; grab one "model name".
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        info["cpu_name"] = line.split(":", 1)[1].strip()
                        break
            info["cpu_logical"] = os.cpu_count()
    except Exception as exc:
        info["cpu_probe_error"] = repr(exc)

    # JAX / jaxlib.
    try:
        import jax
        import jaxlib

        info["jax"] = jax.__version__
        info["jaxlib"] = jaxlib.__version__
        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:
        info["jax_probe_error"] = repr(exc)

    # tmol_jax version.
    try:
        import tmol_jax

        info["tmol_jax"] = getattr(tmol_jax, "__version__", "unknown")
    except Exception as exc:
        info["tmol_jax_probe_error"] = repr(exc)

    return info


# --------------------------------------------------------------------- #
# Probe: upstream tmol. Expected to fail on CPU-only macOS.             #
# --------------------------------------------------------------------- #
def _probe_upstream_tmol() -> Dict[str, Any]:
    """Try ``import tmol`` from the vendored upstream source.

    We prepend the upstream path to ``sys.path`` *temporarily* so we do
    not accidentally shadow ``tmol_jax`` for the rest of the benchmark.
    Either way, we remove it when we're done.
    """
    out: Dict[str, Any] = {"path_prepended": str(_TMOL_UPSTREAM)}
    prepended = False
    if _TMOL_UPSTREAM.exists() and str(_TMOL_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_TMOL_UPSTREAM))
        prepended = True
    try:
        # torch must be importable at all; its absence is a different class
        # of failure than "tmol C extensions missing".
        try:
            import torch  # noqa: F401

            out["torch_available"] = True
        except Exception as exc:
            out["torch_available"] = False
            out["torch_error"] = f"{type(exc).__name__}: {exc}"

        try:
            import tmol as torch_tmol  # type: ignore[import-not-found]

            # If we got here we're a very lucky Linux x86_64 user. Record
            # enough info to verify that the real thing loaded.
            out["import_ok"] = True
            out["version"] = getattr(torch_tmol, "__version__", "unknown")
        except Exception as exc:
            out["import_ok"] = False
            out["error_type"] = type(exc).__name__
            out["error_message"] = str(exc)
    finally:
        if prepended:
            # Remove the upstream path to leave sys.path clean.
            try:
                sys.path.remove(str(_TMOL_UPSTREAM))
            except ValueError:
                pass
        # Drop cached partial imports so they don't leak into later use.
        for name in list(sys.modules):
            if name == "tmol" or name.startswith("tmol."):
                sys.modules.pop(name, None)
    return out


# --------------------------------------------------------------------- #
# Benchmarks (tmol_jax).                                                #
# --------------------------------------------------------------------- #
def _run_jax_benchmarks() -> Dict[str, Any]:
    """Drive the JAX-side timing measurements. Returns a flat dict."""
    import jax
    import jax.numpy as jnp

    from tmol_jax import (
        beta2016_score_function,
        pose_stack_from_pdb,
        run_cart_min,
    )

    results: Dict[str, Any] = {}

    # ----- load + precompute ----- #
    t0 = time.perf_counter()
    pose_stack = pose_stack_from_pdb(str(PDB_PATH))
    results["load_pdb_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sfxn = beta2016_score_function()
    results["build_sfxn_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = sfxn.setup_for_pose_stack(pose_stack)
    _block_until_ready(state)
    results["setup_for_pose_stack_sec"] = time.perf_counter() - t0

    base_coords = pose_stack.coords  # [n_poses=1, n_atoms, 3]
    results["n_atoms"] = int(base_coords.shape[1])
    results["n_poses"] = int(base_coords.shape[0])

    # Total one-time setup cost (everything a user pays before their
    # first score). Useful to quote against upstream's "warmup time".
    results["total_setup_sec"] = (
        results["load_pdb_sec"]
        + results["build_sfxn_sec"]
        + results["setup_for_pose_stack_sec"]
    )

    _banner("tmol_jax: setup breakdown")
    print(f"  parse PDB:             {results['load_pdb_sec']*1000:8.2f} ms")
    print(f"  build score function:  {results['build_sfxn_sec']*1000:8.2f} ms")
    print(
        f"  setup_for_pose_stack:  "
        f"{results['setup_for_pose_stack_sec']*1000:8.2f} ms"
    )
    print(f"  total setup:           {results['total_setup_sec']*1000:8.2f} ms")
    print(f"  n_atoms:               {results['n_atoms']}")

    # ----- single-pose scoring closure ----- #
    def energy_of_coords(coords: jnp.ndarray) -> jnp.ndarray:
        """float32[n_poses] energy for the given coord tensor."""
        return sfxn(pose_stack, state=state, coords_override=coords)

    def scalar_energy(coords: jnp.ndarray) -> jnp.ndarray:
        """Scalar energy (sum over the singleton n_poses axis) for grad."""
        return sfxn(pose_stack, state=state, coords_override=coords).sum()

    # =====================================================================
    # 1) First call (compile cost) vs steady state.
    # =====================================================================
    _banner("tmol_jax: energy — first call vs amortised steady state")

    # Eager first call — captures python-side overhead, tracing, dispatch.
    t0 = time.perf_counter()
    e_eager_first = energy_of_coords(base_coords)
    _block_until_ready(e_eager_first)
    results["eager_first_call_sec"] = time.perf_counter() - t0
    results["energy_initial"] = float(e_eager_first.reshape(-1)[0])

    jit_score = jax.jit(energy_of_coords)

    # jit first call — trace + XLA compile. Expected to dominate.
    t0 = time.perf_counter()
    e_jit_first = jit_score(base_coords)
    _block_until_ready(e_jit_first)
    results["jit_first_call_sec"] = time.perf_counter() - t0

    # Amortised: REPEATS cached calls.
    results["jit_amortised_sec"] = _time(
        jit_score, base_coords, repeats=REPEATS, warmup=2
    )

    print(f"  eager first call:     {results['eager_first_call_sec']*1000:8.2f} ms")
    print(f"  jit  first call:      {results['jit_first_call_sec']*1000:8.2f} ms")
    print(
        f"  jit  amortised ({REPEATS}):"
        f"  {results['jit_amortised_sec']*1000:8.2f} ms"
    )
    print(f"  energy @ input:       {results['energy_initial']:.4f}")

    # =====================================================================
    # 2) Gradient — jit(grad) amortised cost.
    # =====================================================================
    _banner("tmol_jax: gradient — jit(grad(energy)) amortised")
    jit_grad = jax.jit(jax.grad(scalar_energy))
    # First call compiles; measure separately.
    t0 = time.perf_counter()
    g0 = jit_grad(base_coords)
    _block_until_ready(g0)
    results["grad_first_call_sec"] = time.perf_counter() - t0

    results["grad_amortised_sec"] = _time(
        jit_grad, base_coords, repeats=REPEATS, warmup=2
    )

    print(f"  grad first call:      {results['grad_first_call_sec']*1000:8.2f} ms")
    print(
        f"  grad amortised ({REPEATS}): "
        f"{results['grad_amortised_sec']*1000:8.2f} ms"
    )

    # =====================================================================
    # 3) vmap over BATCH perturbed poses vs Python loop baseline.
    # =====================================================================
    _banner(f"tmol_jax: vmap over {BATCH} perturbed poses vs Python loop")

    key = jax.random.PRNGKey(SEED)
    noise = jax.random.normal(key, (BATCH,) + base_coords.shape) * NOISE_STD
    noise = noise.at[0].set(0.0)  # first element = unperturbed
    coords_batch = base_coords[None, ...] + noise
    _block_until_ready(coords_batch)

    # ----- Python loop (the "PyTorch era" baseline) using the jit-cached
    #       single-pose function. One dispatch per element.
    def loop_score(batch: jnp.ndarray) -> jnp.ndarray:
        out: List[jnp.ndarray] = []
        for i in range(batch.shape[0]):
            out.append(jit_score(batch[i]))
        return jnp.stack(out, axis=0)

    # Prime the jit (jit_score already compiled).
    _block_until_ready(loop_score(coords_batch[:2]))
    t0 = time.perf_counter()
    loop_energies = loop_score(coords_batch)
    _block_until_ready(loop_energies)
    results["loop_score_total_sec"] = time.perf_counter() - t0

    # ----- jit(vmap(...)) — the production form.
    jit_vmap = jax.jit(jax.vmap(energy_of_coords))
    t0 = time.perf_counter()
    e_vmap_first = jit_vmap(coords_batch)
    _block_until_ready(e_vmap_first)
    results["vmap_compile_sec"] = time.perf_counter() - t0
    results["vmap_total_sec"] = _time(
        jit_vmap, coords_batch, repeats=5, warmup=1
    )

    results["vmap_speedup_vs_loop"] = (
        results["loop_score_total_sec"] / results["vmap_total_sec"]
        if results["vmap_total_sec"] > 0
        else float("inf")
    )
    # Sanity check that loop and vmap agree.
    max_diff = float(jnp.max(jnp.abs(e_vmap_first - loop_energies)))
    results["vmap_vs_loop_max_abs_diff"] = max_diff

    print(
        f"  loop  over {BATCH}:        {results['loop_score_total_sec']*1000:8.2f} ms"
    )
    print(
        f"  jit(vmap) compile:       {results['vmap_compile_sec']*1000:8.2f} ms"
    )
    print(
        f"  jit(vmap) steady state:  {results['vmap_total_sec']*1000:8.2f} ms"
    )
    print(f"  speedup:                 {results['vmap_speedup_vs_loop']:.2f}x")
    print(f"  max |vmap - loop|:       {max_diff:.3e}")

    # =====================================================================
    # 3b) vmap(grad(...)) — the shape ML training actually needs.
    # =====================================================================
    _banner(
        f"tmol_jax: jit(vmap(grad(energy))) over {BATCH} perturbed poses"
    )

    # Per-pose scalar energy so jax.grad is well-defined elementwise after
    # vmap. ``energy_of_coords`` returns [n_poses=1]; ``.sum()`` reduces to
    # a scalar so vmap-of-grad gives [batch, n_atoms, 3].
    def scalar_energy_per_pose(coords: jnp.ndarray) -> jnp.ndarray:
        return sfxn(pose_stack, state=state, coords_override=coords).sum()

    jit_vmap_grad = jax.jit(jax.vmap(jax.grad(scalar_energy_per_pose)))
    t0 = time.perf_counter()
    g_vmap_first = jit_vmap_grad(coords_batch)
    _block_until_ready(g_vmap_first)
    results["vmap_grad_compile_sec"] = time.perf_counter() - t0
    results["vmap_grad_total_sec"] = _time(
        jit_vmap_grad, coords_batch, repeats=5, warmup=1
    )
    results["vmap_grad_per_pose_sec"] = (
        results["vmap_grad_total_sec"] / max(BATCH, 1)
    )

    # Python-loop baseline for the gradient (using the jit-cached single-
    # pose grad ``jit_grad``).
    def loop_grad(batch: jnp.ndarray) -> jnp.ndarray:
        out: List[jnp.ndarray] = []
        for i in range(batch.shape[0]):
            out.append(jit_grad(batch[i]))
        return jnp.stack(out, axis=0)

    _block_until_ready(loop_grad(coords_batch[:2]))
    t0 = time.perf_counter()
    loop_grads = loop_grad(coords_batch)
    _block_until_ready(loop_grads)
    results["loop_grad_total_sec"] = time.perf_counter() - t0

    results["vmap_grad_speedup_vs_loop"] = (
        results["loop_grad_total_sec"] / results["vmap_grad_total_sec"]
        if results["vmap_grad_total_sec"] > 0
        else float("inf")
    )

    print(
        f"  loop grad over {BATCH}:   "
        f"{results['loop_grad_total_sec']*1000:8.2f} ms"
    )
    print(
        f"  vmap(grad) compile:      "
        f"{results['vmap_grad_compile_sec']*1000:8.2f} ms"
    )
    print(
        f"  vmap(grad) steady state: "
        f"{results['vmap_grad_total_sec']*1000:8.2f} ms "
        f"({results['vmap_grad_per_pose_sec']*1000:.2f} ms/pose)"
    )
    print(f"  speedup:                 {results['vmap_grad_speedup_vs_loop']:.2f}x")

    # =====================================================================
    # 3c) Neighbor-list vs dense LJ/LK (term-level isolation).
    # =====================================================================
    _banner("tmol_jax: LJ/LK dense vs neighbor-list (cutoff=6 A)")
    try:
        from tmol_jax.database import ParameterDatabase
        from tmol_jax.score.ljlk.ljlk_energy_term import LJLKEnergyTerm

        db = ParameterDatabase.get_default()

        # ----- dense path -----
        dense_term = LJLKEnergyTerm(param_db=db, use_neighbor_list=False)
        t0 = time.perf_counter()
        dense_state = dense_term.setup_for_pose_stack(pose_stack)
        _block_until_ready(dense_state)
        results["ljlk_dense_setup_sec"] = time.perf_counter() - t0

        dense_score = jax.jit(
            lambda c: dense_term(pose_stack, dense_state, coords_override=c)
        )
        _block_until_ready(dense_score(base_coords))  # compile
        results["ljlk_dense_score_sec"] = _time(
            dense_score, base_coords, repeats=REPEATS, warmup=2
        )

        dense_grad = jax.jit(
            jax.grad(
                lambda c: sum(
                    v.sum()
                    for v in dense_term(
                        pose_stack, dense_state, coords_override=c
                    ).values()
                )
            )
        )
        _block_until_ready(dense_grad(base_coords))
        results["ljlk_dense_grad_sec"] = _time(
            dense_grad, base_coords, repeats=REPEATS, warmup=2
        )

        # Memory footprint of the pair tensor (dense is [P, N, N] fp32).
        P, N, _3 = base_coords.shape
        results["ljlk_dense_pair_bytes"] = int(P * N * N * 4)

        # ----- neighbor-list path -----
        nbr_term = LJLKEnergyTerm(
            param_db=db, use_neighbor_list=True, neighbor_list_cutoff=6.0
        )
        t0 = time.perf_counter()
        nbr_state = nbr_term.setup_for_pose_stack(pose_stack)
        _block_until_ready(nbr_state)
        results["ljlk_nbr_setup_sec"] = time.perf_counter() - t0

        # Size of the [P, N, k] neighbor index — this is the actual
        # pair-tensor footprint for the NL path.
        max_nbrs = int(nbr_state.neighbor_idx.shape[-1])
        results["ljlk_nbr_max_neighbors"] = max_nbrs
        results["ljlk_nbr_pair_bytes"] = int(P * N * max_nbrs * 4)

        nbr_score = jax.jit(
            lambda c: nbr_term(pose_stack, nbr_state, coords_override=c)
        )
        _block_until_ready(nbr_score(base_coords))
        results["ljlk_nbr_score_sec"] = _time(
            nbr_score, base_coords, repeats=REPEATS, warmup=2
        )

        nbr_grad = jax.jit(
            jax.grad(
                lambda c: sum(
                    v.sum()
                    for v in nbr_term(
                        pose_stack, nbr_state, coords_override=c
                    ).values()
                )
            )
        )
        _block_until_ready(nbr_grad(base_coords))
        results["ljlk_nbr_grad_sec"] = _time(
            nbr_grad, base_coords, repeats=REPEATS, warmup=2
        )

        results["ljlk_score_speedup_nl_vs_dense"] = (
            results["ljlk_dense_score_sec"] / results["ljlk_nbr_score_sec"]
            if results["ljlk_nbr_score_sec"] > 0
            else float("inf")
        )
        results["ljlk_grad_speedup_nl_vs_dense"] = (
            results["ljlk_dense_grad_sec"] / results["ljlk_nbr_grad_sec"]
            if results["ljlk_nbr_grad_sec"] > 0
            else float("inf")
        )
        results["ljlk_memory_ratio_dense_over_nl"] = (
            results["ljlk_dense_pair_bytes"]
            / max(results["ljlk_nbr_pair_bytes"], 1)
        )

        print(f"  dense setup:       {results['ljlk_dense_setup_sec']*1000:8.2f} ms")
        print(f"  dense score:       {results['ljlk_dense_score_sec']*1000:8.2f} ms")
        print(f"  dense grad:        {results['ljlk_dense_grad_sec']*1000:8.2f} ms")
        print(f"  nbrlst setup:      {results['ljlk_nbr_setup_sec']*1000:8.2f} ms")
        print(
            f"  nbrlst score:      {results['ljlk_nbr_score_sec']*1000:8.2f} ms"
        )
        print(f"  nbrlst grad:       {results['ljlk_nbr_grad_sec']*1000:8.2f} ms")
        print(
            f"  speedup (score):   "
            f"{results['ljlk_score_speedup_nl_vs_dense']:.2f}x"
        )
        print(
            f"  speedup (grad):    "
            f"{results['ljlk_grad_speedup_nl_vs_dense']:.2f}x"
        )
        print(
            f"  memory ratio:      "
            f"{results['ljlk_memory_ratio_dense_over_nl']:.1f}x smaller "
            f"(NL max k={max_nbrs})"
        )
    except Exception as exc:  # pragma: no cover
        print(f"  (NL-vs-dense benchmark failed: {exc!r})")
        results["ljlk_nl_vs_dense_error"] = f"{type(exc).__name__}: {exc}"

    # =====================================================================
    # 4) L-BFGS: run_cart_min for MIN_STEPS iterations.
    # =====================================================================
    _banner(f"tmol_jax: run_cart_min for {MIN_STEPS} L-BFGS steps")

    # Warmup: a tiny 2-step run so any internal jit caches are populated
    # before we time the real thing.
    try:
        _ = run_cart_min(pose_stack, sfxn, max_iter=2)
    except Exception as exc:
        # Fall through — benchmark still works, we just don't get a warmup.
        print(f"  (warmup run failed: {exc!r})")

    t0 = time.perf_counter()
    minimized = run_cart_min(pose_stack, sfxn, max_iter=MIN_STEPS)
    _block_until_ready(minimized.coords)
    results["run_cart_min_total_sec"] = time.perf_counter() - t0
    results["run_cart_min_per_step_sec"] = (
        results["run_cart_min_total_sec"] / max(MIN_STEPS, 1)
    )
    e_final = sfxn(minimized)
    _block_until_ready(e_final)
    results["energy_after_min"] = float(e_final.reshape(-1)[0])

    print(
        f"  {MIN_STEPS} steps total:      "
        f"{results['run_cart_min_total_sec']*1000:8.2f} ms"
    )
    print(
        f"  mean per step:           "
        f"{results['run_cart_min_per_step_sec']*1000:8.2f} ms"
    )
    print(f"  energy before:           {results['energy_initial']:.4f}")
    print(f"  energy after:            {results['energy_after_min']:.4f}")

    return results


# --------------------------------------------------------------------- #
# Main driver.                                                          #
# --------------------------------------------------------------------- #
def main() -> int:
    _banner("tmol_jax benchmark vs upstream PyTorch tmol")
    print(f"PDB:     {PDB_PATH}")
    print(f"Batch:   {BATCH}")
    print(f"Repeats: {REPEATS}")
    print(f"MinStep: {MIN_STEPS}")

    if not PDB_PATH.exists():
        print(f"ERROR: PDB not found at {PDB_PATH}", file=sys.stderr)
        return 2

    env = _collect_env()
    _banner("Environment")
    for k, v in env.items():
        print(f"  {k:<22s} {v}")

    _banner("Upstream tmol import probe")
    upstream = _probe_upstream_tmol()
    for k, v in upstream.items():
        # Truncate long error messages to one line in the console.
        if isinstance(v, str) and len(v) > 200:
            v_print = v.splitlines()[0] + " ..."
        else:
            v_print = v
        print(f"  {k:<22s} {v_print}")

    try:
        jax_results = _run_jax_benchmarks()
    except Exception as exc:
        print(f"\nERROR: benchmarks failed: {exc!r}", file=sys.stderr)
        raise

    summary = {
        "env": env,
        "upstream_probe": upstream,
        "jax": jax_results,
        "params": {
            "pdb": str(PDB_PATH),
            "batch": BATCH,
            "repeats": REPEATS,
            "min_steps": MIN_STEPS,
            "noise_std": NOISE_STD,
            "seed": SEED,
        },
    }

    _banner("Machine-readable summary (JSON)")
    print("BENCH_JSON_BEGIN")
    print(json.dumps(summary, indent=2, default=str))
    print("BENCH_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
