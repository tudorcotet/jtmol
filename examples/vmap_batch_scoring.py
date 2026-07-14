"""Benchmark: Python loop vs ``jax.vmap`` vs ``jit(vmap(...))`` for batch scoring.

The single biggest reason we ported tmol to JAX is that ``jax.vmap`` turns
the per-pose energy function into a batched one "for free" — one Python
call, one XLA compile, one fused GPU kernel for all N conformations. In
the upstream PyTorch + custom-CUDA tmol, batching N perturbed structures
means either looping in Python (serial, ~Nx slower) or hand-writing
explicit batched kernels.

This script makes the win measurable:

    1. Load 1ubq as a single-pose ``PoseStack`` (topology is shared).
    2. Generate ``N_BATCH`` perturbed coordinate tensors by adding
       Gaussian noise to the starting coords.
    3. Score the batch three ways and time each:
         a. Naive Python loop over poses (the "PyTorch-era" baseline).
         b. ``jax.vmap`` over the batch axis (eager).
         c. ``jax.jit(jax.vmap(score_fn))`` — the production form.
    4. Report speedup ratios.
    5. Also benchmark the analogous ``jax.grad`` path, because that is
       the shape of every ML training step: evaluate loss + gradient
       on a whole minibatch of structures in parallel.

On a CPU the speedup from vmap alone is modest (vectorised numpy-style
broadcasting) and most of the win comes from ``jit``. On a GPU the win
is dramatic — typically 50-200x vs the Python loop for this pose size,
because each loop iteration pays fixed launch overhead per kernel while
the vmapped graph fuses everything into a single dispatch.

Run it:

    python examples/vmap_batch_scoring.py

Environment overrides:

    TMOL_JAX_PDB     Path to an alternative PDB.
    TMOL_JAX_BATCH   Batch size (default: 100).
    TMOL_JAX_NOISE   Std of Gaussian coord noise in Angstroms (default: 0.1).
    TMOL_JAX_SEED    PRNG seed (default: 0).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------- #
# Make ``jtmol`` importable when this script is run directly as       #
# ``python examples/vmap_batch_scoring.py`` without the package being    #
# installed. We prepend the repo root (the parent of ``examples/``) to   #
# ``sys.path`` so the ``jtmol`` package next to it is resolvable from #
# any working directory.                                                 #
# ---------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------- #
# Bundled 1ubq fixture so the benchmark runs with no arguments.          #
# ---------------------------------------------------------------------- #
DEFAULT_PDB = (
    Path(__file__).resolve().parent.parent
    / "jtmol"
    / "tests"
    / "data"
    / "1ubq.pdb"
)

INPUT_PDB = Path(os.environ.get("TMOL_JAX_PDB", str(DEFAULT_PDB)))
N_BATCH = int(os.environ.get("TMOL_JAX_BATCH", "100"))
NOISE_STD = float(os.environ.get("TMOL_JAX_NOISE", "0.1"))
SEED = int(os.environ.get("TMOL_JAX_SEED", "0"))


def _banner(title: str) -> None:
    line = "-" * 64
    print(f"\n{line}\n{title}\n{line}")


def _time_call(fn, *args, repeats: int = 1) -> float:
    """Run ``fn(*args)`` ``repeats`` times, blocking on the device result.

    ``jax.block_until_ready`` is critical: JAX dispatches are asynchronous,
    so without it the ``perf_counter`` delta just measures how long it took
    to queue the work, not to actually run it.
    """
    # Warmup has already happened at call sites; here we just measure.
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
        # For grad calls ``out`` may be a pytree of arrays — we flatten to
        # a single scalar block via ``jax.block_until_ready`` which is pytree
        # aware in newer JAX.
        _block_until_ready(out)
    return (time.perf_counter() - t0) / repeats


def _block_until_ready(pytree) -> None:
    """Portable ``block_until_ready`` that handles arrays, tuples, and dicts.

    Older JAX only defines the method on arrays. We walk the pytree and call
    it on every leaf so timings are honest even when the function returns
    gradients, ``(loss, grad)`` pairs, etc.
    """
    import jax
    import jax.numpy as jnp

    for leaf in jax.tree_util.tree_leaves(pytree):
        # Only jax arrays have ``block_until_ready``; scalars don't.
        if isinstance(leaf, jnp.ndarray):
            leaf.block_until_ready()


def main() -> int:
    _banner("jtmol: vmap batch-scoring benchmark")
    print(f"Input PDB:     {INPUT_PDB}")
    print(f"Batch size:    {N_BATCH}")
    print(f"Noise std (Å): {NOISE_STD}")
    print(f"Seed:          {SEED}")

    if not INPUT_PDB.exists():
        print(f"ERROR: Input PDB not found at {INPUT_PDB}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ #
    # Imports.                                                           #
    # ------------------------------------------------------------------ #
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        print(f"ERROR: failed to import jax: {exc!r}", file=sys.stderr)
        return 1

    try:
        from jtmol import pose_stack_from_pdb, beta2016_score_function
    except Exception as exc:
        print(f"ERROR: failed to import jtmol: {exc!r}", file=sys.stderr)
        return 1

    backend = jax.default_backend()
    print(f"JAX backend:   {backend}  (devices: {jax.devices()})")

    # ------------------------------------------------------------------ #
    # Load the pose and build the score function.                        #
    #                                                                    #
    # ``setup_for_pose_stack`` precomputes per-term state (bonded graph, #
    # neighbour lists, parameter tables expanded to pose layout, ...).   #
    # We run it exactly once here and pass ``state`` into every scoring  #
    # call — otherwise the vmap / loop would redo this O(n_batch) times  #
    # and the comparison would be meaningless.                           #
    # ------------------------------------------------------------------ #
    _banner("Setup: load pose, build score function, precompute state")
    try:
        pose_stack = pose_stack_from_pdb(str(INPUT_PDB))
        sfxn = beta2016_score_function()
        state = sfxn.setup_for_pose_stack(pose_stack)
        base_coords = pose_stack.coords  # [n_poses=1, max_n_atoms, 3]
        print(f"  base coords shape: {base_coords.shape}")
    except Exception as exc:
        print(f"ERROR: pose/sfxn setup failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Build the batch of perturbed coordinate tensors.                   #
    #                                                                    #
    # Shape: [N_BATCH, 1, max_n_atoms, 3]. The leading axis is what we   #
    # vmap over; the inner ``[1, max_n_atoms, 3]`` matches the shape     #
    # that ``sfxn`` expects for a single pose stack of ``n_poses == 1``. #
    # ------------------------------------------------------------------ #
    _banner("Build perturbed coordinate batch")
    try:
        key = jax.random.PRNGKey(SEED)
        noise = jax.random.normal(key, (N_BATCH,) + base_coords.shape) * NOISE_STD
        # First element has zero noise so batch[0] matches the starting pose.
        noise = noise.at[0].set(0.0)
        coords_batch = base_coords[None, ...] + noise
        # Materialise so batch build time is not folded into any timer below.
        coords_batch.block_until_ready()
        print(f"  coords_batch shape: {coords_batch.shape}")
    except Exception as exc:
        print(f"ERROR: failed to build batch: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Single-pose scoring closure.                                       #
    #                                                                    #
    # This is the function we vmap / loop. Pose topology and the         #
    # precomputed ``state`` are captured in the closure; only ``coords`` #
    # changes across the batch (via ``coords_override``).                #
    # ------------------------------------------------------------------ #
    def energy_of_coords(coords: jnp.ndarray) -> jnp.ndarray:
        """Scalar-ish energy for a single coords tensor ``[1, n_atoms, 3]``.

        Returns ``float32[n_poses]`` (i.e. shape ``[1]``), matching the
        ``ScoreFunction.__call__`` contract. We vmap this over the leading
        batch axis; after the vmap the output shape is ``[N_BATCH, 1]``.
        """
        return sfxn(pose_stack, state=state, coords_override=coords)

    # ================================================================== #
    # Benchmark 1: naive Python loop (the "PyTorch-era" baseline).       #
    # ================================================================== #
    _banner("Bench A: naive Python loop over poses")
    try:
        def loop_score(batch: jnp.ndarray) -> jnp.ndarray:
            """Score each pose in a Python for-loop. One dispatch per pose.

            This is what you'd have to write in upstream tmol: same score
            function, called N times. Every iteration pays (a) Python
            overhead, (b) JAX tracing/dispatch overhead, (c) kernel launch
            overhead on GPU.
            """
            out = []
            for i in range(batch.shape[0]):
                out.append(energy_of_coords(batch[i]))
            return jnp.stack(out, axis=0)

        # Warmup (first call traces / compiles the per-iteration graph and
        # pays one-time JAX machinery cost — we don't want that in the timer).
        warmup_out = loop_score(coords_batch[:2])
        _block_until_ready(warmup_out)

        loop_time = _time_call(loop_score, coords_batch)
        loop_energies = loop_score(coords_batch)
        _block_until_ready(loop_energies)
        print(f"  loop time:     {loop_time*1000:8.2f} ms  (N={N_BATCH})")
        print(f"  per-pose:      {loop_time/N_BATCH*1000:8.2f} ms")
    except Exception as exc:
        print(f"ERROR: loop bench failed: {exc!r}", file=sys.stderr)
        return 1

    # ================================================================== #
    # Benchmark 2: jax.vmap (eager).                                      #
    # ================================================================== #
    _banner("Bench B: jax.vmap (eager, no jit)")
    try:
        vmap_score = jax.vmap(energy_of_coords)

        # Warmup.
        warmup_out = vmap_score(coords_batch[:2])
        _block_until_ready(warmup_out)

        vmap_time = _time_call(vmap_score, coords_batch)
        vmap_energies = vmap_score(coords_batch)
        _block_until_ready(vmap_energies)
        print(f"  vmap time:     {vmap_time*1000:8.2f} ms  (N={N_BATCH})")
        print(f"  per-pose:      {vmap_time/N_BATCH*1000:8.2f} ms")
    except Exception as exc:
        print(f"ERROR: vmap bench failed: {exc!r}", file=sys.stderr)
        return 1

    # ================================================================== #
    # Benchmark 3: jax.jit(jax.vmap(...)) — the production form.          #
    # ================================================================== #
    _banner("Bench C: jit(vmap(...))  (the production form)")
    try:
        jit_vmap_score = jax.jit(jax.vmap(energy_of_coords))

        # First call = trace + XLA compile. Exclude from the main timer.
        t0 = time.perf_counter()
        jit_warm = jit_vmap_score(coords_batch)
        _block_until_ready(jit_warm)
        compile_time = time.perf_counter() - t0
        print(f"  compile+first: {compile_time*1000:8.2f} ms (one-time)")

        # Now time the cached, compiled call. Repeat a few times to average
        # out scheduler noise on short runs.
        jit_time = _time_call(jit_vmap_score, coords_batch, repeats=5)
        jit_energies = jit_vmap_score(coords_batch)
        _block_until_ready(jit_energies)
        print(f"  jit(vmap):     {jit_time*1000:8.2f} ms  (N={N_BATCH}, avg of 5)")
        print(f"  per-pose:      {jit_time/N_BATCH*1000:8.2f} ms")
    except Exception as exc:
        print(f"ERROR: jit(vmap) bench failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Correctness check — all three paths should agree.                  #
    # ------------------------------------------------------------------ #
    try:
        # ``loop_energies`` is [N, 1]; vmap outputs match shape. Compare.
        max_diff_vmap = float(jnp.max(jnp.abs(vmap_energies - loop_energies)))
        max_diff_jit = float(jnp.max(jnp.abs(jit_energies - loop_energies)))
        print(f"\n  max |vmap - loop|:      {max_diff_vmap:.3e}")
        print(f"  max |jit(vmap) - loop|: {max_diff_jit:.3e}")
    except Exception as exc:
        print(f"  (correctness check skipped: {exc!r})", file=sys.stderr)

    # ================================================================== #
    # Speedup table.                                                      #
    # ================================================================== #
    _banner("Speedup summary")
    print(f"  backend:              {backend}")
    print(f"  batch size:           {N_BATCH}")
    print()
    print(f"  {'method':<24s} {'time (ms)':>12s} {'speedup':>10s}")
    print(f"  {'-'*24} {'-'*12} {'-'*10}")
    print(f"  {'Python loop':<24s} {loop_time*1000:>12.2f} {1.0:>10.2f}x")
    print(f"  {'jax.vmap (eager)':<24s} {vmap_time*1000:>12.2f} {loop_time/vmap_time:>10.2f}x")
    print(f"  {'jit(vmap)':<24s} {jit_time*1000:>12.2f} {loop_time/jit_time:>10.2f}x")
    print()
    if backend == "cpu":
        print("  NOTE: running on CPU. Expect a modest speedup (a few x to ~10x).")
        print("        On GPU the same benchmark typically shows 50-200x for")
        print("        jit(vmap) vs the Python loop, because the per-iteration")
        print("        kernel-launch overhead of the loop dominates and vmap")
        print("        fuses the whole batch into one dispatch.")
    else:
        print(f"  NOTE: running on {backend}. A well-warmed jit(vmap) at this")
        print("        batch size typically lands in the 50-200x speedup range.")

    # ================================================================== #
    # Benchmark 4: jax.grad under vmap — the ML training loop shape.     #
    # ================================================================== #
    _banner("Bench D: jax.grad under vmap  (ML training loop shape)")
    try:
        def scalar_energy(coords: jnp.ndarray) -> jnp.ndarray:
            """``jax.grad`` needs a scalar; sum over the (singleton) n_poses axis."""
            return sfxn(pose_stack, state=state, coords_override=coords).sum()

        # Per-pose gradient: dE/d(coords) for one pose. Shape [1, n_atoms, 3].
        grad_fn = jax.grad(scalar_energy)

        # ---- naive loop of grads ----
        def loop_grad(batch: jnp.ndarray) -> jnp.ndarray:
            """One gradient per pose, Python loop. This is the baseline
            you'd write in upstream tmol / PyTorch without batched autograd."""
            out = []
            for i in range(batch.shape[0]):
                out.append(grad_fn(batch[i]))
            return jnp.stack(out, axis=0)

        # Warmup.
        _block_until_ready(loop_grad(coords_batch[:2]))

        loop_grad_time = _time_call(loop_grad, coords_batch)

        # ---- vmap(grad) — one trace, batched differentiation ----
        # Note: jax.vmap(jax.grad(...)) is the canonical form for per-sample
        # gradients. Each batch element's gradient is independent, so this
        # is trivially parallelisable; vmap just tells XLA so.
        vmap_grad_fn = jax.vmap(grad_fn)
        _block_until_ready(vmap_grad_fn(coords_batch[:2]))  # warmup
        vmap_grad_time = _time_call(vmap_grad_fn, coords_batch)

        # ---- jit(vmap(grad)) — the production form ----
        jit_vmap_grad_fn = jax.jit(jax.vmap(grad_fn))
        t0 = time.perf_counter()
        jit_grad_warm = jit_vmap_grad_fn(coords_batch)
        _block_until_ready(jit_grad_warm)
        grad_compile_time = time.perf_counter() - t0
        jit_vmap_grad_time = _time_call(jit_vmap_grad_fn, coords_batch, repeats=5)

        print(f"  loop grad:            {loop_grad_time*1000:8.2f} ms")
        print(f"  vmap(grad) eager:     {vmap_grad_time*1000:8.2f} ms  "
              f"({loop_grad_time/vmap_grad_time:.2f}x)")
        print(f"  jit(vmap(grad)):      {jit_vmap_grad_time*1000:8.2f} ms  "
              f"({loop_grad_time/jit_vmap_grad_time:.2f}x)  "
              f"[compile: {grad_compile_time*1000:.1f} ms]")
        print()
        print("  This is the shape of the inner step of an ML training loop:")
        print("    for each minibatch of N structures, compute loss + gradient")
        print("    for every structure in one fused kernel. No Python loop,")
        print("    no per-sample dispatch overhead.")
    except Exception as exc:
        print(f"ERROR: grad bench failed: {exc!r}", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
