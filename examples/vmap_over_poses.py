"""Example: vmap a jtmol score function over a batch of poses.

The main motivation for porting tmol to JAX is that it makes batching
trivial. In the PyTorch + custom-CUDA upstream, scoring N perturbed
poses meant either looping in Python (slow) or writing explicit batched
CUDA kernels (painful). In JAX we write a single-pose-looking function
and ``vmap`` does the right thing.

This script demonstrates the two idiomatic forms:

    * ``jax.vmap`` over a leading batch axis of coordinates (static
      topology: same residues, same connectivity, just different xyz).
    * ``jax.jit(jax.vmap(...))`` for the common case where you want the
      batched evaluation traced and cached across calls.

We generate a small batch of perturbed ubiquitin structures
(N_BATCH=8 by default) by adding Gaussian noise to the starting
coordinates, then evaluate the total energy for every pose in one go.

Like the other examples, each major call is wrapped in try/except so
that a not-yet-ported submodule produces an informative error, not a
silent no-op.

Run it:

    python examples/vmap_over_poses.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


DEFAULT_PDB = (
    Path(__file__).resolve().parent.parent
    / "jtmol"
    / "tests"
    / "data"
    / "1ubq.pdb"
)

INPUT_PDB = Path(os.environ.get("TMOL_JAX_PDB", str(DEFAULT_PDB)))
N_BATCH = int(os.environ.get("TMOL_JAX_BATCH", "8"))
NOISE_STD = float(os.environ.get("TMOL_JAX_NOISE", "0.1"))  # Angstroms
SEED = int(os.environ.get("TMOL_JAX_SEED", "0"))


def _banner(title: str) -> None:
    line = "-" * 60
    print(f"\n{line}\n{title}\n{line}")


def main() -> int:
    _banner("jtmol: vmap over a batch of perturbed poses")
    print(f"Input PDB:     {INPUT_PDB}")
    print(f"Batch size:    {N_BATCH}")
    print(f"Noise std (Å): {NOISE_STD}")
    print(f"Seed:          {SEED}")

    if not INPUT_PDB.exists():
        print(f"ERROR: Input PDB not found at {INPUT_PDB}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ #
    # Imports. We need jax itself (for vmap/jit/grad/random) alongside   #
    # the jtmol surface. If jax is missing we bail immediately.       #
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

    # ------------------------------------------------------------------ #
    # Load the seed pose and build the score function.                   #
    #                                                                    #
    # The PoseStack returned by ``pose_stack_from_pdb`` has              #
    # ``n_poses == 1``. We keep it as-is and vary the coordinates across #
    # the batch axis via ``coords_override`` — the other pytree fields   #
    # (block types, connectivity, packed block types) stay shared.       #
    # ------------------------------------------------------------------ #
    _banner("Load pose + score function")
    try:
        pose_stack = pose_stack_from_pdb(str(INPUT_PDB))
        sfxn = beta2016_score_function()
        # Pre-run setup once so the vmap doesn't redo it per batch element.
        state = sfxn.setup_for_pose_stack(pose_stack)
        base_coords = pose_stack.coords  # [1, max_n_atoms, 3]
        print(f"  base coords shape: {base_coords.shape}")
    except Exception as exc:
        print(f"ERROR: failed to prep pose + sfxn: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Generate a batch of perturbed coordinate tensors.                  #
    #                                                                    #
    # Shape: [N_BATCH, 1, max_n_atoms, 3] — a batch axis in front of the #
    # existing (n_poses=1, n_atoms, 3) pose_stack coordinate layout.     #
    # The leading axis is the one we will ``vmap`` over.                 #
    # ------------------------------------------------------------------ #
    _banner("Generate perturbed coordinate batch")
    try:
        key = jax.random.PRNGKey(SEED)
        noise = jax.random.normal(key, (N_BATCH,) + base_coords.shape) * NOISE_STD
        # Broadcast-add the noise; first element is "zero noise" so we can
        # sanity-check that batch[0] equals the unperturbed energy.
        noise = noise.at[0].set(0.0)
        coords_batch = base_coords[None, ...] + noise
        print(f"  coords_batch shape: {coords_batch.shape}")
    except Exception as exc:
        print(f"ERROR: failed to build batch: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Single-pose energy closure.                                        #
    #                                                                    #
    # This is the function we will vmap. Note that:                      #
    #   * ``pose_stack`` and ``state`` are closed over — they do not     #
    #     vary across the batch.                                         #
    #   * Only ``coords`` (the argument) is batched via ``coords_override``. #
    #   * Return value is ``float32[n_poses]`` (i.e. [1] here), matching  #
    #     the EnergyTerm contract. After vmap it becomes [N_BATCH, 1].   #
    # ------------------------------------------------------------------ #
    def energy_of_coords(coords: jnp.ndarray) -> jnp.ndarray:
        """Total energy for a single coordinate tensor of shape [1, n_atoms, 3]."""
        return sfxn(pose_stack, state=state, coords_override=coords)

    # ------------------------------------------------------------------ #
    # Eager vmap — useful for debugging, runs each call without jit.     #
    # ------------------------------------------------------------------ #
    _banner("vmap (eager) over the batch")
    try:
        batched_energy_eager = jax.vmap(energy_of_coords)
        t0 = time.perf_counter()
        energies = batched_energy_eager(coords_batch)
        # Force materialization so the timing is honest.
        energies.block_until_ready()
        print(f"  vmap eager: {time.perf_counter() - t0:.2f}s")
        print(f"  energies shape: {energies.shape}  # [N_BATCH, n_poses]")
        print(f"  energies (per batch element):")
        for i, e in enumerate(energies.reshape(-1)):
            print(f"    [{i}]  {float(e):12.4f}")
    except Exception as exc:
        print(f"ERROR: vmap (eager) failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Jit + vmap — the production form. One trace, cached compile,       #
    # runs at XLA speed. Inner vmap then outer jit is the idiomatic      #
    # nesting order; jit compiles the entire batched graph in one shot.  #
    # ------------------------------------------------------------------ #
    _banner("jit(vmap(...)) over the batch")
    try:
        batched_energy_jit = jax.jit(jax.vmap(energy_of_coords))

        # First call = trace + compile.
        t0 = time.perf_counter()
        e_first = batched_energy_jit(coords_batch)
        e_first.block_until_ready()
        print(f"  jit(vmap) first call (compile): {time.perf_counter() - t0:.3f}s")

        # Second call = cached, just runs.
        t0 = time.perf_counter()
        e_second = batched_energy_jit(coords_batch)
        e_second.block_until_ready()
        print(f"  jit(vmap) second call (cached): {time.perf_counter() - t0:.3f}s")

        # The two runs should match the eager result bitwise (deterministic).
        max_diff = float(jnp.max(jnp.abs(e_second - energies)))
        print(f"  max |jit - eager| diff: {max_diff:.3e}")
    except Exception as exc:
        print(f"ERROR: jit(vmap) path failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Takeaway.                                                          #
    #                                                                    #
    # Scaling up is now trivial: point ``coords_batch`` at thousands of  #
    # perturbed structures (or the output of a denoising diffusion step, #
    # or a rotamer-sampling run) and the same ``jax.vmap`` call scores   #
    # them all with a single XLA compile. This is the main payoff of     #
    # the JAX port.                                                      #
    # ------------------------------------------------------------------ #
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
