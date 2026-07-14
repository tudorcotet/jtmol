"""Example: ``jax.grad`` of the jtmol energy w.r.t. arbitrary DOFs.

The killer app of the JAX port is that the whole score function is a
plain differentiable Python function. Drop it into a training loop,
compose it with a neural network, take gradients through it — all
standard JAX idioms apply.

This script demonstrates two differentiable folding patterns:

    1. **Cartesian gradient** — ``dE / d(xyz)``. This is what the
       minimizer already uses; we compute it by hand to show how to
       plug it into an external training loop.

    2. **Gradient through a DOF representation** — ``dE / d(dofs)``
       where ``dofs`` are the (scalar) variables an ML model would
       actually predict (e.g. an offset from a reference pose, or a
       set of per-atom displacements). The point is that you don't
       need to touch jtmol internals to change what "the variables"
       are: any function that builds coordinates from your DOFs and
       passes them to the score function is differentiable end-to-end.

Once you have ``grad_fn = jax.grad(energy_of_dofs)``, hooking it up
to Optax, Flax, or any JAX ML stack is a one-liner.

Run it:

    python examples/differentiable_folding.py
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
N_SGD_STEPS = int(os.environ.get("TMOL_JAX_STEPS", "20"))
LEARNING_RATE = float(os.environ.get("TMOL_JAX_LR", "1e-3"))
SEED = int(os.environ.get("TMOL_JAX_SEED", "0"))


def _banner(title: str) -> None:
    line = "-" * 60
    print(f"\n{line}\n{title}\n{line}")


def main() -> int:
    _banner("jtmol: differentiable folding with jax.grad")
    print(f"Input PDB:        {INPUT_PDB}")
    print(f"SGD steps:        {N_SGD_STEPS}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Seed:             {SEED}")

    if not INPUT_PDB.exists():
        print(f"ERROR: Input PDB not found at {INPUT_PDB}", file=sys.stderr)
        return 2

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
    # Set up the pose and score function.                                #
    #                                                                    #
    # ``state`` is the per-term precomputed state bag. We capture it     #
    # here and close over it so the differentiable energy function does  #
    # not redo the setup at every gradient step.                         #
    # ------------------------------------------------------------------ #
    _banner("Step 1/3: Load pose + build score function")
    try:
        pose_stack = pose_stack_from_pdb(str(INPUT_PDB))
        sfxn = beta2016_score_function()
        state = sfxn.setup_for_pose_stack(pose_stack)
        base_coords = pose_stack.coords  # [n_poses, max_n_atoms, 3]
        print(f"  base_coords shape: {base_coords.shape}")
    except Exception as exc:
        print(f"ERROR: pose/sfxn setup failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Pattern 1: dE/d(xyz) — gradient w.r.t. Cartesian coordinates.      #
    #                                                                    #
    # This is the simplest form. The "DOFs" are literally the atomic     #
    # coordinates, and we differentiate the total energy with respect to #
    # them. You'd use this to plug jtmol into, say, an MD-style       #
    # integrator or a coordinate-space refinement loop.                  #
    # ------------------------------------------------------------------ #
    _banner("Step 2/3: jax.grad of E w.r.t. coordinates")
    try:
        def energy_of_coords(coords: jnp.ndarray) -> jnp.ndarray:
            # ``sfxn(...)`` returns shape ``[n_poses]``; ``.sum()`` makes
            # it a scalar, which is what ``jax.grad`` requires.
            return sfxn(pose_stack, state=state, coords_override=coords).sum()

        grad_of_coords = jax.jit(jax.grad(energy_of_coords))

        t0 = time.perf_counter()
        g = grad_of_coords(base_coords)
        g.block_until_ready()
        print(f"  grad compile+run: {time.perf_counter() - t0:.3f}s")
        print(f"  grad shape: {g.shape}")
        # Force-norm is a quick sanity signal — a pose at a saddle has
        # grad_norm close to zero; a totally noisy pose is usually huge.
        gnorm = float(jnp.linalg.norm(g))
        print(f"  ||grad||_2: {gnorm:.4f}")
    except Exception as exc:
        print(f"ERROR: jax.grad over coords failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Pattern 2: dE/d(dofs) where dofs are a learned / latent variable. #
    #                                                                    #
    # This is the common case for ML integration: your model produces    #
    # some variables (here, per-atom offsets), and you want to train it  #
    # using the jtmol energy as a loss. ``energy_of_dofs`` is a pure  #
    # JAX function — compose it with any optimizer (optax, handwritten   #
    # SGD, ...) and the gradient flows end-to-end.                       #
    #                                                                    #
    # We mimic a tiny SGD loop here with a single "learned" parameter —  #
    # a per-atom displacement initialized to zeros. The update rule is  #
    # plain gradient descent; the point is that the energy is the loss, #
    # no special integration is needed.                                  #
    # ------------------------------------------------------------------ #
    _banner(f"Step 3/3: Gradient descent through DOFs ({N_SGD_STEPS} steps)")
    try:
        # Initial DOFs: the displacement we would "predict". Starts at
        # zero so the initial energy equals the unperturbed pose energy.
        dofs = jnp.zeros_like(base_coords)

        def build_coords(dofs: jnp.ndarray) -> jnp.ndarray:
            """Map DOFs -> Cartesian coords. Replace with whatever your
            model needs (quaternion -> xyz, torsion -> xyz, latent -> xyz,
            etc.). The only requirement is that it is JAX-pure."""
            return base_coords + dofs

        def energy_of_dofs(dofs: jnp.ndarray) -> jnp.ndarray:
            coords = build_coords(dofs)
            return sfxn(pose_stack, state=state, coords_override=coords).sum()

        # ``value_and_grad`` gives us both E and dE/d(dofs) in one trace —
        # exactly what a training loop needs (loss + grad per step).
        loss_and_grad = jax.jit(jax.value_and_grad(energy_of_dofs))

        # Seed some non-zero DOFs so the descent has something to do;
        # this is a stand-in for an ML model producing a noisy prediction.
        rng = jax.random.PRNGKey(SEED)
        dofs = dofs + 0.05 * jax.random.normal(rng, dofs.shape)
        e0, _ = loss_and_grad(dofs)
        print(f"  step   0: E = {float(e0):12.4f}")

        t0 = time.perf_counter()
        for step in range(1, N_SGD_STEPS + 1):
            loss, grad = loss_and_grad(dofs)
            # Plain SGD. Replace with ``optax.adam`` / ``optax.sgd`` if you
            # want momentum and adaptive step sizes; the call site is the
            # same.
            dofs = dofs - LEARNING_RATE * grad
            if step % max(1, N_SGD_STEPS // 10) == 0 or step == N_SGD_STEPS:
                print(f"  step {step:3d}: E = {float(loss):12.4f}")
        print(f"  total descent time: {time.perf_counter() - t0:.2f}s")

        e_final = float(energy_of_dofs(dofs))
        print(f"  final  E: {e_final:12.4f}   (delta {e_final - float(e0):+.4f})")
    except Exception as exc:
        print(f"ERROR: DOF gradient descent failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # Takeaway for ML integration.                                       #
    #                                                                    #
    # ``energy_of_dofs`` is a pure, JIT-friendly, vmap-friendly,         #
    # grad-friendly function. You can:                                   #
    #                                                                    #
    #   * Plug it into an Optax training loop as the loss.               #
    #   * vmap it over a batch of DOFs for parallel folding.             #
    #   * Compose it with a neural net that emits ``dofs`` from a        #
    #     sequence or latent — gradients flow back into the NN weights.  #
    #   * Use ``jax.hessian`` for second-order methods.                  #
    #                                                                    #
    # No custom kernels, no C++ build, no backward-pass bookkeeping. #
    # ------------------------------------------------------------------ #
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
