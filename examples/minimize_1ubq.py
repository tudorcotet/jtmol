"""End-to-end worked example: load 1ubq, score it, minimize, save.

This is the "hello world" for tmol_jax. It walks through the five things
every user eventually wants to do:

    1. Load a PDB into a :class:`PoseStack` pytree.
    2. Build the ``beta_nov2016_cart`` score function.
    3. Look at the per-term energy breakdown.
    4. Run a short Cartesian minimization.
    5. Write the minimized pose back out as a PDB.

Every stage is wrapped in ``try/except`` with an informative message, so
if one layer of the port is still being landed (e.g. the packer, the
minimizer) you can see exactly which call failed without the whole
script silently no-op'ing.

Run it:

    python examples/minimize_1ubq.py

You can point it at a different PDB by setting the ``TMOL_JAX_PDB``
environment variable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------- #
# Locate an input PDB. We default to the bundled 1ubq test fixture — a   #
# 76-residue ubiquitin crystal structure — so this script runs out of    #
# the box without the user providing anything.                           #
# ---------------------------------------------------------------------- #
DEFAULT_PDB = (
    Path(__file__).resolve().parent.parent
    / "tmol_jax"
    / "tests"
    / "data"
    / "1ubq.pdb"
)

INPUT_PDB = Path(os.environ.get("TMOL_JAX_PDB", str(DEFAULT_PDB)))
OUTPUT_PDB = Path(os.environ.get("TMOL_JAX_OUT", "1ubq_min.pdb"))
N_MIN_STEPS = int(os.environ.get("TMOL_JAX_STEPS", "50"))


def _banner(title: str) -> None:
    """Print a visual section header so the log is skimmable."""
    line = "-" * 60
    print(f"\n{line}\n{title}\n{line}")


def main() -> int:
    _banner("tmol_jax: load, score, minimize, save")
    print(f"Input PDB: {INPUT_PDB}")
    print(f"Output PDB: {OUTPUT_PDB}")
    print(f"Minimization steps: {N_MIN_STEPS}")

    if not INPUT_PDB.exists():
        print(f"ERROR: Input PDB not found at {INPUT_PDB}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ #
    # 1. Import the library.                                             #
    #                                                                    #
    # We use the top-level ``from tmol_jax import X`` form — the package #
    # loads symbols lazily, so a broken submodule won't prevent import,  #
    # but an attribute access on that symbol will surface the real       #
    # error right here. That's why we try each symbol separately.        #
    # ------------------------------------------------------------------ #
    try:
        from tmol_jax import (
            pose_stack_from_pdb,
            beta2016_score_function,
            run_cart_min,
            write_pose_stack_pdb,
        )
    except Exception as exc:
        print(f"ERROR: failed to import tmol_jax symbols: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # 2. Parse the PDB into a PoseStack.                                 #
    #                                                                    #
    # ``pose_stack_from_pdb`` returns a PoseStack of size n_poses=1.     #
    # Everything downstream is already batched along that axis; you can #
    # grow the batch dimension later by vmap'ing or stacking pose       #
    # stacks. See examples/vmap_over_poses.py for that.                 #
    # ------------------------------------------------------------------ #
    _banner("Step 1/4: Load PDB into PoseStack")
    try:
        t0 = time.perf_counter()
        pose_stack = pose_stack_from_pdb(str(INPUT_PDB))
        print(f"  loaded in {time.perf_counter() - t0:.2f}s")
        print(f"  n_poses:  {pose_stack.n_poses}")
        print(f"  coords:   shape={pose_stack.coords.shape}"
              f" dtype={pose_stack.coords.dtype}")
    except Exception as exc:
        print(f"ERROR: pose_stack_from_pdb failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # 3. Build the score function.                                       #
    #                                                                    #
    # ``beta2016_score_function`` is the full beta_nov2016_cart preset.  #
    # Under the hood it instantiates the ~10 EnergyTerm subclasses and   #
    # sets the 21 channel weights. All terms are lazy-imported on their  #
    # first weight assignment, so any term-local bug surfaces here.      #
    # ------------------------------------------------------------------ #
    _banner("Step 2/4: Build beta_nov2016_cart score function")
    try:
        sfxn = beta2016_score_function()
        active = [st.name for st, w in sfxn.weights.items() if w != 0.0]
        print(f"  {len(active)} active channels")
    except Exception as exc:
        print(f"ERROR: beta2016_score_function failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # 3b. Print the per-term energy breakdown.                           #
    #                                                                    #
    # ``score_by_term`` returns ``dict[ScoreType, float32[n_poses]]`` of #
    # *weighted* per-channel energies. The total is just the sum.        #
    # ------------------------------------------------------------------ #
    _banner("Step 3/4: Compute energy breakdown")
    try:
        per_term = sfxn.score_by_term(pose_stack)
        total = sfxn(pose_stack)
        print(f"  total energy per pose: {list(total)}")
        print(f"  per-term breakdown:")
        # Sort by absolute magnitude for readability.
        rows = [(st.name, float(v.sum())) for st, v in per_term.items()]
        rows.sort(key=lambda r: -abs(r[1]))
        for name, value in rows:
            print(f"    {name:<20s} {value:12.4f}")
    except Exception as exc:
        print(f"ERROR: energy breakdown failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # 4. Run Cartesian minimization.                                     #
    #                                                                    #
    # ``run_cart_min`` wraps the JAX LBFGS minimizer (see               #
    # tmol_jax.optimization.lbfgs). It returns a new PoseStack with     #
    # updated ``coords``; the rest of the pytree (block types,          #
    # connectivity, etc.) is unchanged. We pass ``max_iter`` via         #
    # ``optimizer_kwargs`` — the minimizer falls back to defaults if     #
    # the kwarg name has drifted.                                        #
    # ------------------------------------------------------------------ #
    _banner(f"Step 4/4: Run {N_MIN_STEPS} steps of cart min")
    try:
        t0 = time.perf_counter()
        try:
            minimized = run_cart_min(
                pose_stack,
                sfxn,
                optimizer_kwargs={"max_iter": N_MIN_STEPS},
            )
        except TypeError:
            # Older/newer API signature — fall back to defaults and let the
            # minimizer pick its own iteration count. Still better than
            # silently doing nothing.
            print("  (optimizer_kwargs not accepted; falling back to defaults)")
            minimized = run_cart_min(pose_stack, sfxn)
        print(f"  minimized in {time.perf_counter() - t0:.2f}s")

        e_final = sfxn(minimized)
        print(f"  final energy per pose: {list(e_final)}")
        delta = float((e_final - total).sum())
        print(f"  delta from start:      {delta:+.4f}")
    except Exception as exc:
        print(f"ERROR: run_cart_min failed: {exc!r}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # 5. Save the minimized pose.                                        #
    # ------------------------------------------------------------------ #
    _banner("Save output PDB")
    try:
        write_pose_stack_pdb(minimized, str(OUTPUT_PDB))
        print(f"  wrote {OUTPUT_PDB}")
    except Exception as exc:
        print(f"ERROR: write_pose_stack_pdb failed: {exc!r}", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
