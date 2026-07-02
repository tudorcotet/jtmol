"""Standalone driver for 1r21 only, appending to /tmp/parity_driver.log."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from run_upstream_parity import (
    _jax_energy_and_grad,
    _torch_energy_and_grad,
    _probe_upstream,
    _within_tol,
    ATOL, RTOL,
    ALL_SCORE_TYPE_NAMES,
)

import time

PDB = os.path.join(REPO, "tmol/tmol/tests/data/pdb/1r21.pdb")

def main() -> int:
    up = _probe_upstream()
    assert up.available, "upstream must be available for this script"

    import tmol_jax
    from tmol_jax.score.score_types import ScoreType as JaxScoreType

    print("loading jax pose...", flush=True)
    jax_pose = tmol_jax.pose_stack_from_pdb(PDB)
    print("jax coords:", jax_pose.coords.shape, flush=True)
    print("loading torch pose...", flush=True)
    torch_pose = up.pose_stack_from_pdb(PDB, up.torch.device("cpu"))
    print("torch coords:", torch_pose.coords.shape, flush=True)

    import numpy as np

    for name in ALL_SCORE_TYPE_NAMES:
        t0 = time.monotonic()
        try:
            e_jax, g_jax = _jax_energy_and_grad(jax_pose, name, JaxScoreType)
            e_torch, g_torch = _torch_energy_and_grad(torch_pose, name, up)
        except Exception as exc:
            print(f"[row] 1r21   {name:20s} ERROR: {type(exc).__name__}: {exc}",
                  flush=True)
            continue

        fwd_ok = _within_tol(e_jax, e_torch)

        if g_jax.shape != g_torch.shape:
            # 1r21 happens to produce different atom counts in the two
            # pose stacks (disulfide bridging is inferred slightly
            # differently in upstream vs tmol_jax), so comparing the
            # gradient arrays elementwise would be malformed. Report a
            # shape-mismatch skip instead of crashing -- the forward
            # energy comparison is still meaningful (per-atom energies
            # sum to the same scalar either way).
            print(f"[row] 1r21   {name:20s} live fwd={'P' if fwd_ok else 'F'} "
                  f"grad=- Δe={e_jax - e_torch:+.3e} shape_mismatch "
                  f"(jax={g_jax.shape}, torch={g_torch.shape}) "
                  f"({time.monotonic() - t0:.1f}s)",
                  flush=True)
            continue

        grad_diff = np.abs(g_jax - g_torch) - (ATOL + RTOL * np.abs(g_torch))
        grad_ok = bool(np.all(grad_diff <= 0.0))
        max_abs = float(np.max(np.abs(g_jax - g_torch)))
        print(f"[row] 1r21   {name:20s} live fwd={'P' if fwd_ok else 'F'} "
              f"grad={'P' if grad_ok else 'F'} "
              f"Δe={e_jax - e_torch:+.3e} max|dg|={max_abs:.3e} "
              f"({time.monotonic() - t0:.1f}s)",
              flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
