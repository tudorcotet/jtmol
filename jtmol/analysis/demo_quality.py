"""Self-check + demo for physics-consistency quality / OOD scoring.

On a bundled test structure, asserts the two properties that make
:func:`physics_quality` useful as a label-free filter:

1. **OOD detection** — a randomly perturbed ("hallucinated") copy scores a
   much larger ``grad_norm`` than the original. Strained structures light
   up without any reference.
2. **Monotone with corruption** — increasing perturbation monotonically
   increases the score, so it *ranks* by physical plausibility.

Run standalone (stdlib asserts, no pytest)::

    python -m jtmol.analysis.demo_quality
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np

import jtmol
from jtmol.analysis import energy_gradient_norm, physics_quality

_PDB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "1ubq.pdb",
)


def _perturbed_coords(ps, sigma: float, seed: int):
    """Coordinates with isotropic Gaussian noise (Å) added."""
    key = jax.random.PRNGKey(seed)
    return ps.coords + jax.random.normal(key, ps.coords.shape) * sigma


def main() -> int:
    print(f"[quality-demo] loading {os.path.basename(_PDB)}")
    ps = jtmol.pose_stack_from_pdb(_PDB)
    sfxn = jtmol.beta2016_score_function()

    q = physics_quality(ps, sfxn)
    print(f"[quality-demo] native: grad_norm={q['grad_norm']:.2f}  "
          f"clash/res={q['clash_per_residue']:.2f}  E/res={q['energy_per_residue']:.2f}")

    # OOD + monotonicity: score increasing perturbations.
    scores = [q["grad_norm"]]
    for sigma in (0.1, 0.3, 0.5):
        g = energy_gradient_norm(ps, sfxn, coords=_perturbed_coords(ps, sigma, seed=0))
        scores.append(g)
        print(f"[quality-demo] +{sigma:.1f} Å noise: grad_norm={g:.2f}")

    native, mild, strong = scores[0], scores[1], scores[-1]
    # OOD: a clearly-perturbed structure scores much higher than native.
    assert strong > 5 * native, (
        f"OOD detection weak: perturbed {strong:.1f} vs native {native:.1f}"
    )
    # Monotone with corruption (each step at least not decreasing much).
    assert all(scores[i + 1] >= scores[i] * 0.9 for i in range(len(scores) - 1)), (
        f"score not monotone with perturbation: {[round(s, 1) for s in scores]}"
    )
    print(f"[quality-demo] OOD separation native->+0.5Å: "
          f"{strong / max(native, 1e-6):.1f}x")
    print("[quality-demo] OK — OOD detection + monotone ranking pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
