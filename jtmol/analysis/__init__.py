"""Structure-analysis tools built on jtmol's differentiable energy.

* :func:`physics_quality` / :func:`energy_gradient_norm`: label-free
  physics-consistency quality & OOD scoring for a structure.
* Hessian / normal-mode analysis: per-atom B-factors (:func:`bfactors`),
  vibrational entropy (:func:`vibrational_entropy`), and the underlying
  :func:`energy_hessian` / :func:`soft_modes`. See
  :mod:`jtmol.analysis.hessian`.
"""

from jtmol.analysis.hessian import (
    bfactors,
    energy_hessian,
    hvp,
    real_atom_indices,
    soft_modes,
    stiff_modes,
    vibrational_entropy,
)
from jtmol.analysis.quality import energy_gradient_norm, physics_quality

__all__ = [
    "physics_quality",
    "energy_gradient_norm",
    "energy_hessian",
    "hvp",
    "soft_modes",
    "stiff_modes",
    "bfactors",
    "vibrational_entropy",
    "real_atom_indices",
]
