"""Coordinate-constraint energy term (pure JAX).

JAX port of :mod:`tmol.score.constraint`. The upstream torch version
supports four functional forms (harmonic distance, harmonic coordinate,
bounded, circular harmonic) plus a rotamer-expansion bookkeeping layer.
This port focuses on the one form that actually gets exercised by
FastRelax in this codebase: **flat-bottom harmonic coordinate
restraints**.

Public surface:

* :class:`~jtmol.score.constraint.constraint_energy_term.ConstraintSet`
  -- user-supplied pytree of ``(target_coords, atom_indices, k, tol)``.
* :class:`~jtmol.score.constraint.constraint_energy_term.ConstraintEnergyTerm`
  -- :class:`~jtmol.score.energy_term.EnergyTerm` subclass producing
  ``ScoreType.constraint`` energies of ``0.5 * k * max(0, d - tol)^2``.
* :func:`~jtmol.score.constraint.utility.create_mainchain_coordinate_constraints`
  -- convenience helper that tethers every N/CA/C atom to its current
  position (FastRelax "freeze backbone").
"""

from __future__ import annotations

from jtmol.score.constraint.constraint_energy_term import (
    ConstraintEnergyTerm,
    ConstraintSet,
)
from jtmol.score.constraint.utility import (
    create_mainchain_coordinate_constraints,
)

__all__ = [
    "ConstraintEnergyTerm",
    "ConstraintSet",
    "create_mainchain_coordinate_constraints",
]
