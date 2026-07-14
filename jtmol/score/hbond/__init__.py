"""Hydrogen-bond scoring in pure JAX.

Pair-centric port of :mod:`tmol.score.hbond`. The module structure is:

* :mod:`params` — parameter containers and per-pose donor/acceptor
  identification (:func:`hbond_params_for_pose_stack`).
* :mod:`potentials` — the analytic / polynomial pair terms
  (:func:`hbond_pair_V` and the four individual components).
* :mod:`hbond_energy_term` — the user-facing :class:`HBondEnergyTerm`
  plus the vectorised :func:`hbond_pose_scores` kernel.

The hbond polynomial evaluations require ``float64`` precision to match
upstream tmol (the coefficients span ~3 orders of magnitude with
alternating signs and a ``float32`` Horner chain loses ~0.004 per
degree-10 evaluation, accumulating to ~0.45 kcal/mol on 1ubq). We
enable JAX's ``x64`` mode at import time so that the polynomial tables
are stored and evaluated in ``float64`` without the user having to set
``JAX_ENABLE_X64=1`` manually. Geometry (coords / angles) stays in
whatever precision the caller supplies; ``_eval_packed_poly`` upcasts
``x`` to the polynomial dtype during Horner.
"""

import jax

# Upstream tmol uses ``bound_poly<11, double>``; matching its precision
# in JAX requires the global x64 flag. Set it before any JAX array is
# created so subsequent users inherit the config. Safe to call
# repeatedly — ``config.update`` is idempotent.
jax.config.update("jax_enable_x64", True)

from jtmol.score.hbond.hbond_energy_term import (
    HBondEnergyTerm,
    hbond_pose_scores,
)
from jtmol.score.hbond.params import (
    HYBRIDIZATION_NONE,
    HYBRIDIZATION_RING,
    HYBRIDIZATION_SP2,
    HYBRIDIZATION_SP3,
    HBondAtomParams,
    HBondGlobalParams,
    HBondPairParams,
    HBondParams,
    HBondPolynomials,
    hbond_params_for_pose_stack,
)
from jtmol.score.hbond.potentials import (
    AH_dist_V,
    AHD_angle_V,
    BAH_angle_V,
    chi_angle_V,
    hbond_pair_V,
)

__all__ = [
    "HYBRIDIZATION_NONE",
    "HYBRIDIZATION_RING",
    "HYBRIDIZATION_SP2",
    "HYBRIDIZATION_SP3",
    "HBondAtomParams",
    "HBondEnergyTerm",
    "HBondGlobalParams",
    "HBondPairParams",
    "HBondParams",
    "HBondPolynomials",
    "AH_dist_V",
    "AHD_angle_V",
    "BAH_angle_V",
    "chi_angle_V",
    "hbond_pair_V",
    "hbond_params_for_pose_stack",
    "hbond_pose_scores",
]
