"""Fused nonbonded scoring on an adaptive Verlet neighbor list.

The substrate and batched fast engine for the LJLK / elec / lk_ball / hbond
pair terms. See :mod:`jtmol.nonbonded.verlet` for the neighbor list,
:mod:`jtmol.nonbonded.multipose` for batched (same- and multi-topology)
scoring, and :mod:`jtmol.nonbonded.fast` for the fast path.
"""

from __future__ import annotations

from jtmol.nonbonded.verlet import (
    NonbondedWorkPlan,
    VerletList,
    brute_within_cutoff_set,
    build_verlet_list,
    build_work_plan,
    max_displacement,
    maybe_rebuild,
    needs_rebuild,
    pair_distances,
    within_cutoff_pair_energy,
    within_cutoff_set,
)

__all__ = [
    "VerletList",
    "NonbondedWorkPlan",
    "build_verlet_list",
    "build_work_plan",
    "needs_rebuild",
    "maybe_rebuild",
    "max_displacement",
    "pair_distances",
    "within_cutoff_pair_energy",
    "within_cutoff_set",
    "brute_within_cutoff_set",
]
