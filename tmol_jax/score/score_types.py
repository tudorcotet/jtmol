"""Enumeration of per-term score types.

Direct port of :mod:`tmol.score.score_types`. In upstream tmol this uses a
custom ``AutoNumber`` class based on ``enum.Enum``; we use the standard
library's :class:`enum.IntEnum` so values are plain Python ints, which
makes them trivially usable as dict keys, ``jax.jit``-static args, and as
indices into a per-type weight vector if we ever want to materialize one.

The list below matches upstream tmol exactly (same members, same order).
The ``n_score_types`` sentinel is kept to preserve upstream's convention
of using ``ScoreType.n_score_types.value`` to size per-type arrays.

Note on :class:`IntEnum` vs ``AutoNumber``: both produce 0-indexed values
in declaration order, so ``ScoreType.fa_ljatr.value == 0`` here just as in
upstream tmol. Users should always address score types by *name*, not by
value, but the equivalence lets older code and data files round-trip.
"""

from __future__ import annotations

from enum import IntEnum


class ScoreType(IntEnum):
    """All per-term energy channels that :class:`~tmol_jax.score.ScoreFunction`
    can weight and sum.

    One :class:`EnergyTerm` may provide several score types (for instance, the
    LJLK term provides ``fa_ljatr``, ``fa_ljrep``, and ``fa_lk``). The score
    function owns the weights; the terms own the unweighted evaluations.
    """

    # LJ / LK pair terms (LJLKEnergyTerm)
    fa_ljatr = 0
    fa_ljrep = 1
    fa_lk = 2

    # Electrostatics (ElecEnergyTerm)
    fa_elec = 3

    # Hydrogen bonding (HBondEnergyTerm)
    hbond = 4

    # Cartesian bonded (CartBondedEnergyTerm)
    cart_lengths = 5
    cart_angles = 6
    cart_torsions = 7
    cart_impropers = 8
    cart_hxltorsions = 9

    # Constraints (ConstraintEnergyTerm)
    constraint = 10

    # Disulfide bonds (DisulfideEnergyTerm)
    disulfide = 11

    # Backbone torsions (BackboneTorsionEnergyTerm)
    omega = 12
    rama = 13

    # Dunbrack rotamer library (DunbrackEnergyTerm)
    dunbrack_rot = 14
    dunbrack_rotdev = 15
    dunbrack_semirot = 16

    # Anisotropic lk (LKBallEnergyTerm)
    lk_ball_iso = 17
    lk_ball = 18
    lk_bridge = 19
    lk_bridge_uncpl = 20

    # Per-residue reference energies (RefEnergyTerm)
    ref = 21

    # Sentinel — keep last. Upstream sizes arrays by this value.
    n_score_types = 22


__all__ = ["ScoreType"]
