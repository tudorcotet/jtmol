"""tmol_jax.score — JAX scoring for molecular structures.

Re-exports the public surface of the scoring package:

* :class:`ScoreFunction`          — builds a weighted sum of energy terms.
* :class:`ScoreFunctionState`     — precomputed per-pose state bag.
* :class:`ScoreType`              — enumeration of per-term score channels.
* :class:`EnergyTerm`             — abstract base class for a single term.
* :func:`beta2016_score_function` — preset matching the Rosetta
                                    beta_nov2016_cart score function.

The individual term implementations live in subpackages
(``tmol_jax.score.ljlk``, ``tmol_jax.score.hbond``, ...) and are imported
lazily by :class:`ScoreFunction` the first time a weight is set for one
of their score types.
"""

from __future__ import annotations

from typing import Optional

from tmol_jax.score.batched import (
    batched_energy,
    batched_value_and_grad,
    pmap_batched_energy,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_function import ScoreFunction, ScoreFunctionState
from tmol_jax.score.score_types import ScoreType


def beta2016_score_function(
    param_db: Optional[object] = None,
    *,
    use_neighbor_list: bool = False,
    neighbor_list_cutoff: float = 6.0,
) -> ScoreFunction:
    """Return a :class:`ScoreFunction` configured for beta_nov2016_cart.

    The weights come directly from upstream tmol's
    ``_non_memoized_beta2016`` helper. See
    https://pubs.acs.org/doi/10.1021/acs.jctc.6b0081 and
    https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00125 for the
    scientific background.

    Parameters
    ----------
    param_db
        Optional :class:`tmol_jax.database.ParameterDatabase`. If
        omitted, we fall back to the process default (via
        ``ParameterDatabase.get_default()``).
    use_neighbor_list
        Toggle the cell-list / half-list path on the LJLK term (which
        contributes 3 channels: fa_ljatr, fa_ljrep, fa_lk). The dense
        ``[P, N, N]`` pair tensor becomes ``[P, N, max_nbr]`` with the
        ``neighbor_list_cutoff`` (default 6 Å, matches LJ/LK ``cpoly_dmax``).
        elec / lk_ball still use the dense path; wiring NL into those
        is a real refactor (1-2 days each) and is left for a future
        pass — combined with :func:`batched_energy`'s ``lax.map``
        chunking, LJLK-NL alone usually closes the OOM on
        ~5000-atom complexes.
    neighbor_list_cutoff
        Cutoff radius for the cell-list builder, Å. The default 6.0
        matches LJLK's ``cpoly_dmax``; lowering it past that drops real
        pair interactions and shifts the score.

    Returns
    -------
    ScoreFunction
        A fresh score function with all 21 beta_nov2016_cart channels
        weighted. Terms are imported lazily in
        :meth:`ScoreFunction.set_weight`, so this is also the moment
        when any term-local import errors surface — that's intentional,
        it keeps the error message close to the user's call site.
    """
    if param_db is None:
        # Imported here so the base ``tmol_jax.score`` package does not
        # pull the whole database layer in at module import time.
        from tmol_jax.database import ParameterDatabase

        param_db = ParameterDatabase.get_default()

    sfxn = ScoreFunction(param_db)

    # Weights are a direct copy of tmol.score.__init__._non_memoized_beta2016.
    # Keep these in lockstep with upstream — if upstream tweaks a weight,
    # regenerate here rather than edit in place.
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.fa_elec, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.lk_ball_iso, -0.38)
    sfxn.set_weight(ScoreType.lk_ball, 0.92)
    sfxn.set_weight(ScoreType.lk_bridge, -0.33)
    sfxn.set_weight(ScoreType.lk_bridge_uncpl, -0.33)
    sfxn.set_weight(ScoreType.omega, 0.48)
    sfxn.set_weight(ScoreType.rama, 0.50)
    sfxn.set_weight(ScoreType.disulfide, 1.25)
    sfxn.set_weight(ScoreType.cart_lengths, 0.5)
    sfxn.set_weight(ScoreType.cart_angles, 0.5)
    sfxn.set_weight(ScoreType.cart_torsions, 0.5)
    sfxn.set_weight(ScoreType.cart_impropers, 0.5)
    sfxn.set_weight(ScoreType.cart_hxltorsions, 0.5)
    sfxn.set_weight(ScoreType.dunbrack_rot, 0.76)
    sfxn.set_weight(ScoreType.dunbrack_rotdev, 0.69)
    sfxn.set_weight(ScoreType.dunbrack_semirot, 0.78)
    sfxn.set_weight(ScoreType.ref, 1.0)
    sfxn.set_weight(ScoreType.gen_torsions, 1.0)

    if use_neighbor_list:
        for term in sfxn.terms:
            if hasattr(term, "use_neighbor_list"):
                term.use_neighbor_list = True
                term.neighbor_list_cutoff = neighbor_list_cutoff

    return sfxn


__all__ = [
    "ScoreFunction",
    "ScoreFunctionState",
    "ScoreType",
    "EnergyTerm",
    "beta2016_score_function",
    "batched_energy",
    "batched_value_and_grad",
    "pmap_batched_energy",
]
