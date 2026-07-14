"""jtmol.score — JAX scoring for molecular structures.

Re-exports the public surface of the scoring package:

* :class:`ScoreFunction`          — builds a weighted sum of energy terms.
* :class:`ScoreFunctionState`     — precomputed per-pose state bag.
* :class:`ScoreType`              — enumeration of per-term score channels.
* :class:`EnergyTerm`             — abstract base class for a single term.
* :func:`beta2016_score_function` — preset matching the Rosetta
                                    beta_nov2016_cart score function.

The individual term implementations live in subpackages
(``jtmol.score.ljlk``, ``jtmol.score.hbond``, ...) and are imported
lazily by :class:`ScoreFunction` the first time a weight is set for one
of their score types.
"""

from __future__ import annotations

from typing import Optional

from jtmol.score.batched import (
    batched_energy,
    batched_value_and_grad,
    pmap_batched_energy,
)
from jtmol.score.energy_term import EnergyTerm
from jtmol.score.score_function import ScoreFunction, ScoreFunctionState
from jtmol.score.score_types import ScoreType


def beta2016_score_function(
    param_db: Optional[object] = None,
    *,
    use_neighbor_list: bool = False,
    neighbor_list_cutoff: float = 6.0,
    soft_core_rep: Optional[float] = None,
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
        Optional :class:`jtmol.database.ParameterDatabase`. If
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
        # Imported here so the base ``jtmol.score`` package does not
        # pull the whole database layer in at module import time.
        from jtmol.database import ParameterDatabase

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
    # jtmol-only generic-atom harmonic bond-stretch + bond-angle. Weight
    # 0.5 mirrors the cart_bonded harmonic scale; exactly 0 on canonical
    # protein residues (generic-atom gate). Placeholder — refit to data.
    sfxn.set_weight(ScoreType.gen_bond_length, 0.5)
    sfxn.set_weight(ScoreType.gen_bond_angle, 0.5)

    if use_neighbor_list:
        for term in sfxn.terms:
            if hasattr(term, "use_neighbor_list"):
                term.use_neighbor_list = True
                term.neighbor_list_cutoff = neighbor_list_cutoff

    # Opt-in soft-core repulsion. OFF by default -> byte-identical to
    # upstream. When set, the LJLK term smoothly caps ``fa_ljrep`` per pair
    # to bound clash-driven forces (minimisation / fine-tuning stability).
    if soft_core_rep is not None:
        for term in sfxn.terms:
            if hasattr(term, "soft_core_rep_cap"):
                term.soft_core_rep_cap = float(soft_core_rep)

    return sfxn


# beta_jan25: the Haddox et al. 2025 refit of beta_nov16's Lennard-Jones
# parameters to reduce core overpacking (steric-clash bias). 15 parameters:
# the fa_rep weight plus the LJ radius + well-depth of the clash-prone
# carbon/hydrogen atom types, with the aliphatic carbons (CH0-CH3) uncoupled
# (they are tied in beta_nov16). Public values from Haddox et al.,
# "Using experimental results of protein design to guide biomolecular
# energy-function development" (bioRxiv 2025), github.com/Haddox/design_guided_optE.
# Jtmol atom-type names (Rosetta name in comment where it differs).
_BETA_JAN25_LJ = {  # jtmol atom type -> (lj_radius, lj_wdepth)
    "Caro": (2.02, 0.070),   # Rosetta aroC
    "Haro": (1.41, 0.021),
    "CH0": (2.04, 0.075),
    "CH1": (2.01, 0.062),
    "CH2": (2.01, 0.064),
    "CH3": (2.06, 0.064),
    "Hapo": (1.45, 0.021),
}
_BETA_JAN25_FA_REP = 0.6191  # beta_nov16: 0.55


def beta_jan25_score_function(
    param_db: Optional[object] = None,
    *,
    use_neighbor_list: bool = False,
    neighbor_list_cutoff: float = 6.0,
    soft_core_rep: Optional[float] = None,
) -> ScoreFunction:
    """Return a :class:`ScoreFunction` for the beta_jan25 energy function.

    beta_jan25 is beta_nov16 (:func:`beta2016_score_function`) with the Haddox
    et al. 2025 LJ refit: a stronger ``fa_rep`` weight and more repulsive LJ
    radii / well-depths on the aliphatic and aromatic carbon and hydrogen
    types, which reduces the core-overpacking (steric-clash) bias of
    beta_nov16. All other terms and weights are identical to beta2016; it takes
    the same options.

    Building on a freshly evolved parameter database leaves the process-default
    (beta_nov16) parameters untouched, so both energy functions can be used
    side by side (the LJLK annotation cache is keyed on the parameter set).
    """
    import attr

    from jtmol.database import ParameterDatabase

    if param_db is None:
        param_db = ParameterDatabase.get_default()
    ljlk = param_db.scoring.ljlk
    new_atp = tuple(
        attr.evolve(p, lj_radius=_BETA_JAN25_LJ[p.name][0],
                    lj_wdepth=_BETA_JAN25_LJ[p.name][1])
        if p.name in _BETA_JAN25_LJ else p
        for p in ljlk.atom_type_parameters
    )
    jan25_db = attr.evolve(
        param_db,
        scoring=attr.evolve(
            param_db.scoring, ljlk=attr.evolve(ljlk, atom_type_parameters=new_atp)),
    )
    sfxn = beta2016_score_function(
        param_db=jan25_db, use_neighbor_list=use_neighbor_list,
        neighbor_list_cutoff=neighbor_list_cutoff, soft_core_rep=soft_core_rep)
    sfxn.set_weight(ScoreType.fa_ljrep, _BETA_JAN25_FA_REP)
    return sfxn


__all__ = [
    "ScoreFunction",
    "ScoreFunctionState",
    "ScoreType",
    "EnergyTerm",
    "beta2016_score_function",
    "beta_jan25_score_function",
    "batched_energy",
    "batched_value_and_grad",
    "pmap_batched_energy",
]
