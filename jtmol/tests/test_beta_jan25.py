"""beta_jan25 energy function: the Haddox et al. 2025 LJ refit of beta_nov16.

Locks the 15 refit parameters (fa_rep weight + LJ radius/well-depth of the
clashing carbon/hydrogen atom types, aliphatic carbons uncoupled) and the
per-parameter-set LJLK annotation cache that lets beta_jan25 and beta_nov16
coexist in one process (the annotation is keyed on the LJ parameters, not
just their presence on the packed block types).
"""

from __future__ import annotations

import numpy as np

import jtmol
from jtmol.score import _BETA_JAN25_FA_REP, _BETA_JAN25_LJ

_PDB = "jtmol/tests/data/1ubq.pdb"


def _terms(sfxn, ps):
    return {s.name: float(np.asarray(v).sum())
            for s, v in sfxn.score_by_term(ps).items()}


def test_beta_jan25_params_are_the_published_15():
    # 7 atom types x (radius, well-depth) = 14, plus the fa_rep weight = 15.
    assert _BETA_JAN25_FA_REP == 0.6191            # beta_nov16: 0.55
    assert len(_BETA_JAN25_LJ) == 7
    assert _BETA_JAN25_LJ["CH3"] == (2.06, 0.064)
    assert _BETA_JAN25_LJ["CH0"] == (2.04, 0.075)
    assert _BETA_JAN25_LJ["Caro"] == (2.02, 0.070)  # Rosetta aroC
    # aliphatic carbons uncoupled (distinct radii), unlike beta_nov16.
    radii = {_BETA_JAN25_LJ[c][0] for c in ("CH0", "CH1", "CH2", "CH3")}
    assert len(radii) > 1


def test_beta_jan25_is_more_repulsive_and_isolated():
    ps = jtmol.pose_stack_from_pdb(_PDB)
    t16 = _terms(jtmol.beta2016_score_function(), ps)
    t25 = _terms(jtmol.beta_jan25_score_function(), ps)

    # the LJ-parameter refit propagates (not just the weight): the RAW
    # repulsion (before weighting) is higher, and attraction changes too.
    assert t25["fa_ljrep"] / 0.6191 > t16["fa_ljrep"] / 0.55 + 1.0
    assert abs(t25["fa_ljatr"] - t16["fa_ljatr"]) > 1.0

    # switching back to beta_nov16 on the SAME pose reproduces it exactly:
    # the per-parameter-set annotation cache must re-annotate correctly.
    t16b = _terms(jtmol.beta2016_score_function(), ps)
    assert abs(sum(t16b.values()) - sum(t16.values())) < 1e-3

    # the process-default parameter database is untouched by building jan25.
    from jtmol.database import ParameterDatabase
    ch3 = next(p for p in ParameterDatabase.get_default().scoring.ljlk
               .atom_type_parameters if p.name == "CH3")
    assert abs(ch3.lj_radius - 2.01176) < 1e-5  # still the beta_nov16 value


if __name__ == "__main__":  # pragma: no cover
    import sys
    bad = []
    for fn in (test_beta_jan25_params_are_the_published_15,
               test_beta_jan25_is_more_repulsive_and_isolated):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {exc!r}")
            bad.append(fn.__name__)
    sys.exit(1 if bad else 0)
