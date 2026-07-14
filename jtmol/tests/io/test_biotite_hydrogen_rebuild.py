"""``pose_stack_from_biotite`` must rebuild missing hydrogens.

Regression guard for the bug where the biotite entry point called the
``pose_stack_construction`` bridge with the wrong argument order (swallowed by
a bare ``except``) and silently dropped to a fallback that leaves absent
hydrogens at the origin. On any heavy-atom-only input (crystal structures,
ESMFold folds) that inflated ``fa_rep`` by ~6 orders of magnitude.

The fix routes both the protein-only and ligand-complex paths through the
bridge (injecting ``canonical_ordering`` so integer ``res_types`` resolve), so
the biotite path builds H exactly like ``pose_stack_from_pdb``.
"""
from __future__ import annotations

import numpy as np
import pytest

_PDB = "jtmol/tests/data/1ubq.pdb"
_NONBONDED = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec", "hbond")


def _terms(sfxn, ps):
    return {s.name: float(np.asarray(v).sum()) for s, v in sfxn.score_by_term(ps).items()}


def test_biotite_rebuilds_hydrogens_matches_pdb_path():
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
    except ImportError:  # pragma: no cover
        pytest.skip("biotite not installed")
    import jtmol
    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    atoms = pdb.PDBFile.read(_PDB).get_structure(model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    atoms = atoms[atoms.element != "H"]  # strip H -> the biotite path must rebuild

    sfxn = jtmol.beta2016_score_function()
    tb = _terms(sfxn, pose_stack_from_biotite(atoms))
    tp = _terms(sfxn, jtmol.pose_stack_from_pdb(_PDB))

    # fa_rep is the canary: H-at-origin inflates it to ~1e7+. Rebuilt H keeps
    # it in the low hundreds, matching the H-rebuilding pdb path.
    assert tb["fa_ljrep"] < 1000.0, (
        f"fa_ljrep={tb['fa_ljrep']:.0f} -- hydrogens not rebuilt (left at origin)"
    )
    for t in _NONBONDED:
        assert abs(tb[t] - tp[t]) < 1.0, (
            f"{t}: biotite {tb[t]:.2f} vs pdb {tp[t]:.2f} -- H-rebuild parity broken"
        )


if __name__ == "__main__":  # pragma: no cover
    import sys
    try:
        test_biotite_rebuilds_hydrogens_matches_pdb_path()
        print("PASS")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc!r}")
        sys.exit(1)
