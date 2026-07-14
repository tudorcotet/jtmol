"""Generic harmonic bond-length + bond-angle sub-terms (jtmol extension).

Asserts the two new :class:`GenBondedEnergyTerm` channels
(:attr:`ScoreType.gen_bond_length`, :attr:`ScoreType.gen_bond_angle`):

* fire (finite AND non-zero) on the cdk2 LG1 small-molecule ligand,
  whose real/strained geometry differs from our single-bond-covalent-radii
  ideals and ideal-hybridization angles, and
* are exactly 0 on a canonical protein (1ubq) -- the generic-atom gate
  keeps protein parity untouched, mirroring gen_torsions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_DATA = Path(__file__).resolve().parent / "data" / "protein_ligand_test"
_PDB = _DATA / "cdk2_complex.pdb"
_TMOL = _DATA / "cdk2.xtal-lig.mmff94.tmol"
_UBQ = Path(__file__).resolve().parent / "data" / "1ubq.pdb"


def _score_genbonded(ps):
    from jtmol.database import ParameterDatabase
    from jtmol.score.genbonded.genbonded_energy_term import GenBondedEnergyTerm
    from jtmol.score.score_types import ScoreType

    term = GenBondedEnergyTerm(ParameterDatabase.get_default())
    state = term.setup_for_pose_stack(ps)
    out = term(ps, state)
    return (
        float(np.asarray(out[ScoreType.gen_bond_length]).sum()),
        float(np.asarray(out[ScoreType.gen_bond_angle]).sum()),
    )


def test_ligand_bond_length_angle_nonzero() -> None:
    try:
        import biotite.structure.io.pdb as bpdb
    except ImportError:  # pragma: no cover
        pytest.skip("biotite not installed")
    if not _PDB.exists() or not _TMOL.exists():  # pragma: no cover
        pytest.skip("cdk2 protein-ligand fixtures not available")

    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    atoms = bpdb.PDBFile.read(str(_PDB)).get_structure(model=1)
    ps = pose_stack_from_biotite(atoms, ligand_params_files=[_TMOL])

    blen, bang = _score_genbonded(ps)
    assert np.isfinite(blen) and np.isfinite(bang), (blen, bang)
    assert blen != 0.0, "gen_bond_length is zero on the cdk2 ligand"
    assert bang != 0.0, "gen_bond_angle is zero on the cdk2 ligand"
    assert blen > 0.0 and bang > 0.0, "harmonic penalties must be non-negative"
    print(f"cdk2 ligand: gen_bond_length={blen:.6f}  gen_bond_angle={bang:.6f}")


def test_protein_bond_length_angle_exactly_zero() -> None:
    import jtmol

    ps = jtmol.pose_stack_from_pdb(str(_UBQ))
    blen, bang = _score_genbonded(ps)
    assert blen == 0.0, f"1ubq gen_bond_length must be 0, got {blen}"
    assert bang == 0.0, f"1ubq gen_bond_angle must be 0, got {bang}"


if __name__ == "__main__":  # pragma: no cover
    test_protein_bond_length_angle_exactly_zero()
    print("1ubq: gen_bond_length=0  gen_bond_angle=0  OK")
    try:
        test_ligand_bond_length_angle_nonzero()
    except pytest.skip.Exception as e:
        print(f"SKIP ligand: {e}")
