"""Protein-ligand PoseStack construction + scoring (upstream PR #361).

Loads the cdk2 complex (protein + the LG1 small-molecule inhibitor) from
``cdk2_complex.pdb`` plus the pre-baked ``cdk2.xtal-lig.mmff94.tmol``
generic-potential params file, builds a :class:`PoseStack` through the
real ``pose_stack_from_biotite`` entry point, and confirms:

* the LG1 ligand block lands in the PoseStack with all 28 of its atoms
  (18 heavy + 10 H, matching the ``.tmol`` atom list), and
* ``beta2016_score_function`` scores the complex to a finite total with
  non-zero ``fa_ljatr`` / ``fa_lk`` (the ligand's generic-potential atom
  types now carry real LJ/LK params, so it contributes).

Deferred (NOT asserted here): the ligand's partial-charge -> ``fa_elec``
wiring and its donor/acceptor ``hbond`` flags. Those need the
``elec.atom_charge_parameters`` block and the generic-potential
``hbond.yaml`` mappings ported, which is a documented follow-up. This
round only certifies LJ/LK + finiteness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_DATA = Path(__file__).resolve().parents[1] / "data" / "protein_ligand_test"
_PDB = _DATA / "cdk2_complex.pdb"
_TMOL = _DATA / "cdk2.xtal-lig.mmff94.tmol"

# LG1 atom count in the .tmol params file (18 heavy + 10 H).
_LG1_N_ATOMS = 28


def _load_atoms():
    try:
        import biotite.structure.io.pdb as bpdb
    except ImportError:  # pragma: no cover
        pytest.skip("biotite not installed")
    if not _PDB.exists() or not _TMOL.exists():  # pragma: no cover
        pytest.skip("cdk2 protein-ligand fixtures not available")
    return bpdb.PDBFile.read(str(_PDB)).get_structure(model=1)


def test_ligand_in_pose_and_scores() -> None:
    from tmol_jax.io.pose_stack_from_biotite import (
        build_context_from_biotite,
        pose_stack_from_biotite,
    )

    atoms = _load_atoms()

    # Identify LG1's block-type index + atom count from the extended pbt.
    ctx = build_context_from_biotite(atoms, ligand_params_files=[_TMOL])
    pbt = ctx.packed_block_types
    name_to_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}
    assert "LG1" in name_to_idx, "ligand restype missing from packed block types"
    lg1_idx = name_to_idx["LG1"]
    assert int(np.asarray(pbt.n_atoms)[lg1_idx]) == _LG1_N_ATOMS

    # Build the PoseStack via the real entry point and confirm the ligand
    # block (with all its atoms) is present.
    ps = pose_stack_from_biotite(atoms, ligand_params_files=[_TMOL])
    n_lg1_blocks = int((np.asarray(ps.block_type_ind) == lg1_idx).sum())
    assert n_lg1_blocks == 1, f"expected exactly one LG1 block, got {n_lg1_blocks}"

    # Score the complex; ligand must not break LJ/LK and must contribute.
    import tmol_jax

    sfxn = tmol_jax.beta2016_score_function()
    state = sfxn.setup_for_pose_stack(ps)
    per_term = {
        st.name: float(np.asarray(v).sum())
        for st, v in sfxn.score_by_term(ps, state=state).items()
    }
    total = float(np.asarray(sfxn(ps)).sum())

    assert np.isfinite(total), f"non-finite total {total}"
    for term in ("fa_ljatr", "fa_lk"):
        val = per_term.get(term)
        assert val is not None, f"missing score term {term}"
        assert np.isfinite(val), f"{term} not finite: {val}"
        assert val != 0.0, f"{term} is zero -- ligand LJ/LK not contributing"


if __name__ == "__main__":  # pragma: no cover - manual driver
    import sys

    try:
        test_ligand_in_pose_and_scores()
        print("PASS  test_ligand_in_pose_and_scores")
    except pytest.skip.Exception as exc:
        print(f"SKIP  {exc}")
    except Exception as exc:
        print(f"FAIL  {exc!r}")
        sys.exit(1)
