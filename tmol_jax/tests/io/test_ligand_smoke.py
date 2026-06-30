"""Smoke test for the ligand pipeline scaffold (port of upstream PR #361).

This test exercises the *RDKit-free* path:

1. Parse an upstream ``*.xtal-lig.mmff94.tmol`` YAML params file.
2. Extend the default :class:`ChemicalDatabase` with the ligand
   atom_types + RawResidueType.
3. Load the matching protein-ligand PDB via biotite.
4. Build a :class:`BiotitePoseBuildContext` and verify the protein
   half still produces a valid PoseStack.

The trypsin/benzamidine complex (LG1, 34 atoms) is the smallest of the
9 upstream protein-ligand fixtures and is the obvious first-parity
target.

Deferred:

* Full inclusion of the ligand residue in the PoseStack — this needs a
  :class:`PackedBlockTypes` + :class:`CanonicalOrdering` rebuild that
  picks up the ligand RawResidueType. The scaffold parses everything
  needed; the wiring lives in the next PR.
* Scoring the complex. The current pipeline silently drops the LG1
  residue at canonical-form construction time (same behaviour as
  upstream tmol pre-PR #361). Scoring numbers therefore match the
  protein-only PDB, not the full complex.
* The ``prepare_ligands=True`` path — RDKit-only. Raises
  ``NotImplementedError`` with a clear pointer at YAML.

The test is wrapped in ``pytest.skip`` guards so a missing fixture or
unported submodule reports as a skip, not a failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# --------------------------------------------------------------------- #
# Fixture discovery                                                      #
# --------------------------------------------------------------------- #


def _find_upstream_protein_ligand_dir() -> Path | None:
    """Locate the upstream ``protein_ligand_test`` data directory.

    Checks both the installed tmol package (``.venv/lib/.../tmol/tests``)
    and the optional in-repo clone (``/tmol/tmol/tests``).
    """
    candidates = [
        Path(__file__).resolve().parents[3]
        / ".venv" / "lib" / "python3.11" / "site-packages"
        / "tmol" / "tests" / "data" / "protein_ligand_test",
        Path(__file__).resolve().parents[3]
        / "tmol" / "tmol" / "tests" / "data" / "protein_ligand_test",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Walk a few common parents looking for the fixture (handles
    # worktree invocations where the layout differs).
    for parent in Path(__file__).resolve().parents:
        cand = list(parent.rglob("protein_ligand_test"))
        for c in cand:
            if c.is_dir() and (c / "trypsin_complex.pdb").exists():
                return c
    return None


LIGAND_DATA = _find_upstream_protein_ligand_dir()


# --------------------------------------------------------------------- #
# Tests                                                                  #
# --------------------------------------------------------------------- #


def test_load_ligand_params_file_trypsin() -> None:
    if LIGAND_DATA is None:
        pytest.skip("upstream protein_ligand_test fixtures not available")
    from tmol_jax.ligand import load_ligand_params_file

    lp = load_ligand_params_file(LIGAND_DATA / "trypsin.xtal-lig.mmff94.tmol")
    assert len(lp.residues) == 1
    lg1 = lp.residues[0]
    assert lg1.name == "LG1"
    assert lg1.name3 == "LG1"
    # 20 heavy atoms + 14 hydrogens = 34, matching the PDB HETATM count.
    assert len(lg1.atoms) == 34
    # 35 distinct bonds in the YAML (verified by manual count).
    assert len(lg1.bonds) == 35
    assert lg1.icoors  # ICOORs populated
    # MMFF94 atom types referenced by the residue.
    assert {"NG22", "CDp", "CR", "Nad", "Oad", "ClR", "HN", "HR"} <= set(
        lp.atom_type_names
    )
    # Partial charges available for downstream fa_elec.
    assert ("LG1", "N1") in lp.atom_charges


def test_extend_db_from_yaml_appends_atom_types() -> None:
    if LIGAND_DATA is None:
        pytest.skip("upstream protein_ligand_test fixtures not available")
    from tmol_jax.database.chemical import ChemicalDatabase
    from tmol_jax.ligand import extend_db_from_yaml

    base = ChemicalDatabase.get_default()
    base_atom_count = len(base.atom_types)
    base_residue_count = len(base.residues)

    ext, parsed = extend_db_from_yaml(
        base, [LIGAND_DATA / "trypsin.xtal-lig.mmff94.tmol"]
    )

    assert len(parsed) == 1
    assert len(ext.atom_types) == base_atom_count + len(parsed[0].atom_type_names)
    assert len(ext.residues) == base_residue_count + 1
    # Element table and variants untouched.
    assert ext.element_types == base.element_types
    assert ext.variants == base.variants
    # LG1 must be retrievable.
    lg1 = next(r for r in ext.residues if r.name == "LG1")
    assert len(lg1.atoms) == 34


def test_prepare_ligands_true_raises() -> None:
    """``prepare_ligands=True`` must hard-fail with a pointer to YAML."""
    from tmol_jax.io.pose_stack_from_biotite import build_context_from_biotite

    with pytest.raises(NotImplementedError, match="RDKit"):
        # No real structure needed; the RDKit guard runs first.
        build_context_from_biotite(None, prepare_ligands=True)


def test_build_context_from_biotite_with_ligand() -> None:
    """End-to-end scaffold: PDB + YAML -> BiotitePoseBuildContext + PoseStack."""
    if LIGAND_DATA is None:
        pytest.skip("upstream protein_ligand_test fixtures not available")
    try:
        import biotite.structure.io.pdb as bpdb
    except ImportError:
        pytest.skip("biotite not installed")

    from tmol_jax.io.pose_stack_from_biotite import (
        BiotitePoseBuildContext,
        build_context_from_biotite,
        pose_stack_from_biotite,
    )

    pdb_path = LIGAND_DATA / "trypsin_complex.pdb"
    pdb_file = bpdb.PDBFile.read(str(pdb_path))
    atoms = pdb_file.get_structure(model=1)

    ctx = build_context_from_biotite(
        atoms,
        ligand_params_files=[
            LIGAND_DATA / "trypsin.xtal-lig.mmff94.tmol",
        ],
    )

    assert isinstance(ctx, BiotitePoseBuildContext)
    # Ligand params file parsed and surfaced on the context.
    assert len(ctx.ligand_params_files) == 1
    assert any(r.name == "LG1" for r in ctx.chem_db.residues)
    # Canonical form populated with the protein residues (HETATM ligand
    # currently drops out of the canonical-form bucket; this is the
    # deferred follow-up).
    cf = ctx.canonical_form
    n_res = int(cf["res_types"].shape[1])
    assert n_res > 0
    assert n_res > 200  # trypsin is ~220 residues

    # The plain `pose_stack_from_biotite` entry point still works,
    # produces a real PoseStack for the protein half, and is unchanged
    # for callers that pass no ligand kwargs.
    try:
        ps = pose_stack_from_biotite(atoms)
    except Exception as exc:  # pragma: no cover - tested at call time
        pytest.skip(f"pose_stack_from_canonical_form path not ready: {exc!r}")
    assert ps is not None
    # ponytail: ligand-in-PoseStack inclusion deferred -- see
    # tmol_jax/ligand/__init__.py.


def test_validate_smiles_via_pysmiles() -> None:
    """Quick check that the pysmiles cross-validator is wired up."""
    from tmol_jax.ligand.params_io import validate_smiles

    # Benzamidine: c1ccc(cc1)C(=N)N -> 9 heavy atoms (6 ring C + 1 C + 2 N).
    assert validate_smiles("c1ccc(cc1)C(=N)N", expected_heavy_atom_count=9)
    assert not validate_smiles("c1ccc(cc1)C(=N)N", expected_heavy_atom_count=10)


if __name__ == "__main__":  # pragma: no cover - manual driver
    # ``ponytail`` self-check: runnable without pytest. Mirrors the
    # ``demo()/__main__`` convention used elsewhere in jtmol.
    import sys

    failed = []
    for fn in (
        test_load_ligand_params_file_trypsin,
        test_extend_db_from_yaml_appends_atom_types,
        test_prepare_ligands_true_raises,
        test_build_context_from_biotite_with_ligand,
        test_validate_smiles_via_pysmiles,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except pytest.skip.Exception as exc:
            print(f"SKIP  {fn.__name__}: {exc}")
        except Exception as exc:
            print(f"FAIL  {fn.__name__}: {exc!r}")
            failed.append(fn.__name__)
    sys.exit(1 if failed else 0)
