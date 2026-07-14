"""MOL2 -> scoreable ligand params (``jtmol.ligand.residue_builder``).

The atom-type classifier + params reader already let jtmol *score* a
pre-baked ``.tmol``; ``residue_builder`` closes the loop by *producing* that
file from a raw MOL2 (atoms + bonds + ICOOR tree + MMFF94 charges). This
certifies the generated params reproduce the shipped ground-truth
``cdk2.xtal-lig.mmff94.tmol`` both structurally and, the real test, in the
score of the actual cdk2 protein-ligand complex.

Requires the ``ligand`` extra (RDKit) + biotite; skips otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_DATA = Path(__file__).resolve().parents[1] / "data" / "protein_ligand_test"
_MOL2 = _DATA / "cdk2.lig.mol2"
_SHIPPED = _DATA / "cdk2.xtal-lig.mmff94.tmol"
_PDB = _DATA / "cdk2_complex.pdb"


def _build_mine(tmp_path) -> tuple[dict, Path]:
    pytest.importorskip("rdkit")
    yaml = pytest.importorskip("yaml")
    from jtmol.ligand.residue_builder import mol2_to_params, mol2_to_params_dict

    if not _MOL2.exists() or not _SHIPPED.exists():  # pragma: no cover
        pytest.skip("cdk2 ligand fixtures not available")
    d = mol2_to_params_dict(_MOL2, "LG1")
    out = Path(tmp_path) / "cdk2_mine.tmol"
    mol2_to_params(_MOL2, out, "LG1")
    return d, out, yaml


def test_generated_params_match_shipped(tmp_path) -> None:
    mine, _out, yaml = _build_mine(tmp_path)
    shipped = yaml.safe_load(_SHIPPED.read_text())
    mr = mine["chemical"]["residues"][0]
    sr = shipped["chemical"]["residues"][0]

    # atoms: name + type, in order (H now sequential like the shipped file).
    ma = [(a["name"], a["atom_type"]) for a in mr["atoms"]]
    sa = [(a["name"], a["atom_type"]) for a in sr["atoms"]]
    assert ma == sa, f"atom name/type mismatch\n mine={ma}\n ship={sa}"

    # bonds: same pairs (topology). Order can differ on resonance bonds
    # without affecting connectivity-based scoring (asserted below).
    mb = {frozenset([b[0], b[1]]) for b in mr["bonds"]}
    sb = {frozenset([b[0], b[1]]) for b in sr["bonds"]}
    assert mb == sb, f"bond topology mismatch: {mb ^ sb}"

    # charges: MMFF94, per atom name.
    mc = {c["atom"]: c["charge"] for c in mine["elec"]["atom_charge_parameters"]}
    sc = {c["atom"]: c["charge"] for c in shipped["elec"]["atom_charge_parameters"]}
    assert set(mc) == set(sc)
    assert max(abs(mc[a] - sc[a]) for a in mc) < 1e-3


def test_generated_params_score_matches_shipped(tmp_path) -> None:
    _mine, out, _yaml = _build_mine(tmp_path)
    bpdb = pytest.importorskip("biotite.structure.io.pdb")
    if not _PDB.exists():  # pragma: no cover
        pytest.skip("cdk2 complex PDB not available")

    import jtmol
    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    atoms = bpdb.PDBFile.read(str(_PDB)).get_structure(model=1)

    def score(tmol: Path) -> dict:
        ps = pose_stack_from_biotite(atoms, ligand_params_files=[tmol])
        sfxn = jtmol.beta2016_score_function()
        st = sfxn.setup_for_pose_stack(ps)
        per = {s.name: float(np.asarray(v).sum())
               for s, v in sfxn.score_by_term(ps, state=st).items()}
        per["TOTAL"] = float(np.asarray(sfxn(ps)).sum())
        return per

    mine, ship = score(out), score(_SHIPPED)
    worst = max(abs(mine[k] - ship[k]) for k in ship)
    # Only fa_elec/TOTAL move, by ~2e-4 (6th-decimal charge rounding); every
    # structural / LJ / LK / hbond term is byte-identical.
    assert worst < 1e-2, {k: (mine[k], ship[k]) for k in ship
                          if abs(mine[k] - ship[k]) > 1e-3}


def test_disconnected_mol_rejected() -> None:
    """A multi-fragment mol (e.g. a salt) must error, not emit a broken file.

    The ICOOR tree is a single BFS from one root; a disconnected mol would
    leave the other fragment's atoms without icoors. Guard rejects it.
    """
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    from jtmol.ligand.atom_typing import assign_tmol_atom_types
    from jtmol.ligand.residue_builder import build_ligand_params_dict

    m = Chem.AddHs(Chem.MolFromSmiles("C.C"))  # two disconnected methanes
    AllChem.EmbedMolecule(m, randomSeed=1)
    try:
        assignments = assign_tmol_atom_types(m)
    except Exception:  # typer may reject the toy mol first, that's fine
        pytest.skip("typer rejected toy mol before builder guard")
    with pytest.raises(ValueError, match="disconnected"):
        build_ligand_params_dict(m, assignments, "LIG")


def _rebuild_rmsd(mol):
    """Round-trip: build the ICOOR tree, feed it to jtmol's own coordinate
    builder, and Kabsch-superimpose the result onto the input. Returns
    ``(rmsd, covers_all, finite)``."""
    import math

    from jtmol.chemical.ideal_coords import build_coords_from_icoors
    from jtmol.ligand.atom_typing import assign_tmol_atom_types
    from jtmol.ligand.residue_builder import (
        _build_atom_tree,
        _find_nbr_atom,
        _icoor_rows,
        _mol_coords,
    )

    names = [a.atom_name for a in assign_tmol_atom_types(mol)]
    coords = _mol_coords(mol)
    order, parent, gps = _build_atom_tree(mol, _find_nbr_atom(mol, coords))
    covers = len(order) == mol.GetNumAtoms()
    rows = _icoor_rows(order, parent, gps, names, coords)
    pos = {idx: k for k, idx in enumerate(order)}
    n = len(order)
    anc = np.zeros((n, 3), np.int32)
    geom = np.zeros((n, 3))
    for k, idx in enumerate(order):
        gp, ggp = gps[idx]
        anc[k] = (pos[parent[idx]], pos[gp], pos[ggp])
        geom[k] = (math.radians(float(rows[k]["phi"].split()[0])),
                   math.radians(float(rows[k]["theta"].split()[0])),
                   rows[k]["d"])
    # forward refs / degeneracies would make the walk-0..n-1 rebuild invalid.
    assert all((anc[i] < i).all() for i in range(3, n)), "forward ancestor ref"
    assert all(len(set(anc[i])) == 3 for i in range(3, n)), "degenerate ancestor"
    rb = build_coords_from_icoors(anc, geom)
    finite = bool(np.isfinite(rb).all())
    truth = np.array([coords[order[k]] for k in range(n)])
    P, Q = rb - rb.mean(0), truth - truth.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    rmsd = float(np.sqrt(((P @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T - Q) ** 2)
                         .sum(1).mean()))
    return rmsd, covers, finite


def test_icoor_tree_rebuilds_cdk2() -> None:
    """The generated ICOOR tree reconstructs geometry through jtmol's builder.

    Guards the real bug that scoring-with-input-coords can't catch: forward /
    degenerate ancestor refs make ``build_coords_from_icoors`` emit NaN. Also
    checks the rebuild is at least as faithful as the shipped ground truth
    (~2.1 A; ours ~1.9 A: the residual is the builder's fixed 3-atom stub).
    """
    pytest.importorskip("rdkit")
    from jtmol.ligand.atom_typing import read_mol2_for_typing

    if not _MOL2.exists():  # pragma: no cover
        pytest.skip("cdk2 ligand fixture not available")
    rmsd, covers, finite = _rebuild_rmsd(read_mol2_for_typing(_MOL2))
    assert covers and finite
    assert rmsd < 2.15, f"rebuild RMSD {rmsd:.3f} worse than shipped ground truth"


def test_breadth_diverse_ligands() -> None:
    """The atom-tree + ICOOR pipeline is robust beyond cdk2: fused rings, a
    large (12-membered) ring, and a resonance/guanidinium group all cover all
    atoms and rebuild finite."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    for smi in ("c1ccc2ccccc2c1",           # naphthalene (fused aromatic)
                "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine (fused N-heterocycle)
                "C1CCCCCCCCCCC1",           # 12-membered ring (large-ring path)
                "NC(=N)N"):                 # guanidinium (resonance)
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, randomSeed=7) != 0:  # pragma: no cover
            pytest.skip(f"embed failed for {smi}")
        rmsd, covers, finite = _rebuild_rmsd(m)
        assert covers, f"{smi}: atom tree missed atoms"
        assert finite, f"{smi}: rebuild non-finite"
        assert rmsd < 3.0, f"{smi}: rebuild RMSD {rmsd:.2f} unreasonable"


def test_keep_input_names_preserves_mol2_names() -> None:
    """``keep_input_names=True`` keeps the MOL2's own atom names (C9/N9 here),
    vs the default canonical scheme (C1/H1)."""
    pytest.importorskip("rdkit")
    from jtmol.ligand.residue_builder import mol2_to_params_dict

    if not _MOL2.exists():  # pragma: no cover
        pytest.skip("cdk2 ligand fixture not available")
    kept = [a["name"] for a in mol2_to_params_dict(_MOL2, "LG1", keep_input_names=True)
            ["chemical"]["residues"][0]["atoms"]]
    canon = [a["name"] for a in mol2_to_params_dict(_MOL2, "LG1")
             ["chemical"]["residues"][0]["atoms"]]
    assert "C9" in kept and "N9" in kept and "C1" not in kept
    assert "C1" in canon and "H1" in canon


def test_smiles_to_params_scoreable(tmp_path) -> None:
    """SMILES -> scoreable params: typed atoms, real MMFF charges, and
    rebuild-valid icoors, loaded through the real extend_db_from_yaml seam.

    benzamidine is the regression case: its charged amidine trips MMFF
    kekulisation *after* the typer mutates the mol, so charges must be read
    before typing (else fa_elec would silently zero the key charged group).
    """
    pytest.importorskip("rdkit")
    yaml = pytest.importorskip("yaml")
    from jtmol.chemical.ideal_coords import build_coords_from_icoors
    from jtmol.chemical.restypes import _refined_from_raw
    from jtmol.database import ParameterDatabase
    from jtmol.ligand import extend_db_from_yaml
    from jtmol.ligand.residue_builder import smiles_to_params

    cases = [("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", 24),
             ("benzamidine", "NC(=N)c1ccccc1", 17)]
    for name, smi, n_atoms in cases:
        out = Path(tmp_path) / f"{name}.tmol"
        smiles_to_params(smi, out, "LIG")
        chem = ParameterDatabase.get_default().chemical
        ext, _ = extend_db_from_yaml(chem, [out])
        rt = {r.name: r for r in ext.residues}["LIG"]
        assert len(rt.atoms) == n_atoms, f"{name}: {len(rt.atoms)} != {n_atoms}"
        doc = yaml.safe_load(out.read_text())
        nz = sum(1 for c in doc["elec"]["atom_charge_parameters"]
                 if abs(c["charge"]) > 1e-9)
        assert nz > 0, f"{name}: no MMFF charges (fa_elec would be zero)"
        ref = _refined_from_raw(rt)
        rb = build_coords_from_icoors(np.asarray(ref.icoors_ancestors),
                                      np.asarray(ref.icoors_geom))
        assert np.isfinite(rb).all(), f"{name}: icoor rebuild non-finite"


def test_smiles_bad_input_raises(tmp_path) -> None:
    pytest.importorskip("rdkit")
    from jtmol.ligand.residue_builder import smiles_to_params

    with pytest.raises(ValueError, match="parse SMILES"):
        smiles_to_params("this is not a smiles", Path(tmp_path) / "x.tmol")


if __name__ == "__main__":  # pragma: no cover - manual driver
    import inspect
    import sys
    import tempfile

    failed = []
    for fn in (test_generated_params_match_shipped,
               test_generated_params_score_matches_shipped,
               test_disconnected_mol_rejected,
               test_icoor_tree_rebuilds_cdk2,
               test_breadth_diverse_ligands,
               test_keep_input_names_preserves_mol2_names,
               test_smiles_to_params_scoreable,
               test_smiles_bad_input_raises):
        try:
            with tempfile.TemporaryDirectory() as td:
                fn(td) if inspect.signature(fn).parameters else fn()
            print(f"PASS  {fn.__name__}")
        except pytest.skip.Exception as exc:
            print(f"SKIP  {fn.__name__}: {exc}")
        except Exception as exc:
            print(f"FAIL  {fn.__name__}: {exc!r}")
            failed.append(fn.__name__)
    sys.exit(1 if failed else 0)
