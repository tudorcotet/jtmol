"""Build a scoreable ligand params file from a MOL2: the "next rung".

jtmol already ports upstream tmol's atom-type classifier
(:func:`jtmol.ligand.atom_typing.assign_tmol_atom_types`, 28/28 parity
on cdk2) and consumes pre-baked ``*.xtal-lig.mmff94.tmol`` params via
:func:`jtmol.ligand.params_io.extend_db_from_yaml`. The missing link
was *producing* that params file from a raw MOL2. This module closes it:

    MOL2  --atom_typing-->  typed mol
          --residue_builder-->  atoms + bonds + ICOOR tree + MMFF94 charges
          --write_ligand_params-->  .xtal-lig.mmff94.tmol
          --extend_db_from_yaml-->  scoreable ligand

The ICOOR-tree + geometry algorithm is a faithful port of upstream tmol's
``tmol.ligand.residue_builder`` (PR #361). The full upstream pipeline's
RDKit perception / dimorphite protonation / chi topology is still not
needed for the MOL2 path. A clean MOL2 already carries bonds, coords, and
(via RDKit) MMFF94 charges.

Requires the optional ``ligand`` extra (RDKit); imports are guarded so
``import jtmol.ligand`` works without it.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Geometry / atom-tree helpers (port of upstream residue_builder)     #
# ------------------------------------------------------------------ #

_MAX_RING_SIZE_TREATED_AS_RING = 8
_AMIDE_N_TYPES = {"Nad", "Nad3"}
_GUANIDINIUM_N_TYPES = {"Ngu1", "Ngu2"}
_PLANAR_N_TYPES = _AMIDE_N_TYPES | _GUANIDINIUM_N_TYPES


def _mol_coords(mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    if mol.GetNumConformers() == 0:
        return np.zeros((n, 3))
    conf = mol.GetConformer(0)
    coords = np.zeros((n, 3))
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i] = (p.x, p.y, p.z)
    return coords


def _find_nbr_atom(mol, coords: np.ndarray) -> int:
    """Root atom: heavy, degree>=2, closest to centre of mass."""
    com = coords.mean(axis=0)
    dists_sq = np.sum((coords - com) ** 2, axis=1)
    best_idx, best_dist = -1, float("inf")
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 1 or atom.GetDegree() < 2:
            continue
        if dists_sq[idx] < best_dist:
            best_dist, best_idx = dists_sq[idx], idx
    if best_idx < 0:  # fallback: first heavy atom
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                best_idx = atom.GetIdx()
                break
    if best_idx < 0:
        raise ValueError("no heavy atom to root the ICOOR tree")
    return best_idx


def _build_atom_tree(mol, root_idx: int):
    """BFS atom tree -> (order, parent, grandparents)."""
    n_atoms = mol.GetNumAtoms()
    adj: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)

    visited = [False] * n_atoms
    parent = {root_idx: root_idx}
    order: list[int] = []
    queue = deque([root_idx])
    visited[root_idx] = True
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nbr in adj[cur]:
            if not visited[nbr]:
                visited[nbr] = True
                parent[nbr] = cur
                queue.append(nbr)

    bfs_pos = {idx: i for i, idx in enumerate(order)}

    def _is_heavy(i: int) -> bool:
        return mol.GetAtomWithIdx(i).GetAtomicNum() != 1

    def _placed(of: int, exclude: set[int], before: int) -> Optional[int]:
        """Neighbor of ``of`` already placed (bfs pos < ``before``), heavy first.

        Ancestors MUST precede the atom in build order for the ICOOR tree to
        reconstruct (``build_coords_from_icoors`` walks 0..n-1). This is the
        constraint the naive neighbor pick missed, producing forward refs.
        """
        cands = [n for n in adj[of]
                 if n not in exclude and bfs_pos.get(n, len(order)) < before]
        if not cands:
            return None
        cands.sort(key=lambda n: (not _is_heavy(n), bfs_pos[n]))
        return cands[0]

    def _any_earlier(exclude: set[int], before: int) -> Optional[int]:
        """Earliest already-placed atom not in ``exclude`` (stub reference).

        Reference atoms need not be bonded, only earlier-placed and distinct,
        so this guarantees a non-duplicate ancestor when no placed neighbor
        exists (e.g. the parent's only earlier bond is the grandparent).
        """
        for at in order[:before]:
            if at not in exclude:
                return at
        return None

    grandparents: dict[int, tuple[int, int]] = {}
    for idx in order:
        p = bfs_pos[idx]
        par = parent[idx]
        # grandparent: tree parent-of-parent when it is a distinct earlier atom
        # (BFS guarantees it precedes), else an already-placed sibling of
        # parent, else any earlier-placed atom.
        gp = parent.get(par, par)
        if gp == par or bfs_pos.get(gp, len(order)) >= p:
            cand = _placed(par, {idx, par}, p)
            if cand is None:
                cand = _any_earlier({idx, par}, p)
            gp = par if cand is None else cand  # atom idx 0 is falsy: use `is None`
        # great-grandparent: tree parent-of-gp, else placed neighbor of gp/par,
        # else any earlier-placed atom distinct from {idx, par, gp}.
        ggp = parent.get(gp, gp)
        if ggp in (gp, par, idx) or bfs_pos.get(ggp, len(order)) >= p:
            cand = _placed(gp, {idx, par, gp}, p)
            if cand is None:
                cand = _placed(par, {idx, par, gp}, p)
            if cand is None:
                cand = _any_earlier({idx, par, gp}, p)
            ggp = gp if cand is None else cand
        grandparents[idx] = (gp, ggp)
    return order, parent, grandparents


def _distance(a, b) -> float:
    return float(np.linalg.norm(a - b))


def _angle(a, b, c) -> float:
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _builder_phi(child, par, gp, ggp) -> float:
    """Dihedral ``phi`` in the convention ``build_coords_from_icoors`` inverts.

    The builder frames each atom from ``(ggp, gp, par)`` and applies
    ``rot_z(phi)``, so the exact inverse is
    ``atan2(-(child-par).x_hat, (child-par).y_hat)`` in that frame, NOT the
    standard 4-atom dihedral (different reference axis), which rebuilds ~0.5 A
    worse. Derived by inverting the builder's frame math; verified by
    round-trip (see tests/io/test_residue_builder.py).
    """
    from jtmol.chemical.ideal_coords import frame_from_coords

    v = child - par
    frame = frame_from_coords(ggp, gp, par)
    return float(np.degrees(np.arctan2(
        -float(np.dot(v, frame[0:3, 0])), float(np.dot(v, frame[0:3, 1])))))


def _icoor_rows(order, parent, grandparents, atom_names, coords) -> list[dict]:
    """Per-atom ICOOR dicts (phi/theta in degrees, the .tmol file convention)."""
    rows = []
    for i, idx in enumerate(order):
        par = parent[idx]
        gp, ggp = grandparents[idx]
        if i == 0:
            d, theta, phi = 0.0, 0.0, 0.0
        elif i == 1:
            d, theta, phi = _distance(coords[idx], coords[par]), 180.0, 0.0
        elif i == 2:
            d = _distance(coords[idx], coords[par])
            theta = 180.0 - _angle(coords[idx], coords[par], coords[gp])
            phi = 0.0
        else:
            d = _distance(coords[idx], coords[par])
            theta = 180.0 - _angle(coords[idx], coords[par], coords[gp])
            phi = _builder_phi(coords[idx], coords[par], coords[gp], coords[ggp])
        rows.append({
            "name": atom_names[idx],
            "phi": f"{phi:.6f} deg",
            "theta": f"{theta:.6f} deg",
            "d": round(d, 6),
            "parent": atom_names[par],
            "grand_parent": atom_names[gp],
            "great_grand_parent": atom_names[ggp],
        })
    return rows


def _is_in_large_ring(mol, bond) -> bool:
    if not bond.IsInRing():
        return False
    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    for ring in mol.GetRingInfo().AtomRings():
        if a in ring and b in ring and len(ring) <= _MAX_RING_SIZE_TREATED_AS_RING:
            return False
    return True


def _is_planar_resonance_bond(atom_a, atom_b, type_a: str, type_b: str) -> bool:
    from rdkit import Chem

    if type_a in _PLANAR_N_TYPES and atom_b.GetAtomicNum() == 6:
        n_atom, c_atom, n_type = atom_a, atom_b, type_a
    elif type_b in _PLANAR_N_TYPES and atom_a.GetAtomicNum() == 6:
        n_atom, c_atom, n_type = atom_b, atom_a, type_b
    else:
        return False
    if n_type in _AMIDE_N_TYPES:
        return any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(c_atom).GetAtomicNum() == 8
            for b in c_atom.GetBonds()
            if b.GetOtherAtom(c_atom).GetIdx() != n_atom.GetIdx()
        )
    others = (nbr for nbr in c_atom.GetNeighbors() if nbr.GetIdx() != n_atom.GetIdx())
    return sum(1 for nbr in others if nbr.GetAtomicNum() == 7) >= 2


def _mmff94_charges_from_mol2(mol2_path, elements) -> Optional[list[float]]:
    """MMFF94 charges from a *fresh* RDKit MOL2 read, index-aligned.

    The classifier's mol carries an aromaticity state RDKit can't kekulise
    for MMFF; a fresh ``MolFromMol2File`` sanitises cleanly and (MOL2 atom
    order being explicit) aligns index-for-index. Returns None if the fresh
    read doesn't match ``elements`` (then the caller falls back to zeros).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
    if m is None or m.GetNumAtoms() != len(elements):
        return None
    if [m.GetAtomWithIdx(i).GetSymbol() for i in range(m.GetNumAtoms())] != list(elements):
        return None
    props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94")
    if props is None:
        return None
    return [round(float(props.GetMMFFPartialCharge(i)), 6)
            for i in range(m.GetNumAtoms())]


def _mmff94_charges_from_mol(mol) -> Optional[list[float]]:
    """MMFF94 charges straight off a sanitised RDKit mol (SMILES/SDF path).

    Unlike the MOL2 path, a SMILES/SDF-derived mol is already kekulisable, so
    MMFF runs on it directly. Returns None if MMFF has no params for it or
    can't kekulise it (caller falls back to zero charges). NOTE: run this
    *before* the atom typer, which mutates the mol's aromaticity in place and
    can then break MMFF kekulisation.
    """
    from rdkit.Chem import AllChem

    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    except Exception:  # e.g. KekulizeException on some ring systems
        return None
    if props is None:
        return None
    return [round(float(props.GetMMFFPartialCharge(i)), 6)
            for i in range(mol.GetNumAtoms())]


# ------------------------------------------------------------------ #
# Public: MOL2 -> params dict / file                                  #
# ------------------------------------------------------------------ #


def build_ligand_params_dict(mol, assignments, res_name: str,
                             charges: Optional[list[float]] = None,
                             keep_input_names: bool = False) -> dict:
    """Assemble the ``.xtal-lig.mmff94.tmol`` dict from a typed RDKit mol.

    ``assignments`` is the list of
    :class:`jtmol.ligand.atom_typing.AtomTypeAssignment` from
    ``assign_tmol_atom_types(mol)`` (atom_name / atom_type / element /
    index). ``charges`` (per mol atom index) is optional: MMFF94 from the
    MOL2 path; a list of zeros if omitted.

    ``keep_input_names``: by default atoms take the typer's canonical names
    (heavy ``N1/C1/...``, hydrogens renumbered ``H1..HN``) so the params
    match ordinary ligand-PDB atom names. Set True to instead keep the input
    MOL2's own atom names verbatim. Use this when you will score a structure
    that carries those exact names.
    """
    from rdkit import Chem

    n = mol.GetNumAtoms()
    names: list = [None] * n
    types: list = [None] * n
    for a in assignments:  # index by mol atom idx, not list position
        names[a.index] = a.atom_name
        types[a.index] = a.atom_type
    if keep_input_names:
        for i in range(n):
            at = mol.GetAtomWithIdx(i)
            if at.HasProp("_TriposAtomName"):
                names[i] = at.GetProp("_TriposAtomName")
    else:
        # Hydrogens: the typer names them by bonded element (HC1/HN1/...);
        # rewrite to the standard sequential H1..HN (atom order) so the params
        # match ordinary ligand-PDB atom names at load. The atom TYPE carries
        # the chemistry, so the name is cosmetic (verified: score-identical).
        h_k = 0
        for i in range(n):
            if types[i] and types[i].startswith("H"):
                h_k += 1
                names[i] = f"H{h_k}"
    type_by_name = {names[i]: types[i] for i in range(n)}
    atoms = [{"name": names[i], "atom_type": types[i]} for i in range(n)]

    bonds = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        na, nb = names[a], names[b]
        bt = bond.GetBondType()
        b_type = {
            Chem.BondType.SINGLE: "SINGLE",
            Chem.BondType.DOUBLE: "DOUBLE",
            Chem.BondType.TRIPLE: "TRIPLE",
            Chem.BondType.AROMATIC: "AROMATIC",
        }.get(bt, "AROMATIC" if bond.GetIsAromatic() else "SINGLE")
        if b_type == "SINGLE" and _is_planar_resonance_bond(
            mol.GetAtomWithIdx(a), mol.GetAtomWithIdx(b),
            type_by_name.get(na, ""), type_by_name.get(nb, ""),
        ):
            b_type = "AROMATIC"
        in_ring = bond.IsInRing() and not _is_in_large_ring(mol, bond)
        bonds.append([na, nb, b_type, bool(in_ring)])

    if mol.GetNumConformers() == 0:
        raise ValueError(
            "mol has no conformer; 3D coords are required to build the ICOOR "
            "tree (the MOL2 path always carries coords)")
    coords = _mol_coords(mol)
    order, parent, grandparents = _build_atom_tree(mol, _find_nbr_atom(mol, coords))
    if len(order) != n:
        raise ValueError(
            f"ligand is disconnected: the atom tree reached {len(order)}/{n} "
            "atoms. Split multi-fragment inputs (e.g. salts) into one residue "
            "each before building params.")
    icoors = _icoor_rows(order, parent, grandparents, names, coords)

    if charges is None:
        charges = [0.0] * n
    charge_rows = [{"res": res_name, "atom": names[i], "charge": charges[i]}
                   for i in range(n)]

    return {
        "version": "1.0",
        "chemical": {"residues": [{
            "name": res_name, "base_name": res_name, "name3": res_name,
            "io_equiv_class": res_name, "atoms": atoms, "atom_aliases": [],
            "bonds": bonds, "icoors": icoors,
        }]},
        "elec": {"atom_charge_parameters": charge_rows},
    }


def mol2_to_params_dict(mol2_path: str | Path, res_name: str = "LG1",
                        keep_input_names: bool = False) -> dict:
    """Read a MOL2, type it, and build the full params dict.

    ``keep_input_names`` keeps the MOL2's own atom names (see
    :func:`build_ligand_params_dict`).
    """
    from jtmol.ligand.atom_typing import (
        assign_tmol_atom_types,
        read_mol2_for_typing,
    )

    mol = read_mol2_for_typing(mol2_path)
    assignments = assign_tmol_atom_types(mol)
    elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    charges = _mmff94_charges_from_mol2(mol2_path, elements)
    if charges is None:
        logger.warning(
            "MMFF94 charges unavailable for %s (fresh RDKit read failed to "
            "sanitise/align); writing zero charges -> ligand fa_elec will be "
            "zero.", mol2_path)
    return build_ligand_params_dict(mol, assignments, res_name, charges=charges,
                                    keep_input_names=keep_input_names)


def mol_to_params_dict(mol, res_name: str = "LG1",
                       keep_input_names: bool = False) -> dict:
    """Build the params dict from any sanitised RDKit mol with a 3D conformer.

    The generic path behind :func:`smiles_to_params` (and usable for SDF /
    a mol you built yourself). The mol needs explicit Hs and one conformer;
    MMFF94 charges are read straight off it. For MOL2 files prefer
    :func:`mol2_to_params_dict` (it round-trips charges through a fresh read).
    """
    from rdkit import Chem

    from jtmol.ligand.atom_typing import assign_tmol_atom_types

    # Charge the mol BEFORE typing: assign_tmol_atom_types mutates the mol's
    # aromaticity in place, which can then break MMFF kekulisation. A pristine
    # copy keeps the same atom order, so charges stay index-aligned.
    charges = _mmff94_charges_from_mol(Chem.Mol(mol))
    assignments = assign_tmol_atom_types(mol)
    if charges is None:
        logger.warning(
            "MMFF94 charges unavailable for the mol; writing zero charges -> "
            "ligand fa_elec will be zero.")
    return build_ligand_params_dict(mol, assignments, res_name, charges=charges,
                                    keep_input_names=keep_input_names)


def smiles_to_params(smiles: str, out_path: str | Path, res_name: str = "LG1",
                     seed: int = 1, optimize: bool = True) -> Path:
    """SMILES -> written scoreable ``.tmol``.

    Parses the SMILES, adds explicit Hs, embeds a 3D conformer (ETKDG) and
    (by default) MMFF-optimises it, then runs the mol -> params pipeline.
    Atom names are the typer's canonical scheme. Raises on parse / embed
    failure. ``seed`` fixes the embedding for reproducibility.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError(f"3D embedding failed for SMILES: {smiles!r}")
    if optimize:
        AllChem.MMFFOptimizeMolecule(mol)
    return write_ligand_params(mol_to_params_dict(mol, res_name), out_path)


def write_ligand_params(params: dict, out_path: str | Path) -> Path:
    """Serialise a params dict to a ``.xtal-lig.mmff94.tmol`` YAML file."""
    import yaml

    out = Path(out_path)
    with out.open("w") as f:
        yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)
    return out


def mol2_to_params(mol2_path: str | Path, out_path: str | Path,
                   res_name: str = "LG1", keep_input_names: bool = False) -> Path:
    """MOL2 -> written ``.xtal-lig.mmff94.tmol`` (the full path).

    The result is directly consumable by
    :func:`jtmol.ligand.params_io.extend_db_from_yaml` and hence
    scoreable via ``build_context_from_biotite(ligand_params_files=[...])``.
    ``keep_input_names`` keeps the MOL2's own atom names (see
    :func:`build_ligand_params_dict`).
    """
    return write_ligand_params(
        mol2_to_params_dict(mol2_path, res_name, keep_input_names=keep_input_names),
        out_path)


__all__ = [
    "build_ligand_params_dict",
    "mol2_to_params_dict",
    "mol_to_params_dict",
    "smiles_to_params",
    "write_ligand_params",
    "mol2_to_params",
]
