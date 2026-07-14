"""Ligand pipeline scaffold for jtmol.

This is the **JAX-compatible, RDKit-free** scaffold of upstream tmol's
``tmol.ligand`` package (PR #361 ``Kdidi/ligand-clean``, hash 230fe74).
Upstream lives at ~6000 LOC across 13 files and leans on RDKit for
SMILES/MOL2/CIF perception, SMARTS atom typing, dimorphite pKa, and
RDKit-driven ICOOR construction. None of that has a JAX-friendly
substitute, so this port covers only the path that **does not** need
RDKit at runtime: pre-baked YAML params files (the ``*.xtal-lig.mmff94.tmol``
fixtures upstream ships).

What is in this package right now:

* :mod:`jtmol.ligand.params_file`: frozen-attrs parsers for the
  ``.xtal-lig.mmff94.tmol`` YAML format. Pure-Python, no torch / no jax /
  no RDKit.
* :mod:`jtmol.ligand.params_io`: :func:`extend_db_from_yaml` extends
  a :class:`ChemicalDatabase` with the atom_types + RawResidueType
  parsed out of one or more YAML params files. The result is the seam
  that :func:`jtmol.io.pose_stack_from_biotite.build_context_from_biotite`
  consumes when called with ``ligand_params_files=[...]``.

* :mod:`jtmol.ligand.atom_typing`: a faithful RDKit port of upstream's
  ``assign_tmol_atom_types`` (the ~1568-line Rosetta generic-potential
  classifier). Reads a MOL2 and assigns generic-potential atom types
  (``CR``, ``Nim``, ``ClR``, …). Requires the optional ``ligand`` extra
  (RDKit); the import is guarded so ``import jtmol.ligand`` still works
  without RDKit installed.

* :mod:`jtmol.ligand.residue_builder`: MOL2 → scoreable params. Ports
  upstream's ICOOR-tree builder (BFS atom tree + internal coordinates) and
  serialises atoms + bonds + ICOORs + MMFF94 charges to a
  ``*.xtal-lig.mmff94.tmol`` file. :func:`mol2_to_params` is the one-call
  entry (MOL2 in, params file out). Verified score-identical to the shipped
  cdk2 params on the real protein-ligand complex. Requires the ``ligand``
  extra (RDKit), guarded so the base import still works without it.

What is **deferred** to a follow-up:

* The full SMILES/CIF perception path (xyz2mol bond perception, dimorphite
  protonation, chi-topology sampling). The MOL2 path above needs none of
  it: a clean MOL2 already carries bonds, coords, and (via RDKit) MMFF94
  charges.
"""

from __future__ import annotations

from jtmol.ligand.params_file import LigandParamsFile, load_ligand_params_file
from jtmol.ligand.params_io import extend_db_from_yaml

# residue_builder imports RDKit lazily (inside functions), so it is safe to
# import here without the `ligand` extra; only *calling* it needs RDKit.
from jtmol.ligand.residue_builder import (
    build_ligand_params_dict,
    mol2_to_params,
    mol2_to_params_dict,
    mol_to_params_dict,
    smiles_to_params,
    write_ligand_params,
)

__all__ = [
    "LigandParamsFile",
    "load_ligand_params_file",
    "extend_db_from_yaml",
    "mol2_to_params",
    "mol2_to_params_dict",
    "mol_to_params_dict",
    "smiles_to_params",
    "build_ligand_params_dict",
    "write_ligand_params",
]

# Atom typing needs RDKit (optional `ligand` extra). Guard the import so
# the base package stays importable without it; the names appear in
# __all__ only when RDKit is present.
try:
    from jtmol.ligand.atom_typing import (  # noqa: F401
        AtomTypeAssignment,
        assign_atom_types_from_mol2,
        assign_tmol_atom_types,
    )

    __all__ += [
        "AtomTypeAssignment",
        "assign_atom_types_from_mol2",
        "assign_tmol_atom_types",
    ]
except ImportError:  # rdkit not installed
    pass
