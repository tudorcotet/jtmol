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

* :mod:`tmol_jax.ligand.params_file` — frozen-attrs parsers for the
  ``.xtal-lig.mmff94.tmol`` YAML format. Pure-Python, no torch / no jax /
  no RDKit.
* :mod:`tmol_jax.ligand.params_io` — :func:`extend_db_from_yaml` extends
  a :class:`ChemicalDatabase` with the atom_types + RawResidueType
  parsed out of one or more YAML params files. The result is the seam
  that :func:`tmol_jax.io.pose_stack_from_biotite.build_context_from_biotite`
  consumes when called with ``ligand_params_files=[...]``.

* :mod:`tmol_jax.ligand.atom_typing` — a faithful RDKit port of upstream's
  ``assign_tmol_atom_types`` (the ~1568-line Rosetta generic-potential
  classifier). Reads a MOL2 and assigns generic-potential atom types
  (``CR``, ``Nim``, ``ClR``, …). Requires the optional ``ligand`` extra
  (RDKit); the import is guarded so ``import tmol_jax.ligand`` still works
  without RDKit installed.

What is **deferred** to a follow-up:

* ``prepare_ligands=True`` (the full RDKit MOL2/CIF/SMILES → params
  orchestration: residue-builder ICOOR tree + params serialization).
  The atom-typing core is ported and parity-verified (28/28 on cdk2);
  the remaining pieces (``residue_builder``, ``params_io`` writer) are
  the next rung.
"""

from __future__ import annotations

from tmol_jax.ligand.params_file import LigandParamsFile, load_ligand_params_file
from tmol_jax.ligand.params_io import extend_db_from_yaml

__all__ = [
    "LigandParamsFile",
    "load_ligand_params_file",
    "extend_db_from_yaml",
]

# Atom typing needs RDKit (optional `ligand` extra). Guard the import so
# the base package stays importable without it; the names appear in
# __all__ only when RDKit is present.
try:
    from tmol_jax.ligand.atom_typing import (  # noqa: F401
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
