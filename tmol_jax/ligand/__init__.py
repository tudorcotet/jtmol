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

What is **deferred** to a follow-up:

* ``prepare_ligands=True`` (the RDKit SMILES/MOL2/CIF → params pipeline).
  Calling :func:`build_context_from_biotite` with that flag raises
  ``NotImplementedError`` and points the caller at the YAML route.
* Full :class:`PackedBlockTypes` + :class:`CanonicalOrdering` rebuild
  for the ligand restypes. The current scaffold parses the ligand
  YAML and surfaces the result on the build context, but the protein
  PoseStack the smoke test exercises does not yet *contain* the ligand
  as a residue. Hooking the ligand restype into the canonical-form
  pipeline (atom-name ↔ canonical-slot mapping, ideal coords from the
  YAML's ``icoors``, palette extension) is the next rung.
"""

from __future__ import annotations

from tmol_jax.ligand.params_file import LigandParamsFile, load_ligand_params_file
from tmol_jax.ligand.params_io import extend_db_from_yaml

__all__ = [
    "LigandParamsFile",
    "load_ligand_params_file",
    "extend_db_from_yaml",
]
