"""``jtmol.io`` -- PDB parsing and canonical-form construction.

Pure-Python / numpy port of ``tmol.io``. All public names are exposed
lazily through ``__getattr__`` so a transient bug in one submodule
does not prevent ``import jtmol.io`` from succeeding.

Public names
------------
- :func:`parse_pdb` -- parse PDB text/file into atom records.
- :func:`pose_stack_from_pdb` -- top-level PDB -> PoseStack helper.
- :class:`CanonicalOrdering` -- per-residue canonical atom ordering.
- :func:`default_canonical_ordering` -- 20-AA default ordering.
- :func:`default_packed_block_types` -- 20-AA default PackedBlockTypes.
- :func:`canonical_form_from_pdb` -- PDB text/file -> canonical-form dict.
- :func:`pose_stack_from_canonical_form` -- canonical-form dict -> PoseStack.
- :func:`canonical_form_from_pose_stack` -- PoseStack -> canonical-form dict.
- :func:`atom_records_from_pose_stack` -- list of per-atom dicts.
- :func:`write_pose_stack_pdb` -- serialise a PoseStack to PDB.
- :func:`deduce_chains` -- chain-break identification via graph components.
"""

from __future__ import annotations

import importlib


_EXPORTS = {
    "parse_pdb": "jtmol.io.pdb_parsing",
    "pose_stack_from_pdb": "jtmol.io.pdb_parsing",
    "CanonicalOrdering": "jtmol.io.canonical_ordering",
    "default_canonical_ordering": "jtmol.io.canonical_ordering",
    "default_packed_block_types": "jtmol.io.canonical_ordering",
    "canonical_form_from_pdb": "jtmol.io.canonical_form",
    "pose_stack_from_canonical_form": "jtmol.io.pose_stack_construction",
    "canonical_form_from_pose_stack": "jtmol.io.pose_stack_deconstruction",
    "atom_records_from_pose_stack": "jtmol.io.write_pose_stack_pdb",
    "write_pose_stack_pdb": "jtmol.io.write_pose_stack_pdb",
    "deduce_chains": "jtmol.io.chain_deduction",
}


def __getattr__(name):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'jtmol.io' has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


# Names that would otherwise collide with submodule names -- bind them
# eagerly so ``from jtmol.io import write_pose_stack_pdb`` returns
# the function rather than the submodule. (Python's attribute lookup
# prefers existing attributes over ``__getattr__``, and submodule
# imports register the submodule on the parent; we need the function
# to win.)
from jtmol.io.write_pose_stack_pdb import (  # noqa: E402
    write_pose_stack_pdb,
)


__all__ = list(_EXPORTS)
