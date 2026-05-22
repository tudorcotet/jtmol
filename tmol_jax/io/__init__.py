"""``tmol_jax.io`` -- PDB parsing and canonical-form construction.

Pure-Python / numpy port of ``tmol.io``. All public names are exposed
lazily through ``__getattr__`` so a transient bug in one submodule
(e.g. while parallel-port agents are still landing code) does not
prevent ``import tmol_jax.io`` from succeeding.

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


_EXPORTS = (
    # Sibling-agent ports (pdb_parsing.py, canonical_form.py, canonical_ordering.py)
    "parse_pdb",
    "pose_stack_from_pdb",
    "CanonicalOrdering",
    "default_canonical_ordering",
    "default_packed_block_types",
    "canonical_form_from_pdb",
    # This-agent ports
    "pose_stack_from_canonical_form",
    "canonical_form_from_pose_stack",
    "atom_records_from_pose_stack",
    "write_pose_stack_pdb",
    "deduce_chains",
)


def _resolve(name: str):
    if name == "parse_pdb":
        from tmol_jax.io.pdb_parsing import parse_pdb

        return parse_pdb
    if name == "pose_stack_from_pdb":
        from tmol_jax.io.pdb_parsing import pose_stack_from_pdb

        return pose_stack_from_pdb
    if name == "CanonicalOrdering":
        from tmol_jax.io.canonical_ordering import CanonicalOrdering

        return CanonicalOrdering
    if name == "default_canonical_ordering":
        from tmol_jax.io.canonical_ordering import default_canonical_ordering

        return default_canonical_ordering
    if name == "default_packed_block_types":
        from tmol_jax.io.canonical_ordering import default_packed_block_types

        return default_packed_block_types
    if name == "canonical_form_from_pdb":
        from tmol_jax.io.canonical_form import canonical_form_from_pdb

        return canonical_form_from_pdb
    if name == "pose_stack_from_canonical_form":
        from tmol_jax.io.pose_stack_construction import (
            pose_stack_from_canonical_form,
        )

        return pose_stack_from_canonical_form
    if name == "canonical_form_from_pose_stack":
        from tmol_jax.io.pose_stack_deconstruction import (
            canonical_form_from_pose_stack,
        )

        return canonical_form_from_pose_stack
    if name == "atom_records_from_pose_stack":
        from tmol_jax.io.write_pose_stack_pdb import atom_records_from_pose_stack

        return atom_records_from_pose_stack
    if name == "write_pose_stack_pdb":
        from tmol_jax.io.write_pose_stack_pdb import write_pose_stack_pdb

        return write_pose_stack_pdb
    if name == "deduce_chains":
        from tmol_jax.io.chain_deduction import deduce_chains

        return deduce_chains
    raise AttributeError(f"module 'tmol_jax.io' has no attribute {name!r}")


def __getattr__(name):
    if name in _EXPORTS:
        return _resolve(name)
    raise AttributeError(f"module 'tmol_jax.io' has no attribute {name!r}")


# Names that would otherwise collide with submodule names -- bind them
# eagerly so ``from tmol_jax.io import write_pose_stack_pdb`` returns
# the function rather than the submodule. (Python's attribute lookup
# prefers existing attributes over ``__getattr__``, and submodule
# imports register the submodule on the parent; we need the function
# to win.)
from tmol_jax.io.write_pose_stack_pdb import (  # noqa: E402
    write_pose_stack_pdb,
)


__all__ = list(_EXPORTS)
