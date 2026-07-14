"""``jtmol.pose`` -- PoseStack, PackedBlockTypes, and builders (JAX port).

Top-level re-exports for the pose package. Imports are lazy via
``__getattr__`` so a transient bug in one submodule does not prevent
``import jtmol.pose`` from succeeding.

Public names
------------
- :class:`PoseStack` -- the batched molecular system container.
- :class:`PackedBlockTypes` -- the residue-type palette attached to a
  ``PoseStack``.
- :class:`PoseStackBuilder` -- convenience constructors for ``PoseStack``.
- :class:`PDBInfo` -- per-atom / per-residue PDB metadata.
"""

from __future__ import annotations

import importlib


_EXPORTS = {
    "PoseStack": "jtmol.pose.pose_stack",
    "PackedBlockTypes": "jtmol.pose.packed_block_types",
    "PoseStackBuilder": "jtmol.pose.pose_stack_builder",
    "PDBInfo": "jtmol.pose.pdb_info",
}


def __getattr__(name):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'jtmol.pose' has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


__all__ = list(_EXPORTS)
