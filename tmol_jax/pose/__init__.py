"""``tmol_jax.pose`` — PoseStack, PackedBlockTypes, and builders (JAX port).

Top-level re-exports for the pose package. Imports are lazy via
``__getattr__`` so that a transient bug in one submodule (e.g. while
the parallel-port agents are still landing code) does not prevent
``import tmol_jax.pose`` from succeeding.

Public names
------------
- :class:`PoseStack` — the batched molecular system container.
- :class:`PackedBlockTypes` — the residue-type palette attached to a
  ``PoseStack``.
- :class:`PoseStackBuilder` — convenience constructors for ``PoseStack``.
- :class:`PDBInfo` — per-atom / per-residue PDB metadata.
"""

from __future__ import annotations


_EXPORTS = (
    "PoseStack",
    "PackedBlockTypes",
    "PoseStackBuilder",
    "PDBInfo",
)


def _resolve(name: str):
    if name == "PoseStack":
        from tmol_jax.pose.pose_stack import PoseStack

        return PoseStack
    if name == "PackedBlockTypes":
        from tmol_jax.pose.packed_block_types import PackedBlockTypes

        return PackedBlockTypes
    if name == "PoseStackBuilder":
        from tmol_jax.pose.pose_stack_builder import PoseStackBuilder

        return PoseStackBuilder
    if name == "PDBInfo":
        from tmol_jax.pose.pdb_info import PDBInfo

        return PDBInfo
    raise AttributeError(f"module 'tmol_jax.pose' has no attribute {name!r}")


def __getattr__(name):
    if name in _EXPORTS:
        return _resolve(name)
    raise AttributeError(f"module 'tmol_jax.pose' has no attribute {name!r}")


__all__ = list(_EXPORTS)
