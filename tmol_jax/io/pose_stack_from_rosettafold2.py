"""RoseTTAFold2 -> :class:`PoseStack` bridge for ``tmol_jax``.

Upstream reference: ``tmol.io.pose_stack_from_rosettafold2``. That
module consumes RF2's per-residue 27-atom layout (see
``tmol.extern.rosettafold2.chemical``: ``num2aa`` + ``aa2long``) and
turns it into a :class:`CanonicalForm`. Like the OpenFold bridge, the
heart of the work is an atom-name mapping from the RF2 ordering into
tmol's canonical ordering -- a mapping that the JAX port has not yet
ported because the sibling ``tmol_jax.io.canonical_ordering`` module is
still being finalised.

Public API is preserved for symmetry with the torch port. Until the
canonical-ordering bridge is complete, :func:`pose_stack_from_rosettafold2`
and :func:`canonical_form_from_rosettafold2` raise
:class:`NotImplementedError` with a pointer to the PDB-based path: write
the RF2 output out as a PDB file and call
:func:`tmol_jax.io.pose_stack_from_pdb`.

Sister helpers :func:`canonical_ordering_for_rosettafold2` and
:func:`packed_block_types_for_rosettafold2` delegate to the default
canonical ordering / PBT so callers who need a serialisable ordering
object (e.g. for diffing poses they've built via the PDB path) still
get a consistent one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_HELP_MESSAGE = (
    "pose_stack_from_rosettafold2 is not implemented in the JAX port. "
    "The RoseTTAFold2 27-atom layout (num2aa/aa2long) has not been ported "
    "into tmol_jax's canonical-ordering tables. As a workaround, write "
    "the RF2 output to a PDB file (RF2's ``writepdb`` helper will do it) "
    "and call ``tmol_jax.io.pose_stack_from_pdb(pdb_text)`` instead. You "
    "can also convert the RF2 output into a ``biotite.structure.AtomArray`` "
    "and call ``tmol_jax.io.pose_stack_from_biotite.pose_stack_from_biotite``."
)


# --------------------------------------------------------------------- #
# Canonical ordering / PBT helpers                                       #
# --------------------------------------------------------------------- #


_CACHE: Dict[str, Any] = {}


def canonical_ordering_for_rosettafold2():
    """Return the default :class:`CanonicalOrdering` for RoseTTAFold2 inputs.

    The upstream tmol port constructed a bespoke subset (20 canonical
    AAs + HIS_D + CYD, with n/c-term patches) from the RF2 atom tables.
    Until the canonical-ordering bridge here is richer, this JAX port
    simply hands back the default canonical ordering; the restype
    coverage is a superset of what RF2 emits, so callers storing a
    reference to this object for post-conversion use won't miss any
    residues.
    """
    if "co" in _CACHE:
        return _CACHE["co"]
    from tmol_jax.io.canonical_ordering import default_canonical_ordering

    co = default_canonical_ordering()
    _CACHE["co"] = co
    return co


def packed_block_types_for_rosettafold2():
    """Return the default :class:`PackedBlockTypes` for RoseTTAFold2 inputs."""
    if "pbt" in _CACHE:
        return _CACHE["pbt"]
    from tmol_jax.io.canonical_ordering import default_packed_block_types

    pbt = default_packed_block_types()
    _CACHE["pbt"] = pbt
    return pbt


# --------------------------------------------------------------------- #
# Stubs                                                                  #
# --------------------------------------------------------------------- #


def canonical_form_from_rosettafold2(
    seq: Any,
    xyz: Any,
    chainlens: List[int],
) -> Dict[str, Any]:
    """Stub: convert RF2 ``(seq, xyz, chainlens)`` to a canonical-form dict.

    Not implemented in the JAX port. Raises :class:`NotImplementedError`
    with guidance on using the PDB / biotite round-trip instead.
    """
    raise NotImplementedError(_HELP_MESSAGE)


def pose_stack_from_rosettafold2(
    seq: Any,
    xyz: Any,
    chainlens: List[int],
    *,
    chain_id: Optional[Any] = None,  # noqa: ARG001 - API parity with torch port
    res_num: Optional[Any] = None,  # noqa: ARG001
    **kwargs: Any,
):
    """Stub: build a :class:`PoseStack` from RF2's ``(seq, xyz, chainlens)``.

    Not implemented in the JAX port. See :data:`_HELP_MESSAGE` for the
    recommended workarounds.
    """
    raise NotImplementedError(_HELP_MESSAGE)


__all__ = [
    "canonical_ordering_for_rosettafold2",
    "packed_block_types_for_rosettafold2",
    "canonical_form_from_rosettafold2",
    "pose_stack_from_rosettafold2",
]
