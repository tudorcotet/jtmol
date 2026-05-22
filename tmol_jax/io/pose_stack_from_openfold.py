"""OpenFold -> :class:`PoseStack` bridge for ``tmol_jax``.

Upstream reference: ``tmol.io.pose_stack_from_openfold``. That module
reads OpenFold's ``{aatype, positions, chain_index}`` output dictionary
and routes it through a carefully-constructed
:class:`CanonicalOrdering` / :class:`CanonicalForm` / :class:`PackedBlockTypes`
triple. The mapping itself relies on OpenFold's big
``restype_name_to_atom14_names`` / atom37 tables in
``tmol.extern.openfold.residue_constants``.

For this JAX port, we expose the same public API but do *not* fully
port the atom-ordering machinery: the shared
``tmol_jax.io.canonical_ordering`` module is still being finalised by a
sibling agent, and the OpenFold atom37 layout is sufficiently involved
that a defensive stub is strictly better than a half-correct mapping.
The happy path for users who have an OpenFold output dict today is to
write it out as a PDB (which OpenFold itself can do) and then call
:func:`tmol_jax.io.pose_stack_from_pdb`.

Concretely:

* :func:`canonical_ordering_for_openfold` / :func:`packed_block_types_for_openfold`
  delegate to the default canonical ordering / PBT so callers who just
  want a serialisable ``CanonicalForm`` to pair with an OpenFold output
  (post-conversion) can still get one.
* :func:`canonical_form_from_openfold` and :func:`pose_stack_from_openfold`
  raise :class:`NotImplementedError` with a pointer to the PDB path.

When the sibling canonical-ordering module gains first-class OpenFold
support, those two stubs should be replaced with direct implementations
mirroring the torch port line-for-line.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


_HELP_MESSAGE = (
    "pose_stack_from_openfold is not implemented in the JAX port. The "
    "OpenFold atom37/atom14 ordering tables have not been ported. As a "
    "workaround, write the OpenFold result out to a PDB file (OpenFold's "
    "``protein.to_pdb`` helper does this) and call "
    "``tmol_jax.io.pose_stack_from_pdb(pdb_text)`` instead. Alternatively, "
    "convert the OpenFold output into a ``biotite.structure.AtomArray`` and "
    "call ``tmol_jax.io.pose_stack_from_biotite.pose_stack_from_biotite``."
)


# --------------------------------------------------------------------- #
# Canonical ordering / PBT helpers                                       #
# --------------------------------------------------------------------- #


_CACHE: Dict[str, Any] = {}


def canonical_ordering_for_openfold():
    """Return the default :class:`CanonicalOrdering` for OpenFold inputs.

    The upstream tmol port built a bespoke subset (canonical 20 AAs +
    HIS_D + CYD with n/c-term patches); the JAX port delegates to the
    shared default until a bespoke subset is needed.
    """
    if "co" in _CACHE:
        return _CACHE["co"]
    from tmol_jax.io.canonical_ordering import default_canonical_ordering

    co = default_canonical_ordering()
    _CACHE["co"] = co
    return co


def packed_block_types_for_openfold():
    """Return the default :class:`PackedBlockTypes` for OpenFold inputs."""
    if "pbt" in _CACHE:
        return _CACHE["pbt"]
    from tmol_jax.io.canonical_ordering import default_packed_block_types

    pbt = default_packed_block_types()
    _CACHE["pbt"] = pbt
    return pbt


# --------------------------------------------------------------------- #
# Stubs                                                                  #
# --------------------------------------------------------------------- #


def canonical_form_from_openfold(openfold_result_dictionary: Any) -> Dict[str, Any]:
    """Stub: convert an OpenFold output dict to a canonical-form dict.

    Not implemented in the JAX port. Raises :class:`NotImplementedError`
    with guidance on using the PDB / biotite round-trip instead.
    """
    raise NotImplementedError(_HELP_MESSAGE)


def pose_stack_from_openfold(
    openfold_result_dictionary: Any,
    *,
    chain_id: Optional[Any] = None,  # noqa: ARG001 - kept for API parity
    res_num: Optional[Any] = None,  # noqa: ARG001
    **kwargs: Any,
):
    """Stub: build a :class:`PoseStack` from an OpenFold output dict.

    Not implemented in the JAX port. See :data:`_HELP_MESSAGE` for the
    recommended workarounds (write the OpenFold output to PDB and call
    :func:`tmol_jax.io.pose_stack_from_pdb`, or route via
    :mod:`tmol_jax.io.pose_stack_from_biotite`).
    """
    raise NotImplementedError(_HELP_MESSAGE)


__all__ = [
    "canonical_ordering_for_openfold",
    "packed_block_types_for_openfold",
    "canonical_form_from_openfold",
    "pose_stack_from_openfold",
]
