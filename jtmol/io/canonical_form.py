"""Alias module for canonical-form helpers.

Upstream tmol exposed these at ``tmol.io.canonical_form``; in this port
the actual implementations live in :mod:`jtmol.io.canonical_ordering`.
This module re-exports them at the legacy path so user imports like
``from jtmol.io.canonical_form import canonical_form_from_pdb`` keep
working.
"""

from __future__ import annotations

from jtmol.io.canonical_ordering import (
    CanonicalOrdering,
    canonical_form_from_pdb,
    default_canonical_ordering,
    default_packed_block_types,
)

__all__ = [
    "CanonicalOrdering",
    "canonical_form_from_pdb",
    "default_canonical_ordering",
    "default_packed_block_types",
]
