"""General-purpose utility helpers for the tmol_jax port.

Modules:
    - :mod:`.units` -- pint-backed (or float-fallback) angle/length parsing.
    - :mod:`.cattrs_helpers` -- lightweight YAML -> frozen attrs structurer.
"""

from __future__ import annotations

from tmol_jax.utility.cattrs_helpers import (
    register_hook,
    structure,
    structure_from_string,
    structure_from_yaml,
)
from tmol_jax.utility.units import (
    BondAngle,
    DihedralAngle,
    Length,
    parse_angle,
    parse_bond_angle,
    parse_dihedral_angle,
    parse_length,
)

__all__ = [
    "BondAngle",
    "DihedralAngle",
    "Length",
    "parse_angle",
    "parse_bond_angle",
    "parse_dihedral_angle",
    "parse_length",
    "register_hook",
    "structure",
    "structure_from_yaml",
    "structure_from_string",
]
