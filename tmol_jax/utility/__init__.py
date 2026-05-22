"""General-purpose utility helpers for the tmol_jax port.

Modules:
    - :mod:`.units` — pint-backed (or float-fallback) angle/length parsing.
    - :mod:`.cattrs_helpers` — lightweight YAML -> frozen attrs structurer.
    - :mod:`.reactive` — memoized property decorator for frozen dataclasses.
    - :mod:`.args` — no-op ``validate_args`` decorator for upstream compatibility.
    - :mod:`.tensor_helpers` — small JAX/NumPy array utilities.
"""

from __future__ import annotations

from tmol_jax.utility.args import validate_args
from tmol_jax.utility.cattrs_helpers import (
    register_hook,
    structure,
    structure_from_string,
    structure_from_yaml,
)
from tmol_jax.utility.reactive import reactive_property
from tmol_jax.utility.tensor_helpers import (
    exclusive_cumsum,
    gather_with_default,
    one_hot,
    pad_to_shape,
    segment_sum_2d,
    where_broadcast,
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
    # units
    "BondAngle",
    "DihedralAngle",
    "Length",
    "parse_angle",
    "parse_bond_angle",
    "parse_dihedral_angle",
    "parse_length",
    # cattrs helpers
    "register_hook",
    "structure",
    "structure_from_yaml",
    "structure_from_string",
    # reactive
    "reactive_property",
    # args
    "validate_args",
    # tensor helpers
    "segment_sum_2d",
    "exclusive_cumsum",
    "gather_with_default",
    "where_broadcast",
    "pad_to_shape",
    "one_hot",
]
