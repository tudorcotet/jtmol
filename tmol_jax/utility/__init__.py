"""General-purpose utility helpers for the tmol_jax port.

Modules:
    - :mod:`.cattrs_helpers` -- lightweight YAML -> frozen attrs structurer.
"""

from __future__ import annotations

from tmol_jax.utility.cattrs_helpers import (
    register_hook,
    structure,
    structure_from_string,
    structure_from_yaml,
)

__all__ = [
    "register_hook",
    "structure",
    "structure_from_yaml",
    "structure_from_string",
]
