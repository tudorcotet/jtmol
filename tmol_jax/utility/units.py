"""Unit parsing helpers — pint-based with a graceful float fallback.

Upstream tmol uses `pint` + `cattr` to let YAML parameter databases carry
unit-qualified strings like ``"1.23 rad"`` or ``"70 deg"``. We preserve that
affordance here: if `pint` is importable, angles and lengths can be parsed
from strings. If not, `parse_*` accepts only numeric values (treated as
radians / angstroms) and the `BondAngle` / `DihedralAngle` / `Length` type
aliases collapse to plain ``float``.

The three type aliases exist mainly so downstream code that imports
``from tmol_jax.utility.units import BondAngle`` keeps working.
"""

from __future__ import annotations

import math
from typing import NewType, Tuple, Union

try:
    import pint  # type: ignore

    _ureg = pint.UnitRegistry()
    _u = _ureg.parse_expression
    _HAS_PINT = True
except ImportError:  # pragma: no cover - exercised only when pint missing
    pint = None  # type: ignore
    _ureg = None
    _u = None
    _HAS_PINT = False


BondAngle = NewType("BondAngle", float)
DihedralAngle = NewType("DihedralAngle", float)
Length = NewType("Length", float)


def _to_radians(val) -> float:
    """Coerce a pint Quantity or plain number into radians as ``float``."""
    if _HAS_PINT and hasattr(val, "to"):
        return float(val.to(_ureg.rad).magnitude)
    return float(val)


def _to_length(val, target_unit: str = "angstrom") -> float:
    """Coerce a pint Quantity or plain number into ``target_unit`` as ``float``."""
    if _HAS_PINT and hasattr(val, "to"):
        return float(val.to(_ureg.parse_units(target_unit)).magnitude)
    return float(val)


def parse_angle(
    angle: Union[float, str, int],
    lim: Tuple[float, float] = (-2 * math.pi, 2 * math.pi),
) -> float:
    """Parse an angle (pint string or numeric) and return radians as ``float``.

    Args:
        angle: Unit-qualified string (e.g. ``"70 deg"``) or numeric radians.
        lim: ``(min, max)`` inclusive bounds in radians; raises ``ValueError``
            if violated. Pass ``(None, None)`` to disable checks.
    """
    if isinstance(angle, str):
        if not _HAS_PINT:
            raise RuntimeError(
                f"Cannot parse string angle {angle!r}: pint is not installed. "
                "Install pint or pass a numeric value in radians."
            )
        parsed = _ureg.parse_expression(angle)
        val = _to_radians(parsed)
    else:
        val = float(angle)

    if lim is not None:
        minv, maxv = lim
        if minv is not None and val < minv:
            raise ValueError(f"angle: {angle!r} outside of allowed range: {lim}")
        if maxv is not None and val > maxv:
            raise ValueError(f"angle: {angle!r} outside of allowed range: {lim}")

    return val


def parse_bond_angle(v: Union[float, str]) -> float:
    """Parse a bond angle on ``[0, pi]`` radians."""
    return parse_angle(v, lim=(0.0, math.pi))


def parse_dihedral_angle(v: Union[float, str]) -> float:
    """Parse a dihedral angle on ``[-pi, pi]`` radians."""
    return parse_angle(v, lim=(-math.pi, math.pi))


def parse_length(v: Union[float, str], target_unit: str = "angstrom") -> float:
    """Parse a length (pint string or numeric) to a numeric value in
    ``target_unit`` (default: angstroms)."""
    if isinstance(v, str):
        if not _HAS_PINT:
            raise RuntimeError(
                f"Cannot parse string length {v!r}: pint is not installed."
            )
        parsed = _ureg.parse_expression(v)
        return _to_length(parsed, target_unit)
    return float(v)


__all__ = [
    "BondAngle",
    "DihedralAngle",
    "Length",
    "parse_angle",
    "parse_bond_angle",
    "parse_dihedral_angle",
    "parse_length",
]
