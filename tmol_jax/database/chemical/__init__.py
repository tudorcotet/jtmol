"""JAX-friendly port of ``tmol.database.chemical``.

Defines the frozen attrs dataclasses that mirror the upstream
``chemical.yaml`` schema and provides a ``ChemicalDatabase.from_file``
loader that does the YAML -> dataclass structuring in pure Python,
without relying on PyTorch or cattrs registered hooks.

Key differences from upstream:
  * No ``torch`` / ``cattr`` import -- these classes are plain, frozen
    attrs dataclasses so they interoperate cleanly with JAX pytrees if
    they ever need to flow through ``jit`` or ``vmap`` (the chemical DB
    is static, so they are normally treated as aux data).
  * We parse angle strings (e.g. ``"180.0 deg"``) into floats in radians
    ourselves, with a small self-contained parser; if ``cattrs`` happens
    to be installed and a caller wants to use it elsewhere, that's fine
    too but we never depend on it here.
  * A recursive ``_structure`` function walks the YAML tree and matches
    ``dict`` keys against each attrs class' field names, converting
    lists into tuples when the annotated type is ``Tuple[...]``.
"""

from __future__ import annotations

import math
import os
import typing
from typing import Any, NewType, Optional, Tuple

import attr
import yaml


# --------------------------------------------------------------------- #
# Angle parsing                                                          #
# --------------------------------------------------------------------- #
#
# Upstream tmol used pint for unit-aware angle parsing; we don't need
# pint for anything else, so we implement a minimal parser that handles
# the forms that show up in chemical.yaml: bare numbers (already radians)
# and strings like ``"180 deg"`` / ``"-1.2 rad"``.

AcceptorHybridization = NewType("AcceptorHybridization", str)
_ACCEPTOR_HYBRIDIZATIONS = {"sp2", "sp3", "ring"}

BondAngle = NewType("BondAngle", float)
DihedralAngle = NewType("DihedralAngle", float)


def _parse_angle(v: Any) -> float:
    """Parse a number or unit-qualified string into radians."""
    if v is None:
        return v  # type: ignore[return-value]
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        # Split into value and optional unit.
        parts = s.split()
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            val = float(parts[0])
            unit = parts[1].lower()
            if unit in ("rad", "radian", "radians"):
                return val
            if unit in ("deg", "degree", "degrees"):
                return val * math.pi / 180.0
            raise ValueError(f"Unknown angle unit: {unit!r} in {v!r}")
        raise ValueError(f"Cannot parse angle: {v!r}")
    raise TypeError(f"Cannot parse angle from type {type(v).__name__}: {v!r}")


def _parse_acceptor_hybridization(v: Any) -> Optional[str]:
    if v is None:
        return None
    if v in _ACCEPTOR_HYBRIDIZATIONS:
        return v
    raise ValueError(f"Invalid AcceptorHybridization value: {v!r}")


# --------------------------------------------------------------------- #
# Dataclasses                                                            #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Element:
    name: str
    atomic_number: int


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomType:
    name: str
    element: str
    is_acceptor: bool = False
    is_donor: bool = False
    is_hydroxyl: bool = False
    is_polarh: bool = False
    acceptor_hybridization: Optional[AcceptorHybridization] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    name: str
    atom_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomAlias:
    name: str
    alt_name: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Icoor:
    name: str
    phi: DihedralAngle
    theta: BondAngle
    d: float
    parent: str
    grand_parent: str
    great_grand_parent: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Connection:
    name: str
    atom: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class UnresolvedAtom:
    atom: Optional[str] = None
    connection: Optional[str] = None
    bond_sep_from_conn: Optional[int] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Torsion:
    name: str
    a: UnresolvedAtom
    b: UnresolvedAtom
    c: UnresolvedAtom
    d: UnresolvedAtom


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChiSamples:
    chi_dihedral: str
    samples: Tuple[float, ...]
    expansions: Tuple[float, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SidechainBuilding:
    chi_samples: ChiSamples


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PolymerProperties:
    is_polymer: bool
    polymer_type: str
    backbone_type: str
    mainchain_atoms: Optional[Tuple[str, ...]]
    sidechain_chirality: str
    termini_variants: Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ProtonationProperties:
    protonated_atoms: Tuple[str, ...]
    protonation_state: str
    pH: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalProperties:
    is_canonical: bool
    polymer: PolymerProperties
    chemical_modifications: Tuple[str, ...]
    connectivity: Tuple[str, ...]
    protonation: ProtonationProperties
    virtual: Tuple[str, ...]


@attr.s(auto_attribs=True)
class RawResidueType:
    """Residue definition.

    Mirrors the upstream class: not slotted / not frozen, so downstream
    patching code is still able to mutate fields in place (e.g. the
    variant-application pipeline does this).
    """

    name: str
    base_name: str
    name3: str
    io_equiv_class: str
    atoms: Tuple[Atom, ...]
    atom_aliases: Tuple[AtomAlias, ...]
    bonds: Tuple[Tuple[str, str], ...]
    connections: Tuple[Connection, ...]
    torsions: Tuple[Torsion, ...]
    icoors: Tuple[Icoor, ...]
    properties: ChemicalProperties
    chi_samples: Tuple[ChiSamples, ...]
    default_jump_connection_atom: str

    def atom_name(self, index: int) -> str:
        return self.atoms[index].name


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IcoorVariant:
    name: str
    source: Optional[str] = None
    phi: Optional[DihedralAngle] = 0.0
    theta: Optional[BondAngle] = 0.0
    d: Optional[float] = 0.0
    parent: Optional[str] = None
    grand_parent: Optional[str] = None
    great_grand_parent: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PolymerPropertiesVariant:
    polymer_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalPropertiesVariant:
    polymer: Optional[PolymerPropertiesVariant] = None


@attr.s(auto_attribs=True)
class VariantType:
    name: str
    display_name: str
    pattern: str
    remove_atoms: Tuple[str, ...]
    add_atoms: Tuple[Atom, ...]
    add_atom_aliases: Tuple[AtomAlias, ...]
    modify_atoms: Tuple[Atom, ...]
    add_connections: Tuple[Connection, ...]
    add_bonds: Tuple[Tuple[str, str], ...]
    icoors: Tuple[IcoorVariant, ...]


# --------------------------------------------------------------------- #
# YAML -> dataclass structurer                                           #
# --------------------------------------------------------------------- #


def _is_attrs_class(t: Any) -> bool:
    return isinstance(t, type) and hasattr(t, "__attrs_attrs__")


def _unwrap_optional(t: Any) -> Tuple[Any, bool]:
    """If ``t`` is ``Optional[X]`` / ``Union[X, None]``, return ``(X, True)``.

    Otherwise return ``(t, False)``.
    """
    origin = typing.get_origin(t)
    if origin is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return args[0], True
    return t, False


def _structure(value: Any, target_type: Any) -> Any:
    """Recursively structure a YAML-decoded value into ``target_type``.

    Handles primitives, ``Optional[T]``, ``Tuple[T, ...]`` /
    ``Tuple[T1, T2]``, and nested attrs dataclasses. Custom hooks are
    applied for angle-like types and for the ``AcceptorHybridization``
    string enum.
    """

    # Unwrap Optional / Union[..., None]
    target_type, _optional = _unwrap_optional(target_type)

    if value is None:
        return None

    # Custom scalar hooks -------------------------------------------------
    if target_type is BondAngle or target_type is DihedralAngle:
        return _parse_angle(value)
    if target_type is AcceptorHybridization:
        return _parse_acceptor_hybridization(value)

    # Nested attrs dataclasses --------------------------------------------
    if _is_attrs_class(target_type):
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected mapping for {target_type.__name__}, got {type(value).__name__}"
            )
        kwargs = {}
        # Use resolved type hints so forward references (``"Foo"``) work.
        try:
            hints = typing.get_type_hints(target_type)
        except Exception:
            hints = {a.name: a.type for a in target_type.__attrs_attrs__}
        for field in target_type.__attrs_attrs__:
            if field.name in value:
                kwargs[field.name] = _structure(
                    value[field.name], hints.get(field.name, field.type)
                )
            elif field.default is not attr.NOTHING:
                # Leave it for attrs to apply the default.
                pass
            else:
                raise KeyError(
                    f"Missing required field {field.name!r} for {target_type.__name__}"
                )
        return target_type(**kwargs)

    # Generic container types ---------------------------------------------
    origin = typing.get_origin(target_type)
    args = typing.get_args(target_type)

    if origin in (tuple, Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            elem_t = args[0]
            return tuple(_structure(v, elem_t) for v in value)
        # Fixed-length tuple (e.g. ``Tuple[str, str]`` for a bond pair).
        if len(args) == len(value):
            return tuple(_structure(v, a) for v, a in zip(value, args))
        # Fall back to element-by-element using the first type arg.
        if args:
            return tuple(_structure(v, args[0]) for v in value)
        return tuple(value)

    if origin is list:
        elem_t = args[0] if args else Any
        return [_structure(v, elem_t) for v in value]

    # Plain primitives (str, int, float, bool, Any, or unknown).
    return value


# --------------------------------------------------------------------- #
# ChemicalDatabase                                                       #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase:
    element_types: Tuple[Element, ...]
    atom_types: Tuple[AtomType, ...]
    residues: Tuple[RawResidueType, ...]
    variants: Tuple[VariantType, ...]

    _default: typing.ClassVar[Optional["ChemicalDatabase"]] = None

    @classmethod
    def get_default(cls) -> "ChemicalDatabase":
        """Load and return the default chemical database (once, memoized).

        Looks for ``tmol_jax/database/default/chemical/chemical.yaml``
        relative to this file.
        """
        if cls._default is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "default", "chemical"
            )
            cls._default = cls.from_file(default_path)
        return cls._default

    @classmethod
    def from_file(cls, path: str) -> "ChemicalDatabase":
        """Load a ``ChemicalDatabase`` from ``<path>/chemical.yaml``."""
        yaml_path = os.path.join(path, "chemical.yaml")
        with open(yaml_path, "r") as infile:
            raw = yaml.safe_load(infile)
        return _structure(raw, cls)


__all__ = [
    "AcceptorHybridization",
    "BondAngle",
    "DihedralAngle",
    "Element",
    "AtomType",
    "Atom",
    "AtomAlias",
    "Icoor",
    "Connection",
    "UnresolvedAtom",
    "Torsion",
    "ChiSamples",
    "SidechainBuilding",
    "PolymerProperties",
    "ProtonationProperties",
    "ChemicalProperties",
    "RawResidueType",
    "IcoorVariant",
    "PolymerPropertiesVariant",
    "ChemicalPropertiesVariant",
    "VariantType",
    "ChemicalDatabase",
]


if __name__ == "__main__":
    # Smoke test: load the default database that ships with tmol_jax.
    # Not executed during packaging -- just here for manual sanity checks.
    db = ChemicalDatabase.get_default()
    print(f"elements: {len(db.element_types)}")
    print(f"atom_types: {len(db.atom_types)}")
    print(f"residues: {len(db.residues)}")
    print(f"variants: {len(db.variants)}")
