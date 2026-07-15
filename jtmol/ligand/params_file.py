"""Parser for upstream tmol's ``*.xtal-lig.mmff94.tmol`` YAML params files.

The format is a plain YAML document with three top-level keys:

  * ``chemical.residues`` — a list of residue defs. Each residue has
    ``name``, ``base_name``, ``name3``, ``io_equiv_class``, ``atoms``
    (list of ``{name, atom_type}``), ``bonds``
    (list of ``[a, b, order, is_ring]``), ``icoors``
    (list of internal-coord entries with phi/theta/d/parent/grand_parent/great_grand_parent),
    ``properties``, ``torsions``, ``chi_samples``, ``connections``,
    and ``default_jump_connection_atom``.
  * ``elec.atom_charge_parameters`` — per-atom partial charges.
  * ``cartbonded.residue_params`` — per-residue length/angle/torsion
    parameters (unused by the current jtmol scaffold; parsed but stored
    raw on the params file).

This module exposes :func:`load_ligand_params_file` which structures a
``.tmol`` file into a frozen :class:`LigandParamsFile` and a list of
``jtmol.database.chemical.RawResidueType`` instances ready to be
appended to the default :class:`ChemicalDatabase`. The YAML's 4-tuple
bond format ``[a, b, order, is_ring]`` is collapsed to the 2-tuple
``(a, b)`` jtmol's ``RawResidueType`` already uses (order/aromaticity
metadata is not load-bearing for PoseStack construction). The parser now
also preserves those records separately on :class:`LigandParamsFile`, so
GS and TI topology packages can be audited without pretending the current
score terms consume bond order.

Pure Python. No torch, JAX, or RDKit imports.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import attr
import yaml

from jtmol.database.chemical import (
    Atom,
    ChemicalProperties,
    Connection,
    Icoor,
    PolymerProperties,
    ProtonationProperties,
    RawResidueType,
)


# --------------------------------------------------------------------- #
# Frozen container                                                       #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LigandParamsFile:
    """A parsed ``*.xtal-lig.mmff94.tmol`` file.

    ``residues`` is the list of :class:`RawResidueType` extracted from
    the ``chemical.residues`` section, ready to extend a
    :class:`ChemicalDatabase`.

    ``atom_type_names`` is the deduped set of atom-type names referenced
    by those residues. Callers may need to ensure these are present in
    the chem-db's ``atom_types`` table before constructing a
    :class:`PackedBlockTypes`.

    ``atom_charges`` maps ``(res_name, atom_name) -> partial_charge`` for
    use by the ``fa_elec`` term. Kept as raw payload for now.

    ``bond_orders`` preserves each YAML bond as
    ``(atom_a, atom_b, order, is_ring)`` keyed by residue name. The current
    scoring graph still uses only the first two fields; this metadata is for
    state validation and must not be mistaken for active bonded parameters.

    ``atom_formal_charges`` preserves explicitly declared integer
    ``formal_charge`` atom fields. Missing declarations remain missing rather
    than being silently invented as zero.

    ``cartbonded_params`` is the verbatim YAML block. Consumed by the
    ``cartbonded`` term in upstream; jtmol does not parse it yet.
    """

    residues: Tuple[RawResidueType, ...]
    atom_type_names: Tuple[str, ...]
    atom_charges: Mapping[Tuple[str, str], float]
    bond_orders: Mapping[str, Tuple[Tuple[str, str, str, bool], ...]]
    atom_formal_charges: Mapping[Tuple[str, str], int]
    cartbonded_params: Mapping[str, Any]
    source_path: str


# --------------------------------------------------------------------- #
# YAML -> dataclass                                                      #
# --------------------------------------------------------------------- #


def _parse_angle_str(v: Any) -> float:
    """Parse ``"180.0 deg"`` / ``"-1.2 rad"`` / a bare number to radians."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    parts = s.split()
    if len(parts) == 1:
        return float(parts[0])
    val = float(parts[0])
    unit = parts[1].lower()
    if unit.startswith("deg"):
        return math.radians(val)
    return val  # rad / unknown -> assume radians


def _structure_icoor(d: Mapping[str, Any]) -> Icoor:
    return Icoor(
        name=str(d["name"]),
        phi=_parse_angle_str(d.get("phi", 0.0)),
        theta=_parse_angle_str(d.get("theta", 0.0)),
        d=float(d.get("d", 0.0)),
        parent=str(d.get("parent", "")),
        grand_parent=str(d.get("grand_parent", "")),
        great_grand_parent=str(d.get("great_grand_parent", "")),
    )


def _default_chem_properties() -> ChemicalProperties:
    """Default ``ChemicalProperties`` for a non-polymer ligand."""
    return ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(
            is_polymer=False,
            polymer_type="NA",
            backbone_type="NA",
            mainchain_atoms=None,
            sidechain_chirality="NA",
            termini_variants=(),
        ),
        chemical_modifications=(),
        connectivity=(),
        protonation=ProtonationProperties(
            protonated_atoms=(),
            protonation_state="neutral",
            pH=7,
        ),
        virtual=(),
    )


def _structure_residue(d: Mapping[str, Any]) -> RawResidueType:
    """Structure one residue entry into a :class:`RawResidueType`.

    The YAML's bond entries are 4-tuples ``[a, b, order, is_ring]``;
    jtmol's :class:`RawResidueType` only stores 2-tuples ``(a, b)`` for
    now (order/aromaticity is consumed by paths jtmol has not ported).
    """
    atoms = tuple(
        Atom(name=str(a["name"]), atom_type=str(a["atom_type"]))
        for a in d.get("atoms", ())
    )
    atom_names = {atom.name for atom in atoms}
    bonds_raw = d.get("bonds", ())
    bonds: List[Tuple[str, str]] = []
    for b in bonds_raw:
        # Accept both 2-tuple and 4-tuple bond entries.
        if len(b) < 2 or str(b[0]) not in atom_names or str(b[1]) not in atom_names:
            raise ValueError(
                f"bond {b!r} in residue {d.get('name')!r} references an "
                f"unknown atom; known atoms are {sorted(atom_names)}"
            )
        bonds.append((str(b[0]), str(b[1])))

    icoors = tuple(_structure_icoor(ic) for ic in d.get("icoors", ()))

    # Inter-residue connection points (e.g. a covalent connection at the
    # electrophilic carbon of a serine-hydrolase tetrahedral intermediate).
    # Absent for ordinary ligands -> empty tuple, unchanged behaviour.
    connections = tuple(
        Connection(name=str(c["name"]), atom=str(c["atom"]))
        for c in d.get("connections", ()) or ()
    )
    if len({connection.name for connection in connections}) != len(connections):
        raise ValueError(f"duplicate connection name in residue {d.get('name')!r}")
    for connection in connections:
        if connection.atom not in atom_names:
            raise ValueError(
                f"connection {connection.name!r} in residue {d.get('name')!r} "
                f"references unknown atom {connection.atom!r}"
            )

    return RawResidueType(
        name=str(d["name"]),
        base_name=str(d.get("base_name", d["name"])),
        name3=str(d.get("name3", d["name"][:3])),
        io_equiv_class=str(d.get("io_equiv_class", d["name"])),
        atoms=atoms,
        atom_aliases=(),
        bonds=tuple(bonds),
        connections=connections,
        torsions=(),
        icoors=icoors,
        properties=_default_chem_properties(),
        chi_samples=(),
        default_jump_connection_atom=str(
            d.get("default_jump_connection_atom", atoms[0].name if atoms else "")
        ),
    )


# --------------------------------------------------------------------- #
# Public loader                                                          #
# --------------------------------------------------------------------- #


def load_ligand_params_file(path: str | Path) -> LigandParamsFile:
    """Load and structure one ``.xtal-lig.mmff94.tmol`` file."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    chemical_block = (raw or {}).get("chemical", {}) or {}
    residue_entries = chemical_block.get("residues", []) or []
    residues = tuple(_structure_residue(d) for d in residue_entries)

    bond_orders: Dict[str, Tuple[Tuple[str, str, str, bool], ...]] = {}
    atom_formal_charges: Dict[Tuple[str, str], int] = {}
    for entry in residue_entries:
        res_name = str(entry["name"])
        records: List[Tuple[str, str, str, bool]] = []
        for bond in entry.get("bonds", ()) or ():
            if len(bond) < 2:
                raise ValueError(
                    f"bond record for {res_name!r} must name two atoms, got {bond!r}"
                )
            order = str(bond[2]).upper() if len(bond) >= 3 else "UNSPECIFIED"
            is_ring = bool(bond[3]) if len(bond) >= 4 else False
            records.append((str(bond[0]), str(bond[1]), order, is_ring))
        bond_orders[res_name] = tuple(records)
        for atom in entry.get("atoms", ()) or ():
            if "formal_charge" in atom:
                atom_formal_charges[(res_name, str(atom["name"]))] = int(
                    atom["formal_charge"]
                )

    atom_type_names: List[str] = []
    seen: set = set()
    for res in residues:
        for atom in res.atoms:
            if atom.atom_type not in seen:
                seen.add(atom.atom_type)
                atom_type_names.append(atom.atom_type)

    elec_block = (raw or {}).get("elec", {}) or {}
    atom_charges: Dict[Tuple[str, str], float] = {}
    for entry in elec_block.get("atom_charge_parameters", []) or []:
        key = (str(entry["res"]), str(entry["atom"]))
        atom_charges[key] = float(entry["charge"])

    cb_block = (raw or {}).get("cartbonded", {}) or {}

    return LigandParamsFile(
        residues=residues,
        atom_type_names=tuple(atom_type_names),
        atom_charges=atom_charges,
        bond_orders=bond_orders,
        atom_formal_charges=atom_formal_charges,
        cartbonded_params=cb_block,
        source_path=str(path),
    )


__all__ = ["LigandParamsFile", "load_ligand_params_file"]
