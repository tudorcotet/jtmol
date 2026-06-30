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
``tmol_jax.database.chemical.RawResidueType`` instances ready to be
appended to the default :class:`ChemicalDatabase`. The YAML's 4-tuple
bond format ``[a, b, order, is_ring]`` is collapsed to the 2-tuple
``(a, b)`` jtmol's ``RawResidueType`` already uses (order/aromaticity
metadata is currently only consumed by the upstream cartbonded path
and is not load-bearing for PoseStack construction).

Pure Python. No torch, JAX, or RDKit imports.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import attr
import yaml

from tmol_jax.database.chemical import (
    Atom,
    ChemicalProperties,
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

    ``cartbonded_params`` is the verbatim YAML block. Consumed by the
    ``cartbonded`` term in upstream; jtmol does not parse it yet.
    """

    residues: Tuple[RawResidueType, ...]
    atom_type_names: Tuple[str, ...]
    atom_charges: Mapping[Tuple[str, str], float]
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
    bonds_raw = d.get("bonds", ())
    bonds: List[Tuple[str, str]] = []
    for b in bonds_raw:
        # Accept both 2-tuple and 4-tuple bond entries.
        bonds.append((str(b[0]), str(b[1])))

    icoors = tuple(_structure_icoor(ic) for ic in d.get("icoors", ()))

    return RawResidueType(
        name=str(d["name"]),
        base_name=str(d.get("base_name", d["name"])),
        name3=str(d.get("name3", d["name"][:3])),
        io_equiv_class=str(d.get("io_equiv_class", d["name"])),
        atoms=atoms,
        atom_aliases=(),
        bonds=tuple(bonds),
        connections=(),
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
        cartbonded_params=cb_block,
        source_path=str(path),
    )


__all__ = ["LigandParamsFile", "load_ligand_params_file"]
