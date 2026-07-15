"""Topology-only residue variants for explicit covalent-state construction.

These variants deliberately do not live in the default chemical database.
Adding a new residue type to that database changes the packed residue-type
palette and can perturb ordinary-protein score setup.  Instead, this module
derives the minimum connection-bearing serine variant from the caller's
already-packed ``SER`` and appends it only to a covalent-state palette.

No partial charges or state-specific bonded parameters are supplied here.
The resulting palette must therefore retain the topology-only score lock.
"""

from __future__ import annotations

from typing import Any

from jtmol.chemical.restypes import _refined_from_raw
from jtmol.database.chemical import (
    Atom,
    ChemicalProperties,
    Connection,
    RawResidueType,
)
from jtmol.pose.packed_block_types import PackedBlockTypes


def _ser_acyl_from_ser(ser: Any) -> Any:
    """Return a connection-bearing, hydrogen-free-OG topology from ``SER``.

    ``OG`` uses the established generic ether/ester oxygen type (``Oet``),
    not the hydroxyl-donor ``OH`` type.  This is atom typing for graph and
    geometry validation only; it is not a calibrated acyl-enzyme/TI energy
    model.
    """

    if "OG" not in {atom.name for atom in ser.atoms}:
        raise ValueError("cannot derive SER_acyl: source SER has no OG atom")

    atoms = tuple(
        Atom(atom.name, "Oet" if atom.name == "OG" else atom.atom_type)
        for atom in ser.atoms
        if atom.name != "HG"
    )
    retained_atom_names = {atom.name for atom in atoms}
    properties = ChemicalProperties(
        is_canonical=False,
        polymer=ser.properties.polymer,
        chemical_modifications=ser.properties.chemical_modifications,
        connectivity=ser.properties.connectivity,
        protonation=ser.properties.protonation,
        virtual=ser.properties.virtual,
    )
    raw = RawResidueType(
        name="SER_acyl",
        base_name="SER_acyl",
        name3="SER",
        io_equiv_class="SER",
        atoms=atoms,
        atom_aliases=tuple(
            alias
            for alias in ser.atom_aliases
            if alias.name in retained_atom_names
        ),
        bonds=tuple(
            bond for bond in ser.bonds if "HG" not in bond
        ),
        connections=tuple(ser.connections) + (Connection("acyl", "OG"),),
        torsions=tuple(
            torsion for torsion in ser.torsions if torsion.name != "chi2"
        ),
        icoors=tuple(icoor for icoor in ser.icoors if icoor.name != "HG"),
        properties=properties,
        chi_samples=tuple(
            samples
            for samples in ser.chi_samples
            if samples.chi_dihedral != "chi2"
        ),
        default_jump_connection_atom=ser.default_jump_connection_atom,
    )
    return _refined_from_raw(raw)


def with_topology_only_ser_acyl(pbt: Any) -> Any:
    """Append ``SER_acyl`` to ``pbt`` without mutating the default palette.

    The variant is appended, never inserted, so all caller residue-type
    indices remain stable.  Palettes without a base ``SER`` are returned
    unchanged; this permits ligand--ligand covalent graphs in custom palettes.
    """

    active = tuple(getattr(pbt, "active_block_types", ()) or ())
    if any(getattr(restype, "name", None) == "SER_acyl" for restype in active):
        return pbt

    ser = next(
        (
            restype
            for restype in active
            if getattr(restype, "name", None) == "SER"
            and getattr(restype, "base_name", None) == "SER"
        ),
        None,
    )
    if ser is None:
        return pbt

    ser_acyl = _ser_acyl_from_ser(ser)
    return PackedBlockTypes.from_restypes(
        pbt.chem_db,
        (*active, ser_acyl),
    )
