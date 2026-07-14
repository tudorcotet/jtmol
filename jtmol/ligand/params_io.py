"""Extend a :class:`ChemicalDatabase` with ligand restypes from YAML.

Upstream tmol's ``tmol.ligand.params_io`` handles reading and writing
Rosetta-style ``.params`` files. jtmol's scaffold only needs the *read*
path for the ``*.xtal-lig.mmff94.tmol`` YAML format that already lives
in the upstream test data, so this module just glues
:func:`jtmol.ligand.params_file.load_ligand_params_file` onto the
default jtmol :class:`ChemicalDatabase` to produce an extended
database the rest of the pipeline can consume.

Pure Python. No torch, JAX, or RDKit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import attr

from jtmol.database.chemical import AtomType, ChemicalDatabase, RawResidueType
from jtmol.ligand.params_file import LigandParamsFile, load_ligand_params_file

logger = logging.getLogger(__name__)


# Element lookup used to seed missing atom_types from a YAML params
# file. Keys are the atom_type-name *prefix* up to the first lowercase
# letter / digit; values are the element symbol. This is a fallback so
# the scaffold doesn't hard-fail when a new atom_type isn't already in
# the default chem-db; for production-grade ligand support the proper
# AtomType definition (with is_donor/is_acceptor/hybridization) must
# live in ``chemical.yaml`` (or be added via an explicit extension).
_FALLBACK_ATOM_TYPE_ELEMENT = {
    "C": "C", "N": "N", "O": "O", "H": "H",
    "S": "S", "P": "P", "F": "F", "Cl": "Cl",
    "Br": "Br", "I": "I", "Na": "Na", "K": "K",
}


def _guess_element_for_atom_type(name: str) -> Optional[str]:
    """Best-effort element guess from a Rosetta-style atom-type name.

    Examples: ``CR`` -> ``C``, ``NG22`` -> ``N``, ``Oad`` -> ``O``,
    ``ClR`` -> ``Cl``, ``HN`` -> ``H``.
    """
    # Two-letter elements first.
    for prefix in ("Cl", "Br", "Na"):
        if name.startswith(prefix):
            return _FALLBACK_ATOM_TYPE_ELEMENT.get(prefix)
    if not name:
        return None
    return _FALLBACK_ATOM_TYPE_ELEMENT.get(name[0])


def extend_db_from_yaml(
    chem_db: Optional[ChemicalDatabase],
    paths: Sequence[str | Path],
    *,
    strict_atom_types: bool = False,
) -> Tuple[ChemicalDatabase, Tuple[LigandParamsFile, ...]]:
    """Return an extended :class:`ChemicalDatabase` that includes the
    residues + atom_types defined in the supplied YAML params files.

    Parameters
    ----------
    chem_db:
        Base database. If ``None``, :meth:`ChemicalDatabase.get_default`
        is used.
    paths:
        ``.xtal-lig.mmff94.tmol`` YAML files to merge.
    strict_atom_types:
        If ``True``, raise when a residue references an atom_type that
        is not already present in the base ``chem_db.atom_types`` and
        cannot be auto-derived from a name prefix (see
        :func:`_guess_element_for_atom_type`).

    Returns
    -------
    (extended_chem_db, parsed_files):
        ``extended_chem_db`` is a fresh :class:`ChemicalDatabase` whose
        ``atom_types`` and ``residues`` tuples contain the originals
        followed by any new entries pulled from ``paths``. The element
        table and variants are unchanged.

        ``parsed_files`` is the per-path :class:`LigandParamsFile`
        objects, which downstream code (e.g. the ``fa_elec`` setup) can
        consult for partial charges.
    """
    if chem_db is None:
        chem_db = ChemicalDatabase.get_default()

    parsed = tuple(load_ligand_params_file(p) for p in paths)

    existing_atom_types = {at.name: at for at in chem_db.atom_types}
    existing_residues = {r.name: r for r in chem_db.residues}

    new_atom_types = list(chem_db.atom_types)
    new_residues = list(chem_db.residues)

    for lp in parsed:
        for at_name in lp.atom_type_names:
            if at_name in existing_atom_types:
                continue
            element = _guess_element_for_atom_type(at_name)
            if element is None:
                if strict_atom_types:
                    raise ValueError(
                        f"Unknown atom_type {at_name!r} in {lp.source_path} and "
                        "no element guess available; supply it via an extension "
                        "to chemical.yaml or pass strict_atom_types=False."
                    )
                logger.warning(
                    "Atom type %r in %s has no element guess; defaulting to 'C'.",
                    at_name, lp.source_path,
                )
                element = "C"
            new = AtomType(name=at_name, element=element)
            new_atom_types.append(new)
            existing_atom_types[at_name] = new

        for res in lp.residues:
            if res.name in existing_residues:
                logger.warning(
                    "Residue %r already in chem-db; keeping the existing "
                    "definition (from %s).", res.name, lp.source_path,
                )
                continue
            new_residues.append(res)
            existing_residues[res.name] = res

    extended = ChemicalDatabase(
        element_types=chem_db.element_types,
        atom_types=tuple(new_atom_types),
        residues=tuple(new_residues),
        variants=chem_db.variants,
    )
    return extended, parsed


def validate_smiles(smiles: str, expected_heavy_atom_count: int) -> bool:
    """Cross-check a SMILES string against an expected heavy-atom count.

    Uses the vendored pysmiles parser (no RDKit). Returns ``True`` if
    the parsed graph has exactly ``expected_heavy_atom_count`` non-
    hydrogen atoms. Used by callers that ship both a SMILES and a YAML
    params file and want to sanity-check the two agree.
    """
    from jtmol.extern.pysmiles import read_smiles

    graph = read_smiles(smiles, explicit_hydrogen=False)
    heavy = sum(
        1 for _, d in graph.nodes(data=True)
        if d.get("element", "").upper() != "H"
    )
    return heavy == expected_heavy_atom_count


__all__ = ["extend_db_from_yaml", "validate_smiles"]
