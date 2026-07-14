"""Patched chemical database.

Ported from ``tmol.chemical.patched_chemdb``.

The variant-patching engine applies every variant in the
:class:`ChemicalDatabase`\'s ``variants`` tuple to every raw residue
whose atomic graph contains the variant\'s SMILES pattern. Each
successful match produces a new :class:`RawResidueType` whose name is
``"<base>:<variant.display_name>"`` and whose atoms, bonds,
connections, and icoors have been modified per the variant spec.

The engine is applied iteratively: after each pass, newly produced
variants are themselves eligible for further patching (so
``ALA:cterm`` can then become ``ALA:cterm:nterm``, etc.). A cache of
sorted variant-name tuples prevents applying the same combination of
variants twice, and a per-residue ``marked_atoms`` set prevents a
single variant from re-patching atoms that have already been modified
by an earlier variant.

After collection, each patched :class:`RawResidueType` is wrapped in a
:class:`RefinedResidueType` so callers see a uniform view of the
residue set.
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import attr
import networkx as nx
import networkx.algorithms.isomorphism as iso

from jtmol.database.chemical import (
    AtomType,
    ChemicalDatabase,
    Element,
    Icoor,
    RawResidueType,
    VariantType,
)
from jtmol.chemical.restypes import RefinedResidueType, _refined_from_raw
from jtmol.extern.pysmiles.read_smiles import read_smiles


# --------------------------------------------------------------------- #
# Residue-graph builder                                                  #
# --------------------------------------------------------------------- #


class RestypeGraphBuilder:
    """Turn a :class:`RawResidueType` into a ``networkx.Graph`` whose
    nodes are atom names (plus one node per outgoing connection) and
    whose ``element`` attribute is used for SMILES pattern matching."""

    def __init__(self, atomtypedict: Dict[str, str]):
        self.atomtypedict = atomtypedict

    def from_raw_res(self, r: RawResidueType) -> nx.Graph:
        mol = nx.Graph()
        for x in r.atoms:
            mol.add_node(x.name, element=self.atomtypedict[x.atom_type])
        for x in r.bonds:
            mol.add_edge(x[0], x[1])
        for x in r.connections:
            # Outgoing connections are represented as pseudo-atoms with
            # the connection name surrounded by ``{}`` so the SMILES
            # pattern can match them via ``[{up}]`` / ``[{down}]`` etc.
            mol.add_node(x.name, element="{" + x.name + "}")
            mol.add_edge(x.name, x.atom)
        return mol


# --------------------------------------------------------------------- #
# Atom / bond / torsion / connection mutators                            #
# --------------------------------------------------------------------- #


def remove_atom(res: RawResidueType, atom: str) -> RawResidueType:
    """Remove all references to ``atom`` from a raw residue in place."""
    res.atoms = tuple(x for x in res.atoms if x.name != atom)
    res.bonds = tuple((x, y) for x, y in res.bonds if x != atom and y != atom)
    res.torsions = tuple(
        x for x in res.torsions if atom not in [x.a.atom, x.b.atom, x.c.atom, x.d.atom]
    )
    res.torsions = tuple(
        x
        for x in res.torsions
        if atom
        not in [x.a.connection, x.b.connection, x.c.connection, x.d.connection]
    )
    res.connections = tuple(x for x in res.connections if x.name != atom)
    return res


def modify_atom(res: RawResidueType, atom) -> RawResidueType:
    """Replace the atom with the same name as ``atom`` in ``res.atoms``."""
    res.atoms = tuple(x if x.name != atom.name else atom for x in res.atoms)
    return res


def update_icoor(res, patch, atoms_remove, namemap):
    """Apply the variant\'s icoor block to the residue\'s icoor tuple.

    For each icoor in ``patch``, pick a source icoor from ``res`` (via
    ``source`` + ``namemap``) and override whichever of ``phi``,
    ``theta``, ``d``, ``parent``, ``grand_parent``,
    ``great_grand_parent`` the patch specifies. Any of these fields
    that reference atoms with ``<...>`` names are mapped through
    ``namemap`` to the matched residue atom.
    """
    new_icoor = []
    for target_i in patch:
        if target_i.source:
            source_name_i = namemap[target_i.source]
            source_idx = [i for i, x in enumerate(res) if x.name == source_name_i]
            source_i = res[source_idx[0]]
        else:
            source_i = Icoor(
                name=None,
                phi=None,
                theta=None,
                d=None,
                parent=None,
                grand_parent=None,
                great_grand_parent=None,
            )

        p_i = source_i.parent if not target_i.parent else target_i.parent
        if p_i in namemap:
            p_i = namemap[p_i]
        gp_i = (
            source_i.grand_parent
            if not target_i.grand_parent
            else target_i.grand_parent
        )
        if gp_i in namemap:
            gp_i = namemap[gp_i]
        ggp_i = (
            source_i.great_grand_parent
            if not target_i.great_grand_parent
            else target_i.great_grand_parent
        )
        if ggp_i in namemap:
            ggp_i = namemap[ggp_i]

        icoor_i = Icoor(
            name=target_i.name,
            phi=source_i.phi if not target_i.phi else target_i.phi,
            theta=source_i.theta if not target_i.theta else target_i.theta,
            d=source_i.d if not target_i.d else target_i.d,
            parent=p_i,
            grand_parent=gp_i,
            great_grand_parent=ggp_i,
        )

        new_icoor.append(icoor_i)

    remove = [namemap[i] for i in atoms_remove]
    icoor = (*(x for x in res if x.name not in remove), *new_icoor)
    return icoor


def get_modified_atoms(patch: VariantType):
    """Return ``(added, modded, deleted)`` lists of atom names / refs.

    ``added`` and ``deleted`` track literal atom changes. ``modded``
    tracks atoms whose connectivity or coordinates change -- it
    includes the ``<...>`` reference strings, which will be mapped to
    concrete atom names after graph matching.
    """
    added: List[str] = []
    modded: List[str] = []
    deleted: List[str] = []

    for i in patch.remove_atoms:
        deleted.append(i)

    for i in patch.add_atoms:
        added.append(i.name)
    for i in patch.modify_atoms:
        modded.append(i.name)
    for i in patch.add_connections:
        modded.append(i.atom)
        added.append(i.name)

    for i, j in patch.add_bonds:
        if i[0] == "<" and i[-1] == ">" and i not in modded:
            modded.append(i)
        if j[0] == "<" and j[-1] == ">" and j not in modded:
            modded.append(j)

    for i in patch.icoors:
        if i.name[0] == "<" and i.name[-1] == ">" and i.name not in modded:
            modded.append(i.name)

    return added, modded, deleted


# --------------------------------------------------------------------- #
# Raw-residue validators                                                 #
# --------------------------------------------------------------------- #


def validate_raw_residue(res: RawResidueType) -> None:
    allatoms = {i.name for i in res.atoms}
    allconns = {i.name for i in res.connections}

    _validate_raw_residue_atoms(res, allatoms)
    _validate_raw_residue_bonds(res, allatoms)
    _validate_raw_residue_torsions(res, allatoms, allconns)
    _validate_raw_residue_connections(res, allatoms)
    _validate_raw_residue_icoors(res, allatoms, allconns)


def _validate_raw_residue_atoms(res, allatoms):
    if len(allatoms) != len(res.atoms):
        duplicated: Dict[str, int] = defaultdict(int)
        for i, at in enumerate(res.atoms):
            if at.name in duplicated:
                continue
            for j, at2 in enumerate(res.atoms[(i + 1) :]):
                if at.name == at2.name:
                    duplicated[at.name] += 1

        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: duplicated_atom_name; atoms may appear only once\n",
                "Offending atoms:\n",
                "\n".join(
                    [
                        f'    "{x}" appears {duplicated[x] + 1} times'
                        for x in duplicated
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_bonds(res, allatoms):
    bad_bonds = []
    for i, j in res.bonds:
        if i not in allatoms:
            bad_bonds.append((i, i, j))
        if j not in allatoms:
            bad_bonds.append((j, i, j))
    if bad_bonds:
        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_bond; must be between declared atoms\n"
                "Offending atoms:\n",
                "\n".join(
                    [
                        f'    Undeclared atom "{x[0]}" in bond ("{x[1]}", "{x[2]}")'
                        for x in bad_bonds
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_torsions(res, allatoms, allconns):
    bad_torsions = []
    for i in res.torsions:
        for a_i in (i.a, i.b, i.c, i.d):
            if a_i.atom is None:
                if a_i.connection is None or a_i.connection not in allconns:
                    bad_torsions.append(("connection", a_i, i))
            else:
                if a_i.atom not in allatoms:
                    bad_torsions.append(("atom", a_i, i))
    if bad_torsions:

        def str_ua(x):
            return x.atom if x.atom is not None else x.connection

        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_torsion; Torsion atoms must be either ",
                "previously-declared connections or previously-declared atoms\n",
                f'Offending atom{"s" if len(bad_torsions) > 1 else ""}:\n',
                "\n".join(
                    [
                        f'    atom "{str_ua(x[1])}" of'
                        f" ({str_ua(x[2].a)}, {str_ua(x[2].b)},"
                        f" {str_ua(x[2].c)}, {str_ua(x[2].d)})"
                        f" is not a declared {x[0]}"
                        for x in bad_torsions
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_connections(res, allatoms):
    bad_conns = []
    for i in res.connections:
        if i.atom not in allatoms:
            bad_conns.append(i)
    if bad_conns:
        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_connection; connection atom must be previously declared\n",
                f'Offending connection{"s" if len(bad_conns) > 1 else ""}:\n',
                "\n".join(
                    [
                        f'    connection "{x.name}" with undeclared atom "{x.atom}"'
                        for x in bad_conns
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_raw_residue_icoors(res, allatoms, allconns):
    bad_icoors = []
    for i in res.icoors:
        if i.name not in allatoms and i.name not in allconns:
            bad_icoors.append((i, "name"))
        for anc in ("parent", "grand_parent", "great_grand_parent"):
            a_i = getattr(i, anc)
            if a_i not in allatoms and a_i not in allconns:
                bad_icoors.append((i, anc))
    if bad_icoors:
        err_msg = "".join(
            [
                f"Bad raw residue: {res.name}\n",
                "Error: illegal_icoor; must reference previoulsy-declared atoms or connections only.\n",
                f'Offending icoor{"s" if len(bad_icoors) > 1 else ""}\n',
                "\n".join(
                    [
                        f'    icoor for {x[0].name}: {x[1]} atom "{getattr(x[0], x[1])}" undeclared'
                        for x in bad_icoors
                    ]
                ),
            ]
        )
        raise RuntimeError(err_msg)


# --------------------------------------------------------------------- #
# Patch validators                                                       #
# --------------------------------------------------------------------- #


def validate_patch(patch: VariantType) -> None:
    addedatoms = {i.name for i in patch.add_atoms}
    added_ats_and_conns = addedatoms.union({i.name for i in patch.add_connections})

    _validate_patch_fields(patch)
    _validate_patch_atom_references(
        patch, "remove_atoms", "remove_nonreference_atom", lambda x: x
    )
    _validate_patch_atom_references(
        patch, "modify_atoms", "modify_nonreference_atom", lambda x: x.name
    )
    _validate_patch_atom_aliases(patch, addedatoms)
    _validate_patch_bonds(patch, added_ats_and_conns)
    _validate_patch_icoors(patch, added_ats_and_conns)


def _validate_patch_fields(patch):
    missing = []
    required_attrs = [
        "name",
        "display_name",
        "pattern",
        "remove_atoms",
        "add_atoms",
        "add_atom_aliases",
        "modify_atoms",
        "add_connections",
        "add_bonds",
        "icoors",
    ]
    for attribute in required_attrs:
        if getattr(patch, attribute) is None:
            missing.append(attribute)
    if missing:
        err_msg = "".join(
            [
                f"Bad patch: {patch.name}\n",
                "Error: Undefined field",
                ("s: " if len(missing) > 1 else ": "),
                ", ".join(missing),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_atom_references(
    patch, atom_list_attribute_name, error_name, get_atom_name
):
    bad_atom_inds = []
    atom_list = getattr(patch, atom_list_attribute_name)
    for i in range(len(atom_list)):
        if (
            get_atom_name(atom_list[i])[0] != "<"
            or get_atom_name(atom_list[i])[-1] != ">"
        ):
            bad_atom_inds.append(i)
    if bad_atom_inds:
        err_msg = "".join(
            [
                f"Bad patch: {patch.name}\n",
                f"Error: {error_name}; ",
                f'atoms listed with "{atom_list_attribute_name}" must begin with "<" and end with ">".\n',
                "Offending atoms: ",
                ",".join([get_atom_name(atom_list[i]) for i in bad_atom_inds]),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_atom_aliases(patch, addedatoms):
    bad_aliases = []
    for i in patch.add_atom_aliases:
        if i.name not in addedatoms:
            bad_aliases.append(i)

    if bad_aliases:
        err_msg = "".join(
            [
                f"Bad patch {patch.name}\n",
                "Error: illegal_add_alias. "
                "Added atom alias must refer to newly added atoms. "
                "Bad add_atom_aliases from: ",
                ", ".join([f'"{x.name}" --> "{x.alt_name}"' for x in bad_aliases]),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_bonds(patch, added_ats_and_conns):
    bad_bonds = []
    for i, j in patch.add_bonds:
        if (i[0] != "<" or i[-1] != ">") and (i not in added_ats_and_conns):
            bad_bonds.append((i, j))
    if bad_bonds:
        err_msg = "".join(
            [
                f"Bad patch {patch.name}\n",
                "Error: illegal bond; "
                "first atom in each bond must be either atom reference ",
                '(start with "<" and end with ">") or an added atom.\n',
                "Offending bonds: ",
                ", ".join([f'("{x[0]}" "{x[1]}")' for x in bad_bonds]),
            ]
        )
        raise RuntimeError(err_msg)


def _validate_patch_icoors(patch, added_ats_and_conns):
    bad_icoors = []
    for i in patch.icoors:
        if (i.name[0] != "<" or i.name[-1] != ">") and (
            i.name not in added_ats_and_conns
        ):
            bad_icoors.append(("name", i))
        for anc in ("source", "parent", "grand_parent", "great_grand_parent"):
            a_i = getattr(i, anc)
            if a_i is None:
                if anc != "source" and getattr(i, "source") is None:
                    bad_icoors.append((anc, i))
                continue
            if (a_i[0] != "<" or a_i[-1] != ">") and (a_i not in added_ats_and_conns):
                bad_icoors.append((anc, i))
    if bad_icoors:
        added_ats_and_conns_list = sorted(added_ats_and_conns)
        err_msg = "".join(
            [
                f"Bad patch {patch.name}\n",
                "Error: illegal_icoor; ",
                "icoor atoms must be for either atom reference ",
                '(start with "<" and end with ">") or an added atom / connection,\n'
                'and ancestor atoms may only be omitted (i.e. "None") if the "source" atom is provided\n'
                "Offending icoors:\n",
                "\n".join(
                    [
                        f'    Icoor for {x[1].name} with atom reference "{x[0]}" of "{getattr(x[1], x[0])}"'
                        for x in bad_icoors
                    ]
                ),
                "\nWhere the added atom / connection list is: ",
                ", ".join([f'"{x}"' for x in added_ats_and_conns_list]),
            ]
        )
        raise RuntimeError(err_msg)


# --------------------------------------------------------------------- #
# The patch engine                                                       #
# --------------------------------------------------------------------- #


def do_patch(res, variant, resgraph, patchgraph, marked):
    """Apply ``variant`` to ``res`` and return ``(new_reses, new_marked)``.

    The residue\'s graph is scanned for subgraph monomorphisms onto the
    patch\'s SMILES graph. If any monomorphism is found whose mapped
    "modded" atom set doesn\'t collide with a previously-marked atom,
    a new :class:`RawResidueType` is produced with the variant applied.
    """
    atoms_match = (
        lambda x, y: ("element" not in y)
        or ("element" not in x)
        or x["element"] == y["element"]
    )
    gm = iso.GraphMatcher(resgraph, patchgraph, node_match=atoms_match)

    added, modded, deleted = get_modified_atoms(variant)
    assert len(modded) + len(deleted) > 0, (
        "Patch " + variant.name + " does not modify any atoms!"
    )

    # Find sub-graph matches that are unique w.r.t. the list of atoms
    # the patch will modify. (A single match site is all we support --
    # applying the same patch twice to distinct sites on the same
    # residue would require a unique-name scheme for added atoms.)
    mod_unique: List[List[str]] = []
    namemaps: List[Dict[str, str]] = []
    for subgraph_x in gm.subgraph_monomorphisms_iter():
        namemap = {
            "<" + patchgraph.nodes[y]["name"] + ">": x for x, y in subgraph_x.items()
        }
        mod_i = [namemap[x] for x in modded]
        if mod_i not in mod_unique:
            mod_unique.append(mod_i)
            namemaps.append(namemap)

    assert len(mod_unique) <= 1, (
        "Patch "
        + variant.name
        + " applies to residue "
        + res.name
        + " multiple times!"
    )

    newreses, newmarked = [], []
    for namemap in namemaps:
        newmark = copy.deepcopy(marked)
        local_modded = [namemaps[0][x] for x in modded]
        local_deleted = [namemaps[0][x] for x in deleted]

        # Atoms bonded to a deleted atom are themselves "modified".
        for i, j in res.bonds:
            if i in local_deleted and j not in local_deleted:
                local_modded.append(j)
            if j in local_deleted and i not in local_deleted:
                local_modded.append(i)

        # Skip this patch site if we\'ve already modified any of these.
        if set(local_modded) & set(newmark):
            continue

        newres = RawResidueType(
            name=res.name + ":" + variant.display_name,
            base_name=res.base_name,
            name3=res.name3,
            io_equiv_class=res.io_equiv_class,
            atoms=res.atoms,
            atom_aliases=res.atom_aliases,
            bonds=res.bonds,
            connections=res.connections,
            torsions=res.torsions,
            icoors=res.icoors,
            properties=res.properties,
            chi_samples=res.chi_samples,
            default_jump_connection_atom=res.default_jump_connection_atom,
        )

        # 1. remove atoms
        for atom in variant.remove_atoms:
            mapped = namemap[atom]
            newres = remove_atom(newres, mapped)

        # 1b. modify atoms
        for atom in variant.modify_atoms:
            atom = attr.evolve(atom, name=namemap[atom.name])
            newres = modify_atom(newres, atom)

        # 2. add atoms
        newres.atoms = (*newres.atoms, *variant.add_atoms)

        # 2b. add atom aliases
        newres.atom_aliases = (*newres.atom_aliases, *variant.add_atom_aliases)

        # 3. add connections
        newconnections = []
        for i in variant.add_connections:
            if i.atom in namemap:
                i = attr.evolve(i, atom=namemap[i.atom])
            newconnections.append(i)
        newres.connections = (*newres.connections, *newconnections)

        # 4. add bonds
        newbonds = []
        for i, j in variant.add_bonds:
            if i in namemap:
                i = namemap[i]
            if j in namemap:
                j = namemap[j]
            newbonds.append((i, j))
        newres.bonds = (*newres.bonds, *newbonds)

        # 5. update icoors
        newres.icoors = update_icoor(
            newres.icoors, variant.icoors, variant.remove_atoms, namemap
        )

        # 6. track which atoms are now "marked" as modified
        newmark.extend(local_modded)
        newmark.extend(added)
        newmark = list(filter(lambda x: x not in local_deleted, newmark))

        newreses.append(newres)
        newmarked.append(newmark)

    return newreses, newmarked


# --------------------------------------------------------------------- #
# PatchedChemicalDatabase                                                #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=True)
class PatchedChemicalDatabase:
    """A chemical database whose residues have been fully variant-patched.

    Attributes
    ----------
    element_types:
        Tuple of :class:`~jtmol.database.chemical.Element` records,
        passed through unchanged from the input database.
    atom_types:
        Tuple of :class:`~jtmol.database.chemical.AtomType` records,
        passed through unchanged from the input database.
    residues:
        Tuple of :class:`~jtmol.chemical.restypes.RefinedResidueType`
        residue types: the original raw residues **plus** every
        reachable variant combination (``"ALA:nterm"``, ``"ALA:cterm"``,
        ``"ALA:cterm:nterm"`` etc.).
    variants:
        The variant tuple from the input database, unchanged.
    """

    element_types: Tuple[Element, ...]
    atom_types: Tuple[AtomType, ...]
    residues: Tuple[RefinedResidueType, ...]
    variants: Tuple[VariantType, ...]

    @classmethod
    def from_chem_db(cls, chemdb: ChemicalDatabase) -> "PatchedChemicalDatabase":
        """Build a :class:`PatchedChemicalDatabase` by applying every
        variant in ``chemdb.variants`` to every raw residue in
        ``chemdb.residues``.

        See the module-level docstring for the overall algorithm.
        """
        G = RestypeGraphBuilder({x.name: x.element for x in chemdb.atom_types})

        for variant in chemdb.variants:
            validate_patch(variant)

        for res in chemdb.residues:
            validate_raw_residue(res)

        # Pre-compute patch graphs once -- SMILES parsing is expensive
        # and deterministic.
        patchgraphs = {
            v.name: read_smiles(v.pattern, explicit_hydrogen=True, do_fill_valence=False)
            for v in chemdb.variants
        }

        patched_residues: List[RawResidueType] = []
        patched_residues_names: List[List[str]] = []

        for res in chemdb.residues:
            # For each base residue, iteratively grow the set of its
            # variants. A variant is keyed by the sorted tuple of
            # applied variant names, so ``(cterm, nterm)`` dedupes
            # with ``(nterm, cterm)``.
            resvariants: List[RawResidueType] = [res]
            resvariantnames: List[List[str]] = [[]]
            marked_atoms: List[List[str]] = [[]]

            done = False
            while not done:
                done = True
                rv_new: List[RawResidueType] = []
                rvn_new: List[List[str]] = []
                ma_new: List[List[str]] = []

                for res_i, name_i, mark_i in zip(
                    resvariants, resvariantnames, marked_atoms
                ):
                    resgraph = G.from_raw_res(res_i)
                    for variant in chemdb.variants:
                        newtag = sorted([*name_i, variant.name])
                        if newtag in resvariantnames or newtag in rvn_new:
                            continue

                        patched_reses, mark_i_new = do_patch(
                            res_i,
                            variant,
                            resgraph,
                            patchgraphs[variant.name],
                            mark_i,
                        )

                        if patched_reses:
                            rv_new.extend(patched_reses)
                            rvn_new.extend([newtag] * len(patched_reses))
                            ma_new.extend(mark_i_new)
                            done = False

                resvariants.extend(rv_new)
                resvariantnames.extend(rvn_new)
                marked_atoms.extend(ma_new)

            patched_residues.extend(resvariants)
            patched_residues_names.extend(resvariantnames)

        for res in patched_residues:
            validate_raw_residue(res)

        refined_residues = tuple(_refined_from_raw(r) for r in patched_residues)

        return cls(
            element_types=chemdb.element_types,
            atom_types=chemdb.atom_types,
            residues=refined_residues,
            variants=chemdb.variants,
        )


# --------------------------------------------------------------------- #
# Default instance                                                       #
# --------------------------------------------------------------------- #


def default_patched_chemdb() -> PatchedChemicalDatabase:
    """Load the default :class:`PatchedChemicalDatabase` shipped with jtmol.

    Reads ``jtmol/database/default/chemical/chemical.yaml`` and
    passes the resulting :class:`ChemicalDatabase` through
    :meth:`PatchedChemicalDatabase.from_chem_db`. The result is cached
    on this module so repeat calls are cheap.
    """
    global _DEFAULT
    if _DEFAULT is None:
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "database", "default", "chemical"
        )
        chemdb = ChemicalDatabase.from_file(default_path)
        _DEFAULT = PatchedChemicalDatabase.from_chem_db(chemdb)
    return _DEFAULT


_DEFAULT: "PatchedChemicalDatabase | None" = None


__all__ = [
    "PatchedChemicalDatabase",
    "default_patched_chemdb",
    "RestypeGraphBuilder",
    "do_patch",
    "remove_atom",
    "modify_atom",
    "update_icoor",
    "get_modified_atoms",
    "validate_raw_residue",
    "validate_patch",
]
