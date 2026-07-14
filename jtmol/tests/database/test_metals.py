"""feature/metals: divalent metal cations load with jtmol-derived params.

These params are OUR OWN (see ljlk.yaml comment), not transcribed from
Rosetta. This test checks the data path: the metal atom types exist in
the chemical database and resolve to nonzero LJ/LK parameters, so a
metal ion placed in a pose would contribute to fa_ljatr/fa_lk.
"""

from __future__ import annotations


def test_metal_atom_types_and_params_load():
    from jtmol.database import ParameterDatabase

    db = ParameterDatabase.get_default()

    # Elements present.
    elems = {e.name for e in db.chemical.element_types}
    for e in ("Zn", "Mg", "Ca"):
        assert e in elems, f"element {e} missing from chemical db"

    # Atom types present with the right element.
    at = {a.name: a for a in db.chemical.atom_types}
    for name, elem in (("Zn2p", "Zn"), ("Mg2p", "Mg"), ("Ca2p", "Ca")):
        assert name in at, f"atom type {name} missing"
        assert at[name].element == elem

    # LJ/LK params resolve with our nonzero values.
    lj = {p.name: p for p in db.scoring.ljlk.atom_type_parameters}
    for name in ("Zn2p", "Mg2p", "Ca2p"):
        assert name in lj, f"{name} missing from ljlk atom_type_parameters"
        assert lj[name].lj_radius > 0, f"{name} lj_radius not positive"
        assert lj[name].lj_wdepth > 0, f"{name} lj_wdepth not positive"

    # Sanity: Zn radius < Ca radius (ionic-size ordering preserved).
    assert lj["Zn2p"].lj_radius < lj["Ca2p"].lj_radius
