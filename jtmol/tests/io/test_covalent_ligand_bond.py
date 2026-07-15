"""Covalent-state topology: catalytic Ser-Ogamma -> mapped ligand C.

Exercises the ``covalent_bonds=`` loader seam end to end, reusing the exact
disulfide connection + bond-separation machinery (see
``docs/COVALENT_STATE_TOPOLOGY.md``). A protein serine's OG is bonded to the
electrophilic ("tetrahedral") carbon of a small-molecule tetrahedral-intermediate
(TI) mimic, so the TI is a genuine covalent adduct:

* the OG->C bond is in the bond graph and ``inter_block_bondsep`` gives path
  length 1 across it (1-2/1-3/1-4 exclusions then propagate across the
  junction, verified via count_pair);
* scoring the topology is categorically rejected because current jtmol does
  not consume bond order or state-specific cartbonded parameters;
* mutating the serine to alanine (no OG) makes the bond *categorically*
  impossible -- the loader refuses (``test_nucleophile_knockout_cannot_bond``);
* a disulfide-containing structure still loads + scores through the shared
  path, unchanged (``test_disulfide_still_loads_and_scores``).

The TI mimic is built in-process from 1qys's Ser A27 (no committed fixture):
a valence-complete tetrahedral C1 bonded to an oxyanion, an alkoxy leaving
group, an acyl substituent, and (through the external connection) Ser OG.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


_DATA = Path(__file__).resolve().parents[1] / "data"
_1QYS = _DATA / "1qys.pdb"
_1CGI = _DATA / "1CGI.pdb"

# A minimal serine-hydrolase tetrahedral-intermediate topology: an sp3
# carbon C1 carrying an oxyanion, alkoxy group, and carbon substituent, with a
# declared covalent connection ``lg_c`` on C1. Formal charges are explicit.
# Fitted partial charges are intentionally absent because topology-only poses
# cannot enter the energy backend.
_TI_TMOL = """\
version: "1.0"
chemical:
  residues:
  - name: TI1
    base_name: TI1
    name3: TI1
    io_equiv_class: TI1
    atom_aliases: []
    atoms:
    - {name: C1, atom_type: CS3, formal_charge: 0}
    - {name: O1, atom_type: OG31, formal_charge: -1}
    - {name: O2, atom_type: Oet, formal_charge: 0}
    - {name: C2, atom_type: CS3, formal_charge: 0}
    - {name: C3, atom_type: CS3, formal_charge: 0}
    - {name: H21, atom_type: Hapo, formal_charge: 0}
    - {name: H22, atom_type: Hapo, formal_charge: 0}
    - {name: H23, atom_type: Hapo, formal_charge: 0}
    - {name: H31, atom_type: Hapo, formal_charge: 0}
    - {name: H32, atom_type: Hapo, formal_charge: 0}
    - {name: H33, atom_type: Hapo, formal_charge: 0}
    bonds:
    - [C1, O1, SINGLE, false]
    - [C1, O2, SINGLE, false]
    - [O2, C2, SINGLE, false]
    - [C1, C3, SINGLE, false]
    - [C2, H21, SINGLE, false]
    - [C2, H22, SINGLE, false]
    - [C2, H23, SINGLE, false]
    - [C3, H31, SINGLE, false]
    - [C3, H32, SINGLE, false]
    - [C3, H33, SINGLE, false]
    connections:
    - {name: lg_c, atom: C1}
    icoors:
    - {name: C1, phi: 0.0 deg, theta: 0.0 deg, d: 0.0, parent: C1, grand_parent: O1, great_grand_parent: O2}
    - {name: O1, phi: 0.0 deg, theta: 180.0 deg, d: 1.25, parent: C1, grand_parent: O1, great_grand_parent: O2}
    - {name: O2, phi: 120.0 deg, theta: 109.5 deg, d: 1.30, parent: C1, grand_parent: O1, great_grand_parent: O2}
    - {name: C2, phi: 120.0 deg, theta: 109.5 deg, d: 1.43, parent: O2, grand_parent: C1, great_grand_parent: O1}
    - {name: C3, phi: -120.0 deg, theta: 109.5 deg, d: 1.50, parent: C1, grand_parent: O1, great_grand_parent: O2}
    - {name: H21, phi: 0.0 deg, theta: 109.5 deg, d: 1.09, parent: C2, grand_parent: O2, great_grand_parent: C1}
    - {name: H22, phi: 120.0 deg, theta: 109.5 deg, d: 1.09, parent: C2, grand_parent: O2, great_grand_parent: C1}
    - {name: H23, phi: -120.0 deg, theta: 109.5 deg, d: 1.09, parent: C2, grand_parent: O2, great_grand_parent: C1}
    - {name: H31, phi: 0.0 deg, theta: 109.5 deg, d: 1.09, parent: C3, grand_parent: C1, great_grand_parent: O1}
    - {name: H32, phi: 120.0 deg, theta: 109.5 deg, d: 1.09, parent: C3, grand_parent: C1, great_grand_parent: O1}
    - {name: H33, phi: -120.0 deg, theta: 109.5 deg, d: 1.09, parent: C3, grand_parent: C1, great_grand_parent: O1}
elec:
  atom_charge_parameters: []
"""


def _biotite_or_skip():
    try:
        import biotite.structure  # noqa: F401
        import biotite.structure.io.pdb  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("biotite not installed")
    if not _1QYS.exists():  # pragma: no cover
        pytest.skip("1qys.pdb fixture not available")


def _build_ti_complex(mutate_ser_to_ala: bool = False):
    """1qys + a TI mimic covalently placed off Ser A27's OG.

    Returns ``(atom_array, params_path)``; caller must ``os.unlink`` the
    params file. When ``mutate_ser_to_ala`` is set, Ser A27 loses its OG/HG
    and is renamed ALA (the nucleophile knockout) -- everything else identical.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdb as bpdb

    arr = bpdb.PDBFile.read(str(_1QYS)).get_structure(model=1)
    ser = arr[(arr.chain_id == "A") & (arr.res_id == 27)]
    og = ser[ser.atom_name == "OG"].coord[0]
    cb = ser[ser.atom_name == "CB"].coord[0]

    # C1 sits 1.43 A from OG along the CB->OG direction (an ester-like Og-C
    # bond); the two oxygens are placed off C1 at finite, non-degenerate
    # positions (exact geometry is irrelevant -- we only need real coords).
    direction = og - cb
    direction /= np.linalg.norm(direction)
    c1 = og + 1.43 * direction
    o1 = c1 + np.array([1.2, 0.0, 0.3], dtype=np.float32)
    o2 = c1 + np.array([0.0, 1.2, 0.3], dtype=np.float32)
    c2 = o2 + np.array([0.2, 1.4, -0.1], dtype=np.float32)
    c3 = c1 + np.array([-1.2, -0.2, 0.5], dtype=np.float32)

    if mutate_ser_to_ala:
        keep = ~(
            (arr.chain_id == "A")
            & (arr.res_id == 27)
            & np.isin(arr.atom_name, ["OG", "HG"])
        )
        arr = arr[keep]
        m = (arr.chain_id == "A") & (arr.res_id == 27)
        arr.res_name[m] = "ALA"

    h21 = c2 + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    h22 = c2 + np.array([-0.3, 0.8, 0.5], dtype=np.float32)
    h23 = c2 + np.array([-0.3, -0.8, 0.5], dtype=np.float32)
    h31 = c3 + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    h32 = c3 + np.array([-0.3, 0.8, 0.5], dtype=np.float32)
    h33 = c3 + np.array([-0.3, -0.8, 0.5], dtype=np.float32)

    lig = struc.AtomArray(11)
    lig.coord = np.array(
        [c1, o1, o2, c2, c3, h21, h22, h23, h31, h32, h33],
        dtype=np.float32,
    )
    lig.chain_id = np.array(["B"] * 11)
    lig.res_id = np.array([501] * 11)
    lig.res_name = np.array(["TI1"] * 11)
    lig.atom_name = np.array(
        ["C1", "O1", "O2", "C2", "C3", "H21", "H22", "H23", "H31", "H32", "H33"]
    )
    lig.element = np.array(["C", "O", "O", "C", "C"] + ["H"] * 6)
    lig.hetero = np.array([True] * 11)

    f = tempfile.NamedTemporaryFile("w", suffix=".tmol", delete=False)
    f.write(_TI_TMOL)
    f.close()
    return arr + lig, f.name


def _find_block(ps, *, base_name=None, name=None):
    abt = ps.packed_block_types.active_block_types
    bti = np.asarray(ps.block_type_ind)[0]
    for i in range(len(bti)):
        if bti[i] < 0:
            continue
        bt = abt[int(bti[i])]
        if base_name is not None and getattr(bt, "base_name", None) == base_name:
            return i, bt
        if name is not None and getattr(bt, "name", None) == name:
            return i, bt
    raise AssertionError(f"no block base_name={base_name} name={name} in pose")


def _conn_slot(bt, conn_name):
    return next(k for k, c in enumerate(bt.connections) if c.name == conn_name)


# --------------------------------------------------------------------------- #
# Baseline isolation: covalent variants must not perturb ordinary proteins    #
# --------------------------------------------------------------------------- #


def test_ser_acyl_is_absent_from_default_palette_and_appended_on_demand() -> None:
    """Keep ordinary block-type indices fixed; append only for covalent IO."""
    from jtmol.chemical.covalent_state import with_topology_only_ser_acyl
    from jtmol.pose.packed_block_types import PackedBlockTypes

    default = PackedBlockTypes.default()
    before = tuple(restype.name for restype in default.active_block_types)
    assert "SER_acyl" not in before

    covalent = with_topology_only_ser_acyl(default)
    after = tuple(restype.name for restype in covalent.active_block_types)
    assert after[:-1] == before
    assert after[-1] == "SER_acyl"
    assert tuple(restype.name for restype in default.active_block_types) == before

    ordinary_ser = next(
        restype for restype in default.active_block_types if restype.name == "SER"
    )
    ser_acyl = covalent.active_block_types[-1]
    assert "HG" in {atom.name for atom in ordinary_ser.atoms}
    assert next(atom.atom_type for atom in ordinary_ser.atoms if atom.name == "OG") == "OH"
    assert "HG" not in {atom.name for atom in ser_acyl.atoms}
    assert next(atom.atom_type for atom in ser_acyl.atoms if atom.name == "OG") == "Oet"
    assert {conn.name: conn.atom for conn in ser_acyl.connections}["acyl"] == "OG"


# --------------------------------------------------------------------------- #
# (a) load + assert bond in graph + bondsep path length 1                   #
# --------------------------------------------------------------------------- #


def test_covalent_ser_ligand_bond_topology() -> None:
    _biotite_or_skip()
    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    comb, params = _build_ti_complex()
    try:
        ps = pose_stack_from_biotite(
            comb,
            ligand_params_files=[params],
            covalent_bonds=[(("A", 27, "OG"), ("B", 501, "C1"))],
            covalent_state_mode="topology_only",
        )
    finally:
        os.unlink(params)

    # The catalytic serine was promoted to the connection-bearing variant.
    sk, bt_s = _find_block(ps, base_name="SER_acyl")
    tk, bt_t = _find_block(ps, name="TI1")
    assert bt_s.atom_to_idx["OG"] >= 0
    assert next(a.atom_type for a in bt_s.atoms if a.name == "OG") == "Oet"
    assert bt_t.state_bond_orders == (
        ("C1", "O1", "SINGLE", False),
        ("C1", "O2", "SINGLE", False),
        ("O2", "C2", "SINGLE", False),
        ("C1", "C3", "SINGLE", False),
        ("C2", "H21", "SINGLE", False),
        ("C2", "H22", "SINGLE", False),
        ("C2", "H23", "SINGLE", False),
        ("C3", "H31", "SINGLE", False),
        ("C3", "H32", "SINGLE", False),
        ("C3", "H33", "SINGLE", False),
    )
    assert bt_t.atom_formal_charges == {
        "C1": 0,
        "O1": -1,
        "O2": 0,
        "C2": 0,
        "C3": 0,
        "H21": 0,
        "H22": 0,
        "H23": 0,
        "H31": 0,
        "H32": 0,
        "H33": 0,
    }
    # Explicit-H valence check for the minimal anionic tetrahedral adduct.
    # C1 has three intra-ligand single bonds plus the external Ser-OG bond;
    # both methyl carbons have four explicit bonds, O2 has two, and the
    # oxyanion O1 has one bond with formal charge -1.
    degree = {atom.name: 0 for atom in bt_t.atoms}
    for atom_a, atom_b, _order, _is_ring in bt_t.state_bond_orders:
        degree[atom_a] += 1
        degree[atom_b] += 1
    degree["C1"] += 1  # external Ser-OG--C1 bond
    assert degree["C1"] == 4
    assert degree["O1"] == 1 and bt_t.atom_formal_charges["O1"] == -1
    assert degree["O2"] == 2
    assert degree["C2"] == degree["C3"] == 4
    methyl_hydrogens = ("H21", "H22", "H23", "H31", "H32", "H33")
    assert all(degree[name] == 1 for name in methyl_hydrogens)
    acyl = _conn_slot(bt_s, "acyl")
    lg_c = _conn_slot(bt_t, "lg_c")

    # 1. The OG->C1 bond is declared in inter_residue_connections both ways.
    irc = np.asarray(ps.inter_residue_connections)[0]
    assert irc[sk, acyl, 0] == tk and irc[sk, acyl, 1] == lg_c
    assert irc[tk, lg_c, 0] == sk and irc[tk, lg_c, 1] == acyl

    # 2. inter_block_bondsep across the OG-C1 connection is 1 (one bond).
    ibs = np.asarray(ps.inter_block_bondsep)[0]
    assert ibs[sk, tk, acyl, lg_c] == 1
    assert ibs[tk, sk, lg_c, acyl] == 1

    # 3. The bond is in the atom-level bond graph: count_pair gives path
    #    length 1 for OG-C1, and the 1-2/1-3 exclusions propagate across the
    #    junction (CB-C1 = 2, OG-O1 = 2), so bonded atoms are excluded from
    #    non-bonded energy (weight 0), not scored as a clash.
    from jtmol.score.common.count_pair import (
        connectivity_weight,
        pairwise_bonded_path_length,
    )

    offs = np.asarray(ps.block_coord_offset)[0]

    def gi(block, bt, atom):
        return int(offs[block]) + [a.name for a in bt.atoms].index(atom)

    pbl = np.asarray(pairwise_bonded_path_length(ps))[0]
    og_i = gi(sk, bt_s, "OG")
    cb_i = gi(sk, bt_s, "CB")
    c1_i = gi(tk, bt_t, "C1")
    o1_i = gi(tk, bt_t, "O1")
    assert pbl[og_i, c1_i] == 1, "OG-C1 must be one bond apart"
    assert pbl[cb_i, c1_i] == 2, "CB-C1 (1-3 across junction) must be 2"
    assert pbl[og_i, o1_i] == 2, "OG-O1 (1-3 across junction) must be 2"
    assert float(connectivity_weight(pbl[og_i, c1_i])) == 0.0

    # 4. The graph cannot be mistaken for a parameterized TI energy.
    import jtmol

    sfxn = jtmol.beta2016_score_function()
    with pytest.raises(ValueError, match="topology-only covalent state"):
        sfxn.setup_for_pose_stack(ps)
    # Supplying a precomputed/foreign state must not bypass the same lock.
    with pytest.raises(ValueError, match="topology-only covalent state"):
        sfxn.score_by_term(ps, state=object())


def test_covalent_state_mode_is_explicit_and_fail_closed() -> None:
    _biotite_or_skip()
    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    comb, params = _build_ti_complex()
    try:
        with pytest.raises(ValueError, match="covalent_state_mode"):
            pose_stack_from_biotite(
                comb,
                ligand_params_files=[params],
                covalent_bonds=[(("A", 27, "OG"), ("B", 501, "C1"))],
            )
        with pytest.raises(NotImplementedError, match="bond order|bond-order"):
            pose_stack_from_biotite(
                comb,
                ligand_params_files=[params],
                covalent_bonds=[(("A", 27, "OG"), ("B", 501, "C1"))],
                covalent_state_mode="parameterized",
            )
    finally:
        os.unlink(params)


# --------------------------------------------------------------------------- #
# (b) nucleophile knockout: the covalent bond categorically cannot form       #
# --------------------------------------------------------------------------- #


def test_nucleophile_knockout_cannot_bond() -> None:
    _biotite_or_skip()
    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    comb, params = _build_ti_complex(mutate_ser_to_ala=True)
    try:
        with pytest.raises(ValueError, match="cannot form|categorically"):
            pose_stack_from_biotite(
                comb,
                ligand_params_files=[params],
                covalent_bonds=[(("A", 27, "OG"), ("B", 501, "C1"))],
                covalent_state_mode="topology_only",
            )
    finally:
        os.unlink(params)


def test_knockout_still_loads_without_the_covalent_bond() -> None:
    """The S160A-style mutant is only *categorical* about the bond, not the
    structure: drop the ``covalent_bonds`` request and the same complex loads
    fine (proving the refusal above is the bond, not a broken input)."""
    _biotite_or_skip()
    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    comb, params = _build_ti_complex(mutate_ser_to_ala=True)
    try:
        ps = pose_stack_from_biotite(comb, ligand_params_files=[params])
    finally:
        os.unlink(params)
    # ALA at A27 (no SER_acyl), TI1 present but unbonded.
    _, bt_a = _find_block(ps, base_name="ALA")
    assert all(c.name != "acyl" for c in bt_a.connections)
    import jtmol

    assert np.isfinite(float(np.asarray(jtmol.beta2016_score_function()(ps)).sum()))


# --------------------------------------------------------------------------- #
# (c) regression: a disulfide-containing structure still loads + scores        #
# --------------------------------------------------------------------------- #


def test_disulfide_still_loads_and_scores() -> None:
    """1CGI (disulfide-rich) loads through the shared connection path and
    scores finite, with its CYD SG-SG bonds still wired at bondsep 1."""
    _biotite_or_skip()
    if not _1CGI.exists():  # pragma: no cover
        pytest.skip("1CGI.pdb fixture not available")
    import biotite.structure.io.pdb as bpdb

    from jtmol.io.pose_stack_from_biotite import pose_stack_from_biotite

    arr = bpdb.PDBFile.read(str(_1CGI)).get_structure(model=1)
    ps = pose_stack_from_biotite(arr)  # no covalent_bonds -> unchanged path

    abt = ps.packed_block_types.active_block_types
    bti = np.asarray(ps.block_type_ind)[0]
    irc = np.asarray(ps.inter_residue_connections)[0]
    ibs = np.asarray(ps.inter_block_bondsep)[0]

    cyd_blocks = [
        i
        for i in range(len(bti))
        if bti[i] >= 0 and getattr(abt[int(bti[i])], "base_name", None) == "CYD"
    ]
    assert len(cyd_blocks) >= 2, "1CGI should have CYD residues"

    wired = 0
    for b in cyd_blocks:
        bt = abt[int(bti[b])]
        dslf = next((k for k, c in enumerate(bt.connections) if c.name == "dslf"), -1)
        if dslf < 0:
            continue
        pj, cj = irc[b, dslf]
        if pj >= 0:
            assert ibs[b, pj, dslf, cj] == 1, "disulfide SG-SG bondsep must be 1"
            wired += 1
    assert wired >= 2, "expected disulfide bonds to be wired"

    import jtmol

    total = float(np.asarray(jtmol.beta2016_score_function()(ps)).sum())
    assert np.isfinite(total)


# --------------------------------------------------------------------------- #
# ligand connection seam unit test (no biotite / no RDKit)                     #
# --------------------------------------------------------------------------- #


def test_ligand_params_file_parses_connections() -> None:
    """The ligand params loader now surfaces declared connection points
    (previously hardcoded to an empty tuple)."""
    from jtmol.ligand.params_file import load_ligand_params_file

    f = tempfile.NamedTemporaryFile("w", suffix=".tmol", delete=False)
    f.write(_TI_TMOL)
    f.close()
    try:
        lp = load_ligand_params_file(f.name)
    finally:
        os.unlink(f.name)
    res = lp.residues[0]
    conns = {c.name: c.atom for c in res.connections}
    assert conns == {"lg_c": "C1"}
    assert lp.bond_orders["TI1"] == (
        ("C1", "O1", "SINGLE", False),
        ("C1", "O2", "SINGLE", False),
        ("O2", "C2", "SINGLE", False),
        ("C1", "C3", "SINGLE", False),
        ("C2", "H21", "SINGLE", False),
        ("C2", "H22", "SINGLE", False),
        ("C2", "H23", "SINGLE", False),
        ("C3", "H31", "SINGLE", False),
        ("C3", "H32", "SINGLE", False),
        ("C3", "H33", "SINGLE", False),
    )
    assert lp.atom_formal_charges == {
        ("TI1", "C1"): 0,
        ("TI1", "O1"): -1,
        ("TI1", "O2"): 0,
        ("TI1", "C2"): 0,
        ("TI1", "C3"): 0,
        ("TI1", "H21"): 0,
        ("TI1", "H22"): 0,
        ("TI1", "H23"): 0,
        ("TI1", "H31"): 0,
        ("TI1", "H32"): 0,
        ("TI1", "H33"): 0,
    }


def test_gs_ti_topologies_keep_distinct_reactive_bond_metadata() -> None:
    """Conserved maps do not erase the carbonyl-to-oxyanion state change."""
    from jtmol.ligand.params_file import load_ligand_params_file

    gs_text = (
        _TI_TMOL.replace("TI1", "GS", 4)
        .replace("formal_charge: -1", "formal_charge: 0", 1)
        .replace("[C1, O1, SINGLE, false]", "[C1, O1, DOUBLE, false]", 1)
    )
    handles = []
    try:
        for payload in (gs_text, _TI_TMOL):
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".tmol", delete=False
            )
            handle.write(payload)
            handle.close()
            handles.append(handle.name)
        gs, ti = map(load_ligand_params_file, handles)
    finally:
        for path in handles:
            os.unlink(path)

    gs_res, ti_res = gs.residues[0], ti.residues[0]
    assert tuple(a.name for a in gs_res.atoms) == tuple(a.name for a in ti_res.atoms)
    assert gs.bond_orders["GS"][0][2] == "DOUBLE"
    assert ti.bond_orders["TI1"][0][2] == "SINGLE"
    assert gs.atom_formal_charges[("GS", "O1")] == 0
    assert ti.atom_formal_charges[("TI1", "O1")] == -1


def test_his_pos_microstate_selector() -> None:
    """Both imidazole hydrogens select the already-parameterized HIS_POS."""
    from jtmol.database import ParameterDatabase
    from jtmol.io.pose_stack_construction import _resolve_his_spcase
    from jtmol.score.elec.params import _partial_charges_dict

    names = ("ND1", "NE2", "HD1", "HE2")
    present = np.array([True, True, True, True])
    coords = np.zeros((4, 3), dtype=np.float32)
    assert _resolve_his_spcase("HIS", names, present, coords) == "hispos"
    charge_table = _partial_charges_dict(
        ParameterDatabase.get_default().scoring.elec
    )["HIS_POS"]
    assert sum(v.get("", 0.0) for v in charge_table.values()) == pytest.approx(1.0)
