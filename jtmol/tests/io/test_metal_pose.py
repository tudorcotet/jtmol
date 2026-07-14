"""Single-atom metal ion (Zn2+) PoseStack construction + scoring.

Companion to ``test_ligand_pose.py``. Closes the deferred wrinkle from the
``feature/metals`` data commit: a metal ion is a SINGLE atom, so it cannot
build an icoor root frame from three real atoms the way the multi-atom cdk2
LG1 ligand does. ``zn.tmol`` uses a self-referential icoor (parent ==
grand_parent == great_grand_parent == the atom itself) which resolves every
ancestor to index 0, giving the icoor builder a trivial identity frame and
letting ``build_coords_from_icoors`` place the lone atom at the origin.

Asserts:
* the ZN block lands in the PoseStack as exactly one block of 1 atom, at the
  coordinate it had in the input structure, and
* the complex scores finite AND the Zn actually contributes to fa_ljatr /
  fa_lk (the terms move when the Zn is removed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_TMOL = Path(__file__).resolve().parents[1] / "data" / "metals_test" / "zn.tmol"


def _build_arr(with_zn: bool):
    """A GLY dipeptide, optionally with a Zn2+ ion ~2.2 A from GLY1's O."""
    try:
        import biotite.structure as struc
    except ImportError:  # pragma: no cover
        pytest.skip("biotite not installed")

    def a(name, resn, resi, elem, xyz, hetero=False):
        return struc.Atom(
            xyz, chain_id="A", res_id=resi, res_name=resn,
            atom_name=name, element=elem, hetero=hetero,
        )

    atoms = [
        a("N", "GLY", 1, "N", [0.0, 0.0, 0.0]),
        a("CA", "GLY", 1, "C", [1.46, 0.0, 0.0]),
        a("C", "GLY", 1, "C", [2.0, 1.4, 0.0]),
        a("O", "GLY", 1, "O", [1.3, 2.4, 0.0]),
        a("N", "GLY", 2, "N", [3.3, 1.5, 0.0]),
        a("CA", "GLY", 2, "C", [4.0, 2.7, 0.0]),
        a("C", "GLY", 2, "C", [5.5, 2.5, 0.0]),
        a("O", "GLY", 2, "O", [6.1, 1.5, 0.0]),
    ]
    if with_zn:
        atoms.append(a("ZN", "ZN", 3, "ZN", [1.3, 4.6, 0.0], hetero=True))
    return struc.array(atoms)


def test_zn_in_pose_and_contributes_to_lj() -> None:
    if not _TMOL.exists():  # pragma: no cover
        pytest.skip("zn.tmol fixture not available")
    import jtmol
    from jtmol.io.pose_stack_from_biotite import (
        build_context_from_biotite,
        pose_stack_from_biotite,
    )

    arr = _build_arr(with_zn=True)

    # ZN restype lands in the packed block types as a 1-atom block.
    ctx = build_context_from_biotite(arr, ligand_params_files=[_TMOL])
    pbt = ctx.packed_block_types
    name_to_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}
    assert "ZN" in name_to_idx, "Zn restype missing from packed block types"
    zn_idx = name_to_idx["ZN"]
    assert int(np.asarray(pbt.n_atoms)[zn_idx]) == 1

    # Exactly one ZN block in the pose, holding its input coordinate.
    ps = pose_stack_from_biotite(arr, ligand_params_files=[_TMOL])
    bti = np.asarray(ps.block_type_ind)
    assert int((bti == zn_idx).sum()) == 1, "expected exactly one ZN block"
    p, b = np.argwhere(bti == zn_idx)[0]
    start = int(np.asarray(ps.block_coord_offset)[p, b])
    zn_xyz = np.asarray(ps.coords)[p, start]
    assert np.allclose(zn_xyz, [1.3, 4.6, 0.0], atol=1e-3), zn_xyz.tolist()

    # Score with the Zn; total finite, fa_ljatr/fa_lk finite.
    sfxn = jtmol.beta2016_score_function()

    def terms(pose):
        state = sfxn.setup_for_pose_stack(pose)
        return {
            st.name: float(np.asarray(v).sum())
            for st, v in sfxn.score_by_term(pose, state=state).items()
        }

    with_zn = terms(ps)
    assert np.isfinite(float(np.asarray(sfxn(ps)).sum()))
    for t in ("fa_ljatr", "fa_lk"):
        assert np.isfinite(with_zn[t]), f"{t} not finite: {with_zn[t]}"

    # The Zn contributes: remove it and the LJ terms move.
    ps_no_zn = pose_stack_from_biotite(_build_arr(with_zn=False))
    without_zn = terms(ps_no_zn)
    for t in ("fa_ljatr", "fa_lk"):
        assert with_zn[t] != without_zn[t], (
            f"{t} unchanged by the Zn ({with_zn[t]}) -- ion not contributing"
        )


if __name__ == "__main__":  # pragma: no cover - manual driver
    test_zn_in_pose_and_contributes_to_lj()
    print("PASS test_zn_in_pose_and_contributes_to_lj")
