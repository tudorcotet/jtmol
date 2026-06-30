"""Protein-ligand PoseStack construction + scoring (upstream PR #361).

Loads the cdk2 complex (protein + the LG1 small-molecule inhibitor) from
``cdk2_complex.pdb`` plus the pre-baked ``cdk2.xtal-lig.mmff94.tmol``
generic-potential params file, builds a :class:`PoseStack` through the
real ``pose_stack_from_biotite`` entry point, and confirms:

* the LG1 ligand block lands in the PoseStack with all 28 of its atoms
  (18 heavy + 10 H, matching the ``.tmol`` atom list), and
* ``beta2016_score_function`` scores the complex to a finite total with
  non-zero ``fa_ljatr`` / ``fa_lk`` (the ligand's generic-potential atom
  types now carry real LJ/LK params, so it contributes).

Deferred (NOT asserted here): the ligand's partial-charge -> ``fa_elec``
wiring and its donor/acceptor ``hbond`` flags. Those need the
``elec.atom_charge_parameters`` block and the generic-potential
``hbond.yaml`` mappings ported, which is a documented follow-up. This
round only certifies LJ/LK + finiteness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_DATA = Path(__file__).resolve().parents[1] / "data" / "protein_ligand_test"
_PDB = _DATA / "cdk2_complex.pdb"
_TMOL = _DATA / "cdk2.xtal-lig.mmff94.tmol"

# LG1 atom count in the .tmol params file (18 heavy + 10 H).
_LG1_N_ATOMS = 28


def _load_atoms():
    try:
        import biotite.structure.io.pdb as bpdb
    except ImportError:  # pragma: no cover
        pytest.skip("biotite not installed")
    if not _PDB.exists() or not _TMOL.exists():  # pragma: no cover
        pytest.skip("cdk2 protein-ligand fixtures not available")
    return bpdb.PDBFile.read(str(_PDB)).get_structure(model=1)


def test_ligand_in_pose_and_scores() -> None:
    from tmol_jax.io.pose_stack_from_biotite import (
        build_context_from_biotite,
        pose_stack_from_biotite,
    )

    atoms = _load_atoms()

    # Identify LG1's block-type index + atom count from the extended pbt.
    ctx = build_context_from_biotite(atoms, ligand_params_files=[_TMOL])
    pbt = ctx.packed_block_types
    name_to_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}
    assert "LG1" in name_to_idx, "ligand restype missing from packed block types"
    lg1_idx = name_to_idx["LG1"]
    assert int(np.asarray(pbt.n_atoms)[lg1_idx]) == _LG1_N_ATOMS

    # Build the PoseStack via the real entry point and confirm the ligand
    # block (with all its atoms) is present.
    ps = pose_stack_from_biotite(atoms, ligand_params_files=[_TMOL])
    n_lg1_blocks = int((np.asarray(ps.block_type_ind) == lg1_idx).sum())
    assert n_lg1_blocks == 1, f"expected exactly one LG1 block, got {n_lg1_blocks}"

    # Score the complex; ligand must not break LJ/LK and must contribute.
    import tmol_jax

    sfxn = tmol_jax.beta2016_score_function()
    state = sfxn.setup_for_pose_stack(ps)
    per_term = {
        st.name: float(np.asarray(v).sum())
        for st, v in sfxn.score_by_term(ps, state=state).items()
    }
    total = float(np.asarray(sfxn(ps)).sum())

    assert np.isfinite(total), f"non-finite total {total}"
    for term in ("fa_ljatr", "fa_lk"):
        val = per_term.get(term)
        assert val is not None, f"missing score term {term}"
        assert np.isfinite(val), f"{term} not finite: {val}"
        assert val != 0.0, f"{term} is zero -- ligand LJ/LK not contributing"


def _lg1_block_slice(ps, pbt, lg1_idx):
    """Return ``(pose, start, n)`` for the single LG1 block in ``ps``."""
    bti = np.asarray(ps.block_type_ind)
    offs = np.asarray(ps.block_coord_offset)
    n_per_type = np.asarray(pbt.n_atoms)
    p, b = np.argwhere(bti == lg1_idx)[0]
    start = int(offs[p, b])
    n = int(n_per_type[lg1_idx])
    return int(p), start, n


def test_ligand_partial_charges_flow_into_fa_elec() -> None:
    """The ligand's parsed ``.tmol`` partial charges reach fa_elec.

    Two assertions:

    1. The elec per-atom ``charges`` array is non-zero at the LG1 block's
       atom slots (the ligand carries real partial charges now).
    2. ``fa_elec`` is finite AND differs from the value obtained by
       zeroing only the ligand atoms' charges -- proving the ligand
       actually contributes to the electrostatics term.
    """
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.io.pose_stack_from_biotite import (
        build_context_from_biotite,
        pose_stack_from_biotite,
    )
    from tmol_jax.score.elec.elec_energy_term import ElecEnergyTerm
    from tmol_jax.score.score_types import ScoreType

    atoms = _load_atoms()

    ctx = build_context_from_biotite(atoms, ligand_params_files=[_TMOL])
    pbt = ctx.packed_block_types
    lg1_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}["LG1"]
    ps = pose_stack_from_biotite(atoms, ligand_params_files=[_TMOL])

    term = ElecEnergyTerm(ParameterDatabase.get_default())
    state = term.setup_for_pose_stack(ps, pbt)
    charges = np.asarray(state.charges)  # [P, N]

    p, start, n = _lg1_block_slice(ps, pbt, lg1_idx)
    lig_charges = charges[p, start : start + n]
    n_nonzero = int((lig_charges != 0.0).sum())
    assert n_nonzero > 0, "ligand atoms carry no partial charge in elec array"

    fa_elec = float(np.asarray(term(ps, state)[ScoreType.fa_elec]).sum())
    assert np.isfinite(fa_elec), f"fa_elec not finite: {fa_elec}"

    # Re-score with only the ligand charges zeroed.
    import jax.numpy as jnp

    ch_zeroed = charges.copy()
    ch_zeroed[p, start : start + n] = 0.0
    state_zeroed = state.__class__(
        charges=jnp.asarray(ch_zeroed),
        bonded_path_length=state.bonded_path_length,
        atom_is_real=state.atom_is_real,
        D=state.D,
        D0=state.D0,
        S=state.S,
        min_dis=state.min_dis,
        max_dis=state.max_dis,
    )
    fa_elec_zeroed = float(
        np.asarray(term(ps, state_zeroed)[ScoreType.fa_elec]).sum()
    )
    assert fa_elec != fa_elec_zeroed, (
        "fa_elec unchanged when ligand charges zeroed -- ligand not "
        f"contributing (both {fa_elec})"
    )


if __name__ == "__main__":  # pragma: no cover - manual driver
    import sys

    failed = []
    for fn in (
        test_ligand_in_pose_and_scores,
        test_ligand_partial_charges_flow_into_fa_elec,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except pytest.skip.Exception as exc:
            print(f"SKIP  {fn.__name__}: {exc}")
        except Exception as exc:
            print(f"FAIL  {fn.__name__}: {exc!r}")
            failed.append(fn.__name__)
    sys.exit(1 if failed else 0)


def test_ligand_participates_in_hbond() -> None:
    """The ligand's polar atoms are classified as donors/acceptors and
    contribute to the hbond term.

    The generic-potential atom types carry ``is_donor`` / ``is_acceptor``
    flags in ``chemical.yaml``, and ``hbond.yaml`` has the generic
    donor/acceptor type mappings + polynomials. So LG1's N/O atoms can
    form hydrogen bonds. We assert (1) the LG1 restype has at least one
    donor or acceptor atom, and (2) the complex hbond total is finite.
    Protein-hbond parity is covered separately by the panel snapshot.
    """
    import tmol_jax
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.io.pose_stack_from_biotite import (
        build_context_from_biotite,
        pose_stack_from_biotite,
    )
    from tmol_jax.score.score_types import ScoreType

    atoms = _load_atoms()
    ctx = build_context_from_biotite(atoms, ligand_params_files=[_TMOL])
    pbt = ctx.packed_block_types
    lg1 = {bt.name: bt for bt in pbt.active_block_types}["LG1"]

    # The hbond term classifies donors/acceptors by atom-type NAME against
    # the chemical database (not the per-restype atom flags, which the
    # .tmol parser leaves False). Check LG1's atom types there.
    chem_flags = {
        at.name: (getattr(at, "is_donor", False), getattr(at, "is_acceptor", False))
        for at in ParameterDatabase.get_default().chemical.atom_types
    }
    lg1_types = [a.atom_type for a in lg1.atoms]
    n_polar = sum(1 for t in lg1_types if any(chem_flags.get(t, (False, False))))
    assert n_polar > 0, (
        "LG1 has no donor/acceptor atom types in the chemical db -- "
        f"ligand cannot hbond (types={sorted(set(lg1_types))})"
    )

    ps = pose_stack_from_biotite(atoms, ligand_params_files=[_TMOL])
    sfxn = tmol_jax.beta2016_score_function()
    per = sfxn.score_by_term(ps)
    hb = float(np.asarray(per[ScoreType.hbond]).sum())
    assert np.isfinite(hb), f"hbond total not finite: {hb}"
