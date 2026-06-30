"""Parity test: MOL2 ligand atom typing vs the Rosetta .tmol reference.

Reads the raw ``cdk2.lig.mol2`` ligand, assigns Rosetta generic-potential
atom types via the ported classifier, and compares against the 28 reference
``atom_type`` fields in ``cdk2.xtal-lig.mmff94.tmol``. Atoms are matched by
order (the .tmol reference and the MOL2 share atom order: heavy atoms then
hydrogens).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

_DATA = Path(__file__).parent.parent / "data" / "protein_ligand_test"
_MOL2 = _DATA / "cdk2.lig.mol2"
_REF_TMOL = _DATA / "cdk2.xtal-lig.mmff94.tmol"

_REF_LINE = re.compile(r"\{name:\s*(?P<name>\S+?),\s*atom_type:\s*(?P<type>\S+?)\}")


def _reference_atom_types() -> list[tuple[str, str]]:
    """Return [(atom_name, atom_type), ...] from the .tmol reference, in order."""
    out: list[tuple[str, str]] = []
    for line in _REF_TMOL.read_text().splitlines():
        m = _REF_LINE.search(line)
        if m:
            out.append((m.group("name"), m.group("type")))
    return out


def test_cdk2_mol2_atom_typing_parity() -> None:
    from tmol_jax.ligand.atom_typing import assign_atom_types_from_mol2

    ref = _reference_atom_types()
    assert len(ref) == 28, f"expected 28 reference atoms, got {len(ref)}"

    got = assign_atom_types_from_mol2(_MOL2)
    assert len(got) == len(ref), f"atom count {len(got)} != reference {len(ref)}"

    mismatches = [
        (i, name, exp, got[i])
        for i, (name, exp) in enumerate(ref)
        if got[i] != exp
    ]
    n_match = len(ref) - len(mismatches)
    detail = "\n".join(
        f"  atom {i} ({name}): expected {exp!r}, got {g!r}"
        for i, name, exp, g in mismatches
    )
    assert not mismatches, (
        f"cdk2 atom-typing parity {n_match}/{len(ref)}; mismatches:\n{detail}"
    )
