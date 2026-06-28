"""Verify jtmol's converted .npz scoring tables match upstream .bin/.zip.

Loads each table from BOTH the cached .npz and the original
torch-pickled binary (via the loaders in
``tmol_jax/database/scoring/``, which already convert tensors -> numpy
and shim the upstream attrs classes). Compares every numpy array
field-by-field.

Run::

    uv run python scripts/verify_npz_bin_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = REPO_ROOT / "tmol_jax" / "database" / "default" / "scoring"
sys.path.insert(0, str(REPO_ROOT))

# ponytail: floats compared with strict tolerance; conversion is supposed
# to be a pure tensor->numpy cast with no recompute, so we expect bit-equal.
FLOAT_RTOL = 1e-6
FLOAT_ATOL = 1e-8


def _array_eq(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, f"shape {a.shape} != {b.shape}"
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        # Cast to common float, then compare; conversion script casts
        # bbstep/bbstart -> float64 but the .zip path may carry float32
        # so we compare in the wider precision.
        if not np.allclose(a, b, rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
                           equal_nan=True):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            return False, (
                f"floats differ: max|dx|={diff.max():.3e} "
                f"at idx {np.unravel_index(int(np.argmax(diff)), a.shape)}, "
                f"dtypes {a.dtype}/{b.dtype}"
            )
        bit_equal = a.dtype == b.dtype and np.array_equal(a, b)
        return True, ("bit-equal" if bit_equal
                      else f"~equal (dtype {a.dtype}/{b.dtype})")
    if not np.array_equal(a, b):
        return False, "ints/strings differ"
    return True, "bit-equal"


def verify_rama() -> list[str]:
    from tmol_jax.database.scoring.rama import RamaDatabase

    npz_t = RamaDatabase.load_tables_from_npz(str(DB_DIR / "rama.npz"))
    bin_t = RamaDatabase.load_tables_from_zip(str(DB_DIR / "rama.zip"))
    notes = [f"rama: npz={len(npz_t)} tables, zip={len(bin_t)} tables"]
    assert len(npz_t) == len(bin_t), "table count mismatch"
    for i, (a, b) in enumerate(zip(npz_t, bin_t)):
        for field in ("table_id", "table", "bbstep", "bbstart"):
            va = getattr(a, field)
            vb = getattr(b, field)
            if field == "table_id":
                ok = va == vb
                msg = "equal" if ok else f"{va!r} != {vb!r}"
            else:
                ok, msg = _array_eq(np.asarray(va), np.asarray(vb))
            notes.append(f"  t{i} {a.table_id:>20} .{field}: "
                         f"{'OK' if ok else 'FAIL'} {msg}")
    return notes


def verify_omega() -> list[str]:
    from tmol_jax.database.scoring.omega_bbdep import OmegaBBDepDatabase

    npz_t = OmegaBBDepDatabase.load_tables_from_npz(
        str(DB_DIR / "omega_bbdep.npz"))
    bin_t = OmegaBBDepDatabase.load_tables_from_zip(
        str(DB_DIR / "omega_bbdep.zip"))
    notes = [f"omega_bbdep: npz={len(npz_t)} tables, "
             f"zip={len(bin_t)} tables"]
    assert len(npz_t) == len(bin_t), "table count mismatch"
    for i, (a, b) in enumerate(zip(npz_t, bin_t)):
        for field in ("table_id", "mu", "sigma", "bbstep", "bbstart"):
            va = getattr(a, field)
            vb = getattr(b, field)
            if field == "table_id":
                ok = va == vb
                msg = "equal" if ok else f"{va!r} != {vb!r}"
            else:
                ok, msg = _array_eq(np.asarray(va), np.asarray(vb))
            notes.append(f"  t{i} {a.table_id:>20} .{field}: "
                         f"{'OK' if ok else 'FAIL'} {msg}")
    return notes


def verify_dunbrack() -> list[str]:
    from tmol_jax.database.scoring.dunbrack_libraries import (
        DunbrackRotamerLibrary,
    )

    npz_rot, npz_semi = DunbrackRotamerLibrary.load_libraries_from_npz(
        str(DB_DIR / "dunbrack.npz"))
    bin_rot, bin_semi = DunbrackRotamerLibrary.load_libraries_from_bin(
        str(DB_DIR / "dunbrack.bin"))

    notes = [f"dunbrack: npz rot={len(npz_rot)} semi={len(npz_semi)}, "
             f"bin rot={len(bin_rot)} semi={len(bin_semi)}"]
    assert len(npz_rot) == len(bin_rot)
    assert len(npz_semi) == len(bin_semi)

    rot_fields = (
        "rotamers", "rotamer_probabilities", "rotamer_means",
        "rotamer_stdvs", "prob_sorted_rot_inds",
        "backbone_dihedral_start", "backbone_dihedral_step",
        "rotamer_alias",
    )

    def _diff_lib(prefix, a, b, extra=()):
        sub = []
        if a.table_name != b.table_name:
            sub.append(f"  {prefix} table_name mismatch: "
                       f"{a.table_name!r} != {b.table_name!r}")
        for field in rot_fields:
            va = np.asarray(getattr(a.rotameric_data, field))
            vb = np.asarray(getattr(b.rotameric_data, field))
            ok, msg = _array_eq(va, vb)
            sub.append(f"  {prefix} data.{field}: "
                       f"{'OK' if ok else 'FAIL'} {msg}")
        for field in extra:
            va = np.asarray(getattr(a, field))
            vb = np.asarray(getattr(b, field))
            ok, msg = _array_eq(va, vb)
            sub.append(f"  {prefix} {field}: "
                       f"{'OK' if ok else 'FAIL'} {msg}")
        return sub

    for i, (a, b) in enumerate(zip(npz_rot, bin_rot)):
        notes.extend(_diff_lib(f"rot{i:02d} {a.table_name:>5}", a, b))

    semi_extra = (
        "non_rot_chi_start", "non_rot_chi_step", "non_rot_chi_period",
        "rotameric_chi_rotamers", "nonrotameric_chi_probabilities",
        "rotamer_boundaries",
    )
    for i, (a, b) in enumerate(zip(npz_semi, bin_semi)):
        notes.extend(_diff_lib(f"semi{i:02d} {a.table_name:>5}", a, b,
                               extra=semi_extra))
    return notes


def main() -> None:
    out: list[str] = []
    for name, fn in (("rama", verify_rama),
                     ("omega_bbdep", verify_omega),
                     ("dunbrack", verify_dunbrack)):
        out.append(f"=== {name} ===")
        try:
            out.extend(fn())
        except Exception as e:  # noqa: BLE001
            out.append(f"  ERROR: {type(e).__name__}: {e}")
        out.append("")
    text = "\n".join(out)
    print(text)
    summary_path = REPO_ROOT / "docs" / "_npz_parity_run.log"
    summary_path.write_text(text)


if __name__ == "__main__":
    main()
