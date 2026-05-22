"""One-time conversion of torch-pickled scoring tables to numpy ``.npz``.

The upstream tmol database ships three files that are ``torch.save``
pickles of attrs dataclasses with ``torch.Tensor`` fields:

* ``tmol_jax/database/default/scoring/rama.zip``
* ``tmol_jax/database/default/scoring/omega_bbdep.zip``
* ``tmol_jax/database/default/scoring/dunbrack.bin``

This script uses the existing ``*.from_file`` loaders (which internally
``import torch``) to read each payload once, and then re-serialises the
extracted numpy arrays into a sibling ``.npz``. After this, the three
loaders can read the ``.npz`` at runtime without importing torch.

Run (from the repo root)::

    python scripts/convert_binary_tables.py

The script is idempotent -- rerunning overwrites the target ``.npz``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = REPO_ROOT / "tmol_jax" / "database" / "default" / "scoring"


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def convert_rama() -> Path:
    """Convert ``rama.zip`` -> ``rama.npz``."""
    from tmol_jax.database.scoring.rama import RamaDatabase

    zip_path = DB_DIR / "rama.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing source file: {zip_path}")

    tables = RamaDatabase.load_tables_from_zip(str(zip_path))

    arrays: dict[str, np.ndarray] = {
        "__n_tables": np.array(len(tables), dtype=np.int32),
    }
    for i, t in enumerate(tables):
        arrays[f"t{i}__table_id"] = np.array(t.table_id)
        arrays[f"t{i}__table"] = np.asarray(t.table, dtype=np.float32)
        arrays[f"t{i}__bbstep"] = np.asarray(t.bbstep, dtype=np.float64)
        arrays[f"t{i}__bbstart"] = np.asarray(t.bbstart, dtype=np.float64)

    out = DB_DIR / "rama.npz"
    np.savez(out, **arrays)
    return out


def convert_omega_bbdep() -> Path:
    """Convert ``omega_bbdep.zip`` -> ``omega_bbdep.npz``."""
    from tmol_jax.database.scoring.omega_bbdep import OmegaBBDepDatabase

    zip_path = DB_DIR / "omega_bbdep.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing source file: {zip_path}")

    tables = OmegaBBDepDatabase.load_tables_from_zip(str(zip_path))

    arrays: dict[str, np.ndarray] = {
        "__n_tables": np.array(len(tables), dtype=np.int32),
    }
    for i, t in enumerate(tables):
        arrays[f"t{i}__table_id"] = np.array(t.table_id)
        arrays[f"t{i}__mu"] = np.asarray(t.mu, dtype=np.float32)
        arrays[f"t{i}__sigma"] = np.asarray(t.sigma, dtype=np.float32)
        arrays[f"t{i}__bbstep"] = np.asarray(t.bbstep, dtype=np.float64)
        arrays[f"t{i}__bbstart"] = np.asarray(t.bbstart, dtype=np.float64)

    out = DB_DIR / "omega_bbdep.npz"
    np.savez(out, **arrays)
    return out


def convert_dunbrack() -> Path:
    """Convert ``dunbrack.bin`` -> ``dunbrack.npz``."""
    from tmol_jax.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

    bin_path = DB_DIR / "dunbrack.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing source file: {bin_path}")

    rot_libs, semi_libs = DunbrackRotamerLibrary.load_libraries_from_bin(
        str(bin_path)
    )

    arrays: dict[str, np.ndarray] = {
        "__n_rot": np.array(len(rot_libs), dtype=np.int32),
        "__n_semi": np.array(len(semi_libs), dtype=np.int32),
    }

    rot_data_fields = (
        "rotamers",
        "rotamer_probabilities",
        "rotamer_means",
        "rotamer_stdvs",
        "prob_sorted_rot_inds",
        "backbone_dihedral_start",
        "backbone_dihedral_step",
        "rotamer_alias",
    )

    for i, lib in enumerate(rot_libs):
        prefix = f"rot{i}__"
        arrays[prefix + "table_name"] = np.array(lib.table_name)
        for field in rot_data_fields:
            arrays[prefix + "data__" + field] = np.asarray(
                getattr(lib.rotameric_data, field)
            )

    for i, lib in enumerate(semi_libs):
        prefix = f"semi{i}__"
        arrays[prefix + "table_name"] = np.array(lib.table_name)
        for field in rot_data_fields:
            arrays[prefix + "data__" + field] = np.asarray(
                getattr(lib.rotameric_data, field)
            )
        arrays[prefix + "non_rot_chi_start"] = np.asarray(
            lib.non_rot_chi_start, dtype=np.float64
        )
        arrays[prefix + "non_rot_chi_step"] = np.asarray(
            lib.non_rot_chi_step, dtype=np.float64
        )
        arrays[prefix + "non_rot_chi_period"] = np.asarray(
            lib.non_rot_chi_period, dtype=np.float64
        )
        arrays[prefix + "rotameric_chi_rotamers"] = np.asarray(
            lib.rotameric_chi_rotamers
        )
        arrays[prefix + "nonrotameric_chi_probabilities"] = np.asarray(
            lib.nonrotameric_chi_probabilities
        )
        arrays[prefix + "rotamer_boundaries"] = np.asarray(lib.rotamer_boundaries)

    out = DB_DIR / "dunbrack.npz"
    np.savez(out, **arrays)
    return out


def main() -> None:
    _ensure_repo_on_path()

    print(f"Converting binary scoring tables in {DB_DIR} ...")

    out_rama = convert_rama()
    print(f"  wrote {out_rama} ({os.path.getsize(out_rama):,} bytes)")

    out_omega = convert_omega_bbdep()
    print(f"  wrote {out_omega} ({os.path.getsize(out_omega):,} bytes)")

    out_dun = convert_dunbrack()
    print(f"  wrote {out_dun} ({os.path.getsize(out_dun):,} bytes)")

    print("Done. The scoring loaders will now prefer these .npz files and "
          "fall back to torch only if they are missing.")


if __name__ == "__main__":
    main()
