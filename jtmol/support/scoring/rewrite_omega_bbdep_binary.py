"""Convert Rosetta backbone-dependent omega text tables to tmol format.

JAX port of :mod:`tmol.support.scoring.rewrite_omega_bbdep_binary`. The
torch-specific bits are swapped for :mod:`numpy` and :mod:`pickle` —
the resulting file is a vanilla Python pickle containing a populated
:class:`jtmol.database.scoring.omega_bbdep.OmegaBBDepDatabase`
instance, which is what that module's loader is designed to consume.

For each of the five residue classes (``all``, ``gly``, ``prepro``,
``pro``, ``valile``) this script parses ``omega_ppdep.<class>.txt``
into 36x36 ``mu`` / ``sigma`` grids and pairs them with the lookup
rows loaded from ``jtmol/database/default/scoring/omega_bbdep.yaml``.

Invoke via::

    python -m jtmol.support.scoring.rewrite_omega_bbdep_binary \
        --r3_bbdepomega_dir ~/Rosetta/main/database/scoring/score_functions/omega/
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy
import yaml

from jtmol.database.scoring._structure import structure
from jtmol.database.scoring.omega_bbdep import (
    OmegaBBDepDatabase,
    OmegaBBDepTables,
)


def parse_lines_as_ndarrays(lines, mucol=4, sigmacol=5):
    mu = numpy.zeros((36, 36), dtype=float)
    sigma = numpy.zeros((36, 36), dtype=float)
    for line in lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        x, y = int(cols[0]), int(cols[1])

        mu[x, y] = float(cols[mucol])
        sigma[x, y] = float(cols[sigmacol])
    return mu, sigma


def parse_all_tables(r3_bbdepomega_dir):
    """Read the five ``omega_ppdep.*.txt`` tables and return a dict."""
    tables = {}
    for tid in ["all", "gly", "prepro", "pro", "valile"]:
        path = os.path.join(r3_bbdepomega_dir, f"omega_ppdep.{tid}.txt")
        with open(path) as fh:
            tables[tid] = parse_lines_as_ndarrays(fh.readlines())

    return tables


def create_omega_db(r3_bbdepomega_dir):
    """Build an :class:`OmegaBBDepDatabase` from a Rosetta omega directory."""
    tables = parse_all_tables(r3_bbdepomega_dir)

    bbdep_omega_tables = []
    for aa, (mu, sig) in tables.items():
        bbdep_omega_tables.append(
            OmegaBBDepTables(
                table_id=aa,
                mu=numpy.asarray(mu, dtype=numpy.float32),
                sigma=numpy.asarray(sig, dtype=numpy.float32),
                bbstep=(numpy.pi / 18.0, numpy.pi / 18.0),
                bbstart=(numpy.pi / 36.0, numpy.pi / 36.0),
            )
        )

    path_lookup = "jtmol/database/default/scoring/omega_bbdep.yaml"
    with open(path_lookup, "r") as infile_lookup:
        raw = yaml.safe_load(infile_lookup)
        bbdep_omega_lookup = structure(
            raw["omega_bbdep_lookup"],
            OmegaBBDepDatabase.__attrs_attrs__[1].type,  # bbdep_omega_lookup type
        )

    input_uniq_id = hash(r3_bbdepomega_dir)
    uniq_id = path_lookup + "," + str(input_uniq_id)

    return OmegaBBDepDatabase(
        uniq_id=uniq_id,
        bbdep_omega_lookup=bbdep_omega_lookup,
        bbdep_omega_tables=tuple(bbdep_omega_tables),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosetta_dir", default=os.path.join(Path.home(), "Rosetta/main/")
    )
    args, _ = parser.parse_known_args()
    parser.add_argument(
        "--r3_bbdepomega_dir",
        default=os.path.join(
            args.rosetta_dir, "database/scoring/score_functions/omega/"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../database/default/scoring/omega_bbdep.pkl",
        ),
    )
    args = parser.parse_args()

    db = create_omega_db(args.r3_bbdepomega_dir)
    with open(args.output, "wb") as fh:
        pickle.dump(db, fh)
