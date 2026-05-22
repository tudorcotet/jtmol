"""Convert Rosetta Ramachandran / P_AA_pp tables to tmol format.

JAX port of :mod:`tmol.support.scoring.rewrite_rama_binary`. The
algorithm is identical to upstream — the only substitutions are:

* ``torch.Tensor`` / ``torch.save`` -> :mod:`numpy` + :mod:`pickle`.
* :class:`tmol.numeric.bspline.BSplineInterpolation` is not yet ported
  to JAX; we use :func:`scipy.interpolate.RectBivariateSpline` of
  degree 3 which is equivalent to a cubic B-spline interpolator for
  this offline resampling step. :mod:`scipy` is an optional dependency
  limited to the support scripts; it is *not* required at import time
  by :mod:`tmol_jax.database.scoring.rama`.

tmol stores energy grids that are the weighted sum of ``E_rama`` and
``E_p_aa_pp``. Preprocessing steps (unchanged from upstream):

1. convert rama probabilities to energies
2. convert P_AA_pp (and P_AA normalization) to energies
3. resample P_AA_pp (gridded at 5, 15, 25 ...) to the rama grid
   (0, 10, 20 ...)
4. combine the weighted sum using the beta16 weights
   (rama = 0.5, paapp = 0.61)

Large changes in energy may appear in low-probability regions — the
upstream script's ``eps`` M-estimate comment still applies.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy
import yaml

from tmol_jax.database.scoring._structure import structure
from tmol_jax.database.scoring.rama import RamaDatabase, RamaTables


def parse_paa(lines):
    prob_tables = {}
    for line in lines:
        cols = line.split()
        if cols[0][0] == "#":
            continue
        prob_tables[cols[0]] = float(cols[1])
    return prob_tables


def parse_lines_as_ndarrays(lines, aacol=0, phipsicol=1, probcol=3):
    prob_tables = {}
    for line in lines:
        cols = line.split()
        if cols[0][0] == "#":
            continue
        curaa = cols[aacol]
        if curaa not in prob_tables.keys():
            prob_tables[curaa] = numpy.zeros((36, 36), dtype=float)
        phi_ind, psi_ind = (
            int(float(x)) // 10 + 18 for x in cols[phipsicol : (phipsicol + 2)]
        )
        prob_tables[curaa][phi_ind, psi_ind] = float(cols[probcol])
    return prob_tables


def _bspline_resample_minus_half(energies: numpy.ndarray) -> numpy.ndarray:
    """Resample a 36x36 grid shifted by -0.5 in both axes using cubic BSpline.

    Matches upstream's ``BSplineInterpolation.interpolate`` at
    ``torch.arange(36) - 0.5`` in both axes; boundary handling is
    extrapolation (SciPy's default for ``RectBivariateSpline``).
    """
    try:
        from scipy.interpolate import RectBivariateSpline  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rewrite_rama_binary needs `scipy` installed for the "
            "P_AA_pp resampling step."
        ) from exc

    x = numpy.arange(36, dtype=float)
    spline = RectBivariateSpline(x, x, energies, kx=3, ky=3)
    new = x - 0.5
    return spline(new, new)


def parse_all_tables(rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir):
    with open(os.path.join(r3_rama_dir, "all.ramaProb")) as fh:
        general = parse_lines_as_ndarrays(fh.readlines())
    with open(os.path.join(r3_rama_dir, "prepro.ramaProb")) as fh:
        prepro = parse_lines_as_ndarrays(fh.readlines())
    with open(os.path.join(r3_paapp_dir, "a20.prop")) as fh:
        paapp = parse_lines_as_ndarrays(fh.readlines(), 2, 0)
    with open(os.path.join(r3_paa_dir, "P_AA")) as fh:
        paa = parse_paa(fh.readlines())

    for aa, prob in paapp.items():
        # convert p_aa_pp to energies (do not normalize; normalization is
        # across amino-acid types).
        energies = -numpy.log(prob / paa[aa])
        # resample, shifting -5 degrees.
        paapp[aa] = _bspline_resample_minus_half(energies)

    for aa, prob in prepro.items():
        # convert rama to energies.
        # NOTE: this normalization is not done properly in R3 for prepro.
        prob /= numpy.sum(prob)
        entropy = numpy.sum(prob * numpy.log(prob))
        energies = -numpy.log(prob) + entropy
        prepro[aa] = rama_wt * energies + paapp_wt * paapp[aa]

    for aa, prob in general.items():
        prob /= numpy.sum(prob)
        entropy = numpy.sum(prob * numpy.log(prob))
        energies = -numpy.log(prob) + entropy
        general[aa] = rama_wt * energies + paapp_wt * paapp[aa]

    return (general, prepro)


def create_rama_database(rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir):
    general, prepro = parse_all_tables(
        rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir
    )

    rama_tables = []
    for aa, prob in general.items():
        rama_tables.append(
            RamaTables(
                table_id=aa,
                table=numpy.asarray(prob, dtype=numpy.float32),
                bbstep=(numpy.pi / 18.0, numpy.pi / 18.0),
                bbstart=(-numpy.pi, -numpy.pi),
            )
        )

    for aa, prob in prepro.items():
        rama_tables.append(
            RamaTables(
                table_id=aa + "_prepro",
                table=numpy.asarray(prob, dtype=numpy.float32),
                bbstep=(numpy.pi / 18.0, numpy.pi / 18.0),
                bbstart=(-numpy.pi, -numpy.pi),
            )
        )

    path_lookup = "tmol_jax/database/default/scoring/rama.yaml"
    with open(path_lookup, "r") as infile_lookup:
        raw = yaml.safe_load(infile_lookup)
        rama_lookup = structure(
            raw["rama_lookup"],
            RamaDatabase.__attrs_attrs__[1].type,  # rama_lookup type
        )

    input_uniq_id = hash(
        (rama_wt, r3_rama_dir, paapp_wt, r3_paapp_dir, r3_paa_dir)
    )
    uniq_id = path_lookup + "," + str(input_uniq_id)
    return RamaDatabase(
        uniq_id=uniq_id,
        rama_lookup=rama_lookup,
        rama_tables=tuple(rama_tables),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosetta_dir", default=os.path.join(Path.home(), "Rosetta/main/")
    )
    args, _ = parser.parse_known_args()
    parser.add_argument(
        "--r3_rama_dir",
        default=os.path.join(
            args.rosetta_dir,
            "database/scoring/score_functions/rama/fd_beta_nov2016/",
        ),
    )
    parser.add_argument(
        "--r3_paapp_dir",
        default=os.path.join(
            args.rosetta_dir,
            "database/scoring/score_functions/P_AA_pp/shapovalov/10deg/kappa131/",
        ),
    )
    parser.add_argument(
        "--r3_paa_dir",
        default=os.path.join(
            args.rosetta_dir, "database/scoring/score_functions/P_AA_pp/"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../database/default/scoring/rama.pkl",
        ),
    )
    args = parser.parse_args()

    db = create_rama_database(
        0.5, args.r3_rama_dir, 0.61, args.r3_paapp_dir, args.r3_paa_dir
    )
    with open(args.output, "wb") as fh:
        pickle.dump(db, fh)
