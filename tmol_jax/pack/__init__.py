"""tmol_jax.pack — rotamer packing / design engine (pure-JAX port).

Ported from ``tmol.pack``. The pipeline follows the upstream engine at
a structural level:

    build_rotamers (Dunbrack) -> pairwise-RPE computation ->
    Goldstein DEE pruning -> simulated annealing -> impose assignment

The individual stages live in submodules:

* :mod:`rotamer_sampler` — Dunbrack-backed chi enumeration. Builds
  ``[n_atoms, 3]`` coord arrays by applying chi rotations to the
  current residue's coordinates (or its ideal coords).
* :mod:`pairwise_energies` — ``E1[n_res, n_rot]`` one-body and
  ``E2[n_res, n_res, n_rot, n_rot]`` two-body energy tables. Uses a
  shifted-LJ proxy between rotamer atom sets for pair energies +
  rotamer-vs-backbone LJ for one-body. Optionally adds Dunbrack
  ``-log P`` per-rotamer penalties.
* :mod:`dee` — iterated single-rotamer Goldstein Dead-End Elimination.
* :mod:`sim_annealer` — Metropolis SA over the reduced rotamer graph.
* :mod:`packer` — top-level :func:`pack_rotamers` orchestration plus
  the :class:`PackerTask` palette/designability container.

Exact vs. simplified
--------------------

What's exact in this port:

* Rotamer sampler: looks up ``(phi, psi)`` in the real Dunbrack
  library, returns the top-N rotamers ranked by probability, applies
  library chi means to the current backbone via explicit Rodrigues
  rotations (correct sign, correct downstream atom set). Matches the
  upstream top-rotamer set for all 18 rotameric/semirotameric AAs.
* One-body energy: LJ against every backbone atom of every other
  residue (captures backbone clashes — the dominant driver of packing
  at the one-body level).
* Pair energy: shifted + faded LJ evaluated exactly like the upstream
  ``lj_atr_rep`` potential (same sigma, epsilon, d_lin, cpoly_dmin/dmax
  constants), between all heavy-atom pairs of the two rotamers.
* DEE: standard iterated single-rotamer Goldstein criterion; matches
  Desmet–Spriet–Lasters '92.
* SA: Metropolis acceptance with geometric temperature schedule and
  incremental O(n_res) delta-energy per move.

What's simplified:

* The LJ-only pair energy drops electrostatics (``fa_elec``), LK
  desolvation (``fa_lk``), LK-ball bridges, and H-bond directional
  scoring. These add a few percent to typical pair energies; the
  ordering of rotamer options is dominated by the LJ term.
* No count-pair weighting between residues. We assume all inter-
  residue atom pairs have bond separation ≥ 5 (cp weight = 1).
* DEE is single-rotamer only. No Goldstein pairs-DEE, no split-DEE.
* SA is single-chain. No parallel trajectories, no sub-quench step.
* The fallback at the end of :func:`pack_rotamers` re-scores the best
  assignment against the full ``score_function`` and returns the
  starting pose if the optimized pose is worse — a safety net against
  the LJ proxy mis-ordering rotamers for edge cases.

Public API::

    from tmol_jax.pack import sample_rotamers, pack_rotamers, PackerTask
    from tmol_jax.pack import build_pairwise_energies, goldstein_dee, run_sa
"""

from tmol_jax.pack.dee import goldstein_dee
from tmol_jax.pack.packer import PackerTask, pack_rotamers
from tmol_jax.pack.pairwise_energies import (
    PairwiseEnergyTables,
    build_pairwise_energies,
)
from tmol_jax.pack.rotamer_sampler import sample_rotamers
from tmol_jax.pack.sim_annealer import run_sa

__all__ = [
    "sample_rotamers",
    "pack_rotamers",
    "PackerTask",
    "PairwiseEnergyTables",
    "build_pairwise_energies",
    "goldstein_dee",
    "run_sa",
]
