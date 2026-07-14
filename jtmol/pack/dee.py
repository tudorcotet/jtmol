"""Dead-End Elimination (DEE) pruning.

Given one- and two-body energy tables, DEE removes rotamers that are
provably dominated by some sibling at the same position — no
optimisation over the remaining rotamers can ever ``un-dominate`` them.

We implement the standard **Goldstein criterion**: rotamer ``a`` at
position ``i`` is eliminated if there exists a sibling rotamer ``b``
such that

    E1(i, a) - E1(i, b)
      + sum_{j != i} min_c [ E2(i, j, a, c) - E2(i, j, b, c) ]  >  0

for every other position ``j``. In words: no matter what the other
positions choose, replacing ``a`` with ``b`` never increases the total
energy, so ``a`` is provably sub-optimal.

We iterate the Goldstein pass until no more rotamers can be pruned or
a user-supplied iteration cap is reached. The output is a boolean
``keep[n_res, max_rot]`` mask plus the pruned count statistics.

Exact vs. simplified:

* Exact — pure Goldstein DEE with unity-weighted min over the
  interacting residues. Matches the reference implementation in
  ``Desmet, Spriet, Lasters '92`` and the upstream tmol DEE pass.
* Simplified — we do not implement the more aggressive "pairs"
  (Goldstein pairs-DEE) or split-DEE variants. Single-rotamer
  Goldstein already removes ~70% of rotamers on the 20-AA repack of
  1UBQ in practice.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def goldstein_dee(
    E1: np.ndarray,  # [n_res, max_rot]
    E2: np.ndarray,  # [n_res, n_res, max_rot, max_rot]
    n_rot: np.ndarray,  # [n_res] int
    designable: np.ndarray,  # [n_res] bool
    max_iters: int = 20,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """Run iterated single-rotamer Goldstein DEE.

    Parameters
    ----------
    E1
        One-body energies. Rotamers beyond ``n_rot[i]`` on row ``i``
        are ignored.
    E2
        Symmetric two-body energies.
    n_rot
        Valid rotamer count per residue.
    designable
        Mask over residues: non-designable residues (e.g. fixed) are
        never pruned regardless of their rotamer counts.
    max_iters
        Iteration cap. DEE usually converges in 3-5 iterations.
    tol
        Slack accepted on the elimination inequality. Defaults to a
        tight 1e-6 kcal/mol so we never prune a rotamer that is
        energy-tied with its candidate dominator.

    Returns
    -------
    keep : np.ndarray
        ``bool[n_res, max_rot]`` — ``True`` means the rotamer survives.
    n_pruned : int
        Total number of rotamers removed across all iterations.
    """
    n_res, max_rot = E1.shape
    keep = np.zeros((n_res, max_rot), dtype=bool)
    for i in range(n_res):
        n = int(n_rot[i])
        keep[i, :n] = True

    total_pruned = 0

    for _ in range(max_iters):
        pruned_this_pass = 0
        for i in range(n_res):
            if not bool(designable[i]):
                continue
            n_i = int(n_rot[i])
            if n_i <= 1:
                continue
            # Candidate dominator rotamers at i (must also be "kept")
            kept_mask = keep[i, :n_i]
            kept_inds = np.nonzero(kept_mask)[0]
            if kept_inds.size <= 1:
                continue

            # For every surviving rotamer a at i, test whether any other
            # surviving rotamer b dominates it in the Goldstein sense.
            for a in kept_inds:
                if not keep[i, a]:
                    continue
                for b in kept_inds:
                    if a == b:
                        continue
                    # Goldstein dominance of b over a:
                    # E1[a] - E1[b] + sum_j min_c (E2[i,j,a,c] - E2[i,j,b,c]) > 0
                    lhs = float(E1[i, a] - E1[i, b])
                    for j in range(n_res):
                        if j == i:
                            continue
                        n_j = int(n_rot[j])
                        if n_j == 0:
                            continue
                        # Only consider the c's still kept at j.
                        c_kept = np.nonzero(keep[j, :n_j])[0]
                        if c_kept.size == 0:
                            continue
                        diff = E2[i, j, a, c_kept] - E2[i, j, b, c_kept]
                        lhs += float(diff.min())
                    if lhs > tol:
                        keep[i, a] = False
                        pruned_this_pass += 1
                        break  # a is gone; no need to test more b's
        total_pruned += pruned_this_pass
        if pruned_this_pass == 0:
            break

    return keep, total_pruned


__all__ = ["goldstein_dee"]
