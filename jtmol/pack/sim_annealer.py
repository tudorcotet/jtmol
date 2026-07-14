"""Simulated annealing over a pre-computed rotamer graph.

After DEE has pruned the rotamer palette, the packing problem is
exactly a graph MAP-inference instance with per-node costs ``E1`` and
per-edge costs ``E2``. The upstream engine runs a highly tuned GPU
kernel (``pack_anneal`` in ``tmol/pack/compiled``) that trajectorizes
500 hi-temp chains + 10 lo-temp seeds per top chain. We implement a
pragmatic single-chain Metropolis SA that is simpler to reason about
and sufficient for warming-up the pose before minimisation.

Exact vs. simplified:

* Exact — Metropolis acceptance, incremental delta-energy computation
  (O(n_res) per move, not O(n_res^2)), best-energy tracking across the
  trajectory, geometric temperature schedule from ``T_hi`` to
  ``T_lo``.
* Simplified — single-chain (no parallel trajectories), no
  sub-quench, random uniform move proposals only (no "pick low-energy
  rotamer first" heuristic).
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np


def run_sa(
    E1: np.ndarray,  # [n_res, max_rot]
    E2: np.ndarray,  # [n_res, n_res, max_rot, max_rot]
    keep: np.ndarray,  # [n_res, max_rot] bool
    n_rot: np.ndarray,  # [n_res] int
    designable: np.ndarray,  # [n_res] bool
    n_steps: int = 1000,
    T_hi: float = 100.0,
    T_lo: float = 0.3,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Run Metropolis SA over the rotamer graph.

    Returns
    -------
    best_assignment : np.ndarray [n_res] int32
        The rotamer index chosen at each residue (``-1`` for residues
        with no kept rotamers).
    best_energy : float
        Total (one-body + two-body) energy of the best assignment.
    """
    n_res = E1.shape[0]
    rng = random.Random(seed)

    # Build per-residue list of surviving rotamer indices.
    surviving = []
    for i in range(n_res):
        n = int(n_rot[i])
        s = np.nonzero(keep[i, :n])[0].tolist()
        surviving.append(s)

    # Initial assignment: lowest-E1 surviving rotamer at each residue.
    assignment = np.zeros(n_res, dtype=np.int32)
    for i in range(n_res):
        if len(surviving[i]) == 0:
            assignment[i] = -1
            continue
        s = np.array(surviving[i], dtype=np.int32)
        best_local = s[int(np.argmin(E1[i, s]))]
        assignment[i] = int(best_local)

    def total_energy(a: np.ndarray) -> float:
        """Sum of E1 and E2 over the chosen assignment."""
        e = 0.0
        for i in range(n_res):
            if a[i] < 0:
                continue
            e += float(E1[i, a[i]])
            for j in range(i + 1, n_res):
                if a[j] < 0:
                    continue
                e += float(E2[i, j, a[i], a[j]])
        return e

    current_E = total_energy(assignment)
    best = assignment.copy()
    best_E = current_E

    # Indices eligible for proposal (>1 surviving rotamer and designable).
    movable = [
        i for i in range(n_res)
        if bool(designable[i]) and len(surviving[i]) > 1
    ]
    if not movable:
        return best, best_E

    # Temperature schedule: geometric from T_hi to T_lo.
    if n_steps > 1:
        log_ratio = math.log(T_lo / T_hi) / (n_steps - 1)
    else:
        log_ratio = 0.0

    for step in range(n_steps):
        T = T_hi * math.exp(log_ratio * step)
        i = rng.choice(movable)
        cur = int(assignment[i])
        alternatives = [r for r in surviving[i] if r != cur]
        if not alternatives:
            continue
        proposed = rng.choice(alternatives)

        # Incremental delta energy: replacing rotamer `cur` with `proposed`
        # at residue i affects E1[i] + sum_{j!=i} E2[i,j,*,a[j]].
        d = float(E1[i, proposed] - E1[i, cur])
        for j in range(n_res):
            if j == i or assignment[j] < 0:
                continue
            d += float(E2[i, j, proposed, assignment[j]] - E2[i, j, cur, assignment[j]])

        if d <= 0.0 or rng.random() < math.exp(-d / max(T, 1e-6)):
            assignment[i] = proposed
            current_E += d
            if current_E < best_E:
                best = assignment.copy()
                best_E = current_E

    return best, best_E


__all__ = ["run_sa"]
