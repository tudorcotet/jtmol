"""Jit-safety regression: the energy and its gradient must stay finite
under jit on strained / degenerate geometry.

Motivation: the disulfide term shipped three separate jit-only bugs
(``jnp.linalg.norm`` at a zero vector, a masked-pair geometry leak, and a
``1 - erf`` catastrophic cancellation whose ``log`` went to ``-inf`` under
XLA fusion). Each produced a finite *eager* value but a non-finite *jit*
value on strained inputs (badly-folded predictions, coincident atoms),
which silently stalled minimisation and made structures unscoreable. The
other terms avoid this by routing every ``sqrt``/``norm``/``arccos``
through the double-``where`` / clipped helpers in ``score.common.geom``.

This test locks that property in: for a plain protein and a disulfide
protein, under (a) heavy Gaussian perturbation, (b) exact atom
coincidence, and (c) exact collinearity, every active beta2016 channel's
jitted energy and gradient must be finite. A regression in any term's
degenerate-geometry handling fails here.

Run: ``python -m jtmol.tests.test_jit_safety``
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np

import jtmol

_DATA = os.path.join(os.path.dirname(__file__), "data")
# 1ubq: plain protein. 1CGI: has disulfides (the term that was buggy) and
# was itself unscoreable before the erfc fix.
_PDBS = {"1ubq": os.path.join(_DATA, "1ubq.pdb"),
         "1CGI": os.path.join(_DATA, "1CGI.pdb")}


def _degenerate(coords: np.ndarray, mode: str) -> jnp.ndarray:
    c = coords.copy()  # [1, N, 3]
    if mode == "perturb":
        rng = np.random.default_rng(0)
        c = c + rng.standard_normal(c.shape).astype(c.dtype) * 1.0
    elif mode == "coincident":
        c[0, 5:12] = c[0, 5]  # several distinct atoms onto one point (dist=0)
    elif mode == "collinear":
        b = c[0, 20]
        for k in range(1, 4):
            c[0, 20 + k] = b + np.array([float(k), 0.0, 0.0], dtype=c.dtype)
    return jnp.asarray(c)


def _energy_and_grad_finite(pid: str, path: str) -> list[str]:
    ps = jtmol.pose_stack_from_pdb(path)
    sf = jtmol.beta2016_score_function()
    state = sf.setup_for_pose_stack(ps)

    def energy(coords):
        return sf(ps, state=state, coords_override=coords).sum()

    e_jit = jax.jit(energy)
    g_jit = jax.jit(jax.grad(energy))
    base = np.asarray(ps.coords)
    failures = []
    for mode in ["perturb", "coincident", "collinear"]:
        c = _degenerate(base, mode)
        E = float(e_jit(c))
        g = g_jit(c)
        if not np.isfinite(E):
            failures.append(f"{pid}/{mode}: jit energy non-finite ({E})")
        n_bad = int(jnp.sum(~jnp.isfinite(g)))
        if n_bad:
            failures.append(f"{pid}/{mode}: jit gradient has {n_bad} non-finite entries")
    return failures


def main() -> int:
    all_failures: list[str] = []
    for pid, path in _PDBS.items():
        f = _energy_and_grad_finite(pid, path)
        status = "OK" if not f else "FAIL"
        print(f"[jit-safety] {pid}: {status}")
        all_failures.extend(f)
    if all_failures:
        print("[jit-safety] FAILURES:")
        for m in all_failures:
            print("  " + m)
        return 1
    print("[jit-safety] OK — all terms finite under perturb/coincident/collinear.")
    return 0


def test_jit_safety() -> None:
    assert not (_energy_and_grad_finite("1ubq", _PDBS["1ubq"])
                + _energy_and_grad_finite("1CGI", _PDBS["1CGI"]))


if __name__ == "__main__":
    raise SystemExit(main())
