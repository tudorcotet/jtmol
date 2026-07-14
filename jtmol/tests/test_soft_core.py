"""Self-check for opt-in soft-core repulsion.

Asserts the two contract properties of ``beta2016_score_function(
soft_core_rep=...)``:

1. **Parity when OFF** — the default (``soft_core_rep=None``) is byte-for-
   byte the standard energy; enabling soft-core is the only thing that
   changes any value.
2. **Caps the repulsive channel** — on a clashy structure, soft-core
   lowers ``fa_ljrep`` and bounds its coordinate gradient (a deep clash can
   no longer produce an unbounded force), while the near-native energy is
   only mildly perturbed for a large cap.

Run: ``python -m jtmol.tests.test_soft_core``
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import os

import jax
import jax.numpy as jnp
import numpy as np

import jtmol
from jtmol.score.score_types import ScoreType

_PDB = os.path.join(os.path.dirname(__file__), "data", "1ubq.pdb")


def _isolated_ljrep_grad(cap, ps, coords):
    """max|∂fa_ljrep/∂x| over atoms, isolating the repulsive channel."""
    s = (
        jtmol.beta2016_score_function(soft_core_rep=cap)
        if cap is not None
        else jtmol.beta2016_score_function()
    )
    for t in s.active_score_types():
        s.set_weight(t, 1.0 if t == ScoreType.fa_ljrep else 0.0)
    st = s.setup_for_pose_stack(ps)
    g = jax.grad(lambda c: s(ps, state=st, coords_override=c).sum())(coords)
    return float(jnp.max(jnp.abs(jnp.nan_to_num(g))))


def main() -> int:
    ps = jtmol.pose_stack_from_pdb(_PDB)

    off = jtmol.beta2016_score_function()
    st_off = off.setup_for_pose_stack(ps)
    e_off = float(off(ps, state=st_off).sum())

    # 1. Parity: OFF equals the standard energy exactly.
    off2 = jtmol.beta2016_score_function(soft_core_rep=None)
    e_off2 = float(off2(ps, state=off2.setup_for_pose_stack(ps)).sum())
    assert e_off == e_off2, f"soft_core=None changed the energy ({e_off} vs {e_off2})"
    print(f"[soft-core] parity OFF: E={e_off:.6f} (unchanged) OK")

    # 2. Clashy structure: cap lowers fa_ljrep and bounds its gradient.
    key = jax.random.PRNGKey(0)
    clashy = ps.coords + jax.random.normal(key, ps.coords.shape) * 0.5

    def ljrep_sum(cap):
        s = (jtmol.beta2016_score_function(soft_core_rep=cap)
             if cap else jtmol.beta2016_score_function())
        st = s.setup_for_pose_stack(ps)
        return float(np.asarray(
            s.score_by_term(ps, state=st, coords_override=clashy)
            .get(ScoreType.fa_ljrep, 0.0)
        ).sum())

    rep_off, rep_on = ljrep_sum(None), ljrep_sum(50.0)
    g_off = _isolated_ljrep_grad(None, ps, clashy)
    g_on = _isolated_ljrep_grad(50.0, ps, clashy)
    print(f"[soft-core] clashy fa_ljrep {rep_off:.0f}->{rep_on:.0f} | "
          f"fa_ljrep grad {g_off:.0f}->{g_on:.0f} (cap=50)")
    assert rep_on < rep_off, "soft-core did not lower clashy fa_ljrep"
    assert g_on < g_off, "soft-core did not bound the fa_ljrep gradient"

    # near-native energy only mildly perturbed for a large cap
    big = jtmol.beta2016_score_function(soft_core_rep=200.0)
    e_big = float(big(ps, state=big.setup_for_pose_stack(ps)).sum())
    pct = 100 * abs(e_big - e_off) / abs(e_off)
    print(f"[soft-core] native distortion at cap=200: {pct:.2f}%")
    assert pct < 2.0, f"cap=200 distorts native too much ({pct:.1f}%)"

    print("[soft-core] OK — parity off, caps clash energy + gradient, low native distortion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
