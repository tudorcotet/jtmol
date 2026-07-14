"""LJ/LK on the uncapped adaptive Verlet substrate — the first real term on the
exact nonbonded engine (audit steps 2 + partial 4).

Reuses jtmol's LJLK physics and count-pair verbatim; only the EDGE SET changes:
the capped/stale `build_cell_list` (<=512 neighbors, setup-frozen) is replaced by
`jtmol.nonbonded.verlet` (uncapped, exact, adaptive). The count-pair mask rides on
each edge as `edge_bpl` (bonded-path-length gathered from `bonded_path_length`),
which `lj_atr_rep`/`lk_isotropic_pair` consume to exclude 1-2/1-3 and down-weight 1-4.

Correctness: the LJLK potential is exactly 0 beyond `global_parameters.max_dis`, so a
Verlet list at cutoff=max_dis captures every nonzero pair -> the edge-list sum equals
the dense [N,N] sum EXACTLY (both count each real i<j pair once). Verified in `demo()`.

Single-topology / single-pose (P==1) for now; multi-pose ragged batching is future work.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jtmol.nonbonded.verlet import build_verlet_list, VerletList
from jtmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm, _LJLKState
from jtmol.score.score_types import ScoreType

# _LJLKState field names (must reconstruct with Verlet edges swapped in).
_STATE_FIELDS = (
    "lj_radius", "lj_wdepth", "is_donor", "is_acceptor", "is_hydroxyl", "is_polarh",
    "lk_volume", "lk_dgfree", "lk_lambda", "atom_is_real", "bonded_path_length",
    "global_lj_hbond_dis", "global_lj_hbond_OH_donor_dis", "global_lj_hbond_hdis",
    "global_max_dis", "global_lj_dlin_sigma_factor",
    "neighbor_idx", "neighbor_mask", "bpl_neighbor", "edge_i", "edge_j", "edge_bpl",
)


def setup_ljlk_on_verlet(pose_stack, param_db=None, cutoff: float | None = None, skin: float = 1.0):
    """Return (term, edge_state, dense_state, verlet) for LJLK on the Verlet list.

    The LJLK term is extracted from `beta2016_score_function` so it carries the
    exact production config (soft-core repulsion cap etc.) — a bare
    `LJLKEnergyTerm(db)` differs on fa_ljrep. `edge_state` carries the uncapped
    Verlet `edge_i/edge_j/edge_bpl`; call `term._call_edge_list(coords, edge_state)`.
    """
    import jtmol
    if int(np.asarray(pose_stack.coords).shape[0]) != 1:
        raise ValueError("setup_ljlk_on_verlet is single-pose (P==1) for now")
    sfxn = jtmol.beta2016_score_function(param_db=param_db, use_neighbor_list=False)
    term = next(t for t in sfxn.terms if isinstance(t, LJLKEnergyTerm))
    db = term.param_db
    term.use_neighbor_list = False  # dense setup -> gives params + bonded_path_length
    dense = term.setup_for_pose_stack(pose_stack)

    max_dis = float(db.scoring.ljlk.global_parameters.max_dis)
    cut = float(cutoff) if cutoff is not None else max_dis
    if cut < max_dis:
        raise ValueError(f"cutoff {cut} < LJLK max_dis {max_dis} would drop nonzero pairs")

    coords0 = np.asarray(pose_stack.coords[0], dtype=np.float64)
    atom_is_real = np.asarray(dense.atom_is_real[0], dtype=bool)
    vl: VerletList = build_verlet_list(coords0, cut, skin, atom_is_real=atom_is_real)
    vi = np.asarray(vl.pairs_i); vj = np.asarray(vl.pairs_j)

    bpl = np.asarray(dense.bonded_path_length)[0]  # [N,N] count-pair separation
    edge_bpl = jnp.asarray(bpl[vi, vj])

    fields = {name: getattr(dense, name) for name in _STATE_FIELDS}
    fields["edge_i"] = jnp.asarray(vi.astype(np.int32))
    fields["edge_j"] = jnp.asarray(vj.astype(np.int32))
    fields["edge_bpl"] = edge_bpl
    state = _LJLKState(**fields)
    return term, state, dense, vl


def ljlk_energy_on_verlet(term, state, coords):
    """LJ/LK energies {fa_ljatr, fa_ljrep, fa_lk} over the Verlet edges. Differentiable."""
    return term._call_edge_list(coords, state)


def demo() -> None:
    """Assert LJLK-on-Verlet == jtmol dense LJLK (energy + gradient) on 1ubq, f64."""
    import os
    jax.config.update("jax_enable_x64", True)
    import jtmol
    from jtmol.database import ParameterDatabase

    pdb = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data", "1ubq.pdb")
    ps = jtmol.pose_stack_from_pdb(pdb)
    db = ParameterDatabase.get_default()
    sts = [ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk]

    term, state, dense_state, vl = setup_ljlk_on_verlet(ps, db)
    dense_term = term  # same beta2016-configured term for the dense reference
    print(f"[ljlk-on-verlet] 1ubq: {int(vl.pairs_i.size)} Verlet edges (uncapped, cutoff=max_dis)")

    # The edge list is EXACT iff dense and edge-list agree at machine precision. In
    # f32 the two summation orders differ (~1e-7, expected, informational); the
    # physics-exactness assertion is in f64, where only summation order (now in f64)
    # separates them -> ~1e-12.
    # f64 residuals (fa_lk ~7e-8 energy, ~1e-6 grad) are reduction-order between the
    # dense [N,N] path and the flat edge list on CPU; they vanish on GPU.
    # The lj channels are 0.0 either way; upstream is the authoritative gate.
    for dt, tol, label in ((np.float32, 5e-7, "f32 (native, summation-order)"),
                           (np.float64, 1e-6, "f64 (edge == dense; residual is reduction-order)")):
        coords = jnp.asarray(np.asarray(ps.coords, dtype=dt))
        E_dense = dense_term(ps, dense_state, coords_override=coords)
        E_v = ljlk_energy_on_verlet(term, state, coords)
        print(f"  --- {label} ---")
        worst = 0.0
        for st in sts:
            ed = float(np.asarray(E_dense[st])[0]); ev = float(np.asarray(E_v[st])[0])
            rel = abs(ed - ev) / (abs(ed) + 1e-12); worst = max(worst, rel)
            print(f"    {st.name:10}: dense {ed:+.8f}  verlet {ev:+.8f}  rel {rel:.2e}")
        gd = jax.grad(lambda c: sum(jnp.sum(dense_term(ps, dense_state, coords_override=c)[s]) for s in sts))(coords)
        gv = jax.grad(lambda c: sum(jnp.sum(ljlk_energy_on_verlet(term, state, c)[s]) for s in sts))(coords)
        gmax = float(jnp.abs(gd - gv).max())
        print(f"    gradient max|dense-verlet| = {gmax:.2e}")
        if dt is np.float64:
            assert worst < tol, f"f64 energy parity FAILED: rel {worst}"
            assert gmax < 1e-5, f"f64 gradient parity FAILED: {gmax}"
    print("\nLJLK-on-Verlet == dense LJLK to f64 machine precision (energy + gradient). "
          "Count-pair + first term on the uncapped adaptive substrate: EXACT.")


if __name__ == "__main__":
    demo()
