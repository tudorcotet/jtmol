"""fa_elec (Coulomb electrostatics) on the uncapped adaptive Verlet substrate —
the second real term on the exact nonbonded engine.

Mirrors the VERIFIED template ``jtmol.nonbonded.ljlk``: jtmol's elec physics and
count-pair are reused VERBATIM; only the EDGE SET changes. The capped/stale
``build_cell_list`` (<=512 neighbors, setup-frozen) is replaced by
``jtmol.nonbonded.verlet`` (uncapped, exact, adaptive).

Count-pair: elec uses a DIFFERENT representation than LJLK internally — a
rep-remapped bonded-path-length (``pair_bonded_path_length``: each atom's path to
its count-pair *representative* plus the inter-block ``inter_block_bondsep`` link,
not the raw all-bonds distance LJLK gathers). But that difference is entirely in
how the ``[N,N] bonded_path_length`` matrix is BUILT. Per edge it is the same
single field as LJLK — the matrix value at ``(i, j)`` — so we reuse ``dense``'s
already-remapped matrix and gather it at the Verlet ``(i, j)`` pairs. Nothing about
the remap logic is touched or reimplemented here.

Cutoff: elec's potential is exactly 0 at/beyond ``global_params.max_dis`` (== 5.5
in the default params; potentials.py region dispatch leaves E=0 for
``dist >= max_dis``). NOTE this differs from LJLK, whose max_dis is 6.0, and from
the elec term's own ``neighbor_list_cutoff`` default of 6.0 (a loose superset). We
cut at max_dis=5.5 — the smallest radius that drops NO nonzero pair — so the
edge-list sum equals the dense ``[N,N]`` sum exactly. Proven by the cutoff sweep in
``demo()`` (energy is invariant as the cutoff grows; larger radii only add E=0
edges).

Single-topology / single-pose (P==1) for now; multi-pose ragged batching is future
work.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jtmol.nonbonded.verlet import build_verlet_list, VerletList
from jtmol.score.elec.elec_energy_term import ElecEnergyTerm, _ElecState
from jtmol.score.score_types import ScoreType

# _ElecState field names (reconstruct with Verlet edges swapped in). The dense
# state has neighbor_idx/neighbor_mask/bpl_neighbor == None (dense setup); they
# stay None on the edge path — `_call_edge_list` never reads them.
_STATE_FIELDS = (
    "charges", "bonded_path_length", "atom_is_real",
    "D", "D0", "S", "min_dis", "max_dis",
    "neighbor_idx", "neighbor_mask", "bpl_neighbor",
    "edge_i", "edge_j", "edge_bpl",
)


def setup_elec_on_verlet(pose_stack, param_db=None, cutoff: float | None = None, skin: float = 1.0):
    """Return (term, edge_state, dense_state, verlet) for fa_elec on the Verlet list.

    The elec term is extracted from ``beta2016_score_function`` so it carries the
    exact production config — a bare ``ElecEnergyTerm(db)`` can differ (this is the
    same trap LJLK hit on fa_ljrep). ``edge_state`` carries the uncapped Verlet
    ``edge_i/edge_j/edge_bpl``; call ``term._call_edge_list(coords, edge_state)``.
    """
    import jtmol
    if int(np.asarray(pose_stack.coords).shape[0]) != 1:
        raise ValueError("setup_elec_on_verlet is single-pose (P==1) for now")
    sfxn = jtmol.beta2016_score_function(param_db=param_db, use_neighbor_list=False)
    term = next(t for t in sfxn.terms if isinstance(t, ElecEnergyTerm))
    term.use_neighbor_list = False  # dense setup -> gives charges + bonded_path_length
    dense = term.setup_for_pose_stack(pose_stack)

    # elec is EXACTLY 0 beyond max_dis (5.5), unlike LJLK (6.0). Cut there so no
    # nonzero pair is dropped; a larger cutoff only adds E=0 edges (see demo sweep).
    max_dis = float(term.global_params.max_dis)
    cut = float(cutoff) if cutoff is not None else max_dis
    if cut < max_dis:
        raise ValueError(f"cutoff {cut} < elec max_dis {max_dis} would drop nonzero pairs")

    coords0 = np.asarray(pose_stack.coords[0], dtype=np.float64)
    atom_is_real = np.asarray(dense.atom_is_real[0], dtype=bool)
    vl: VerletList = build_verlet_list(coords0, cut, skin, atom_is_real=atom_is_real)
    vi = np.asarray(vl.pairs_i); vj = np.asarray(vl.pairs_j)

    # Same per-edge gather as LJLK: the [N,N] count-pair matrix at (i, j). For elec
    # this matrix is the rep-remapped path length built by pair_bonded_path_length;
    # we take it straight from `dense` — the remap is never re-derived here.
    bpl = np.asarray(dense.bonded_path_length)[0]  # [N,N]
    edge_bpl = jnp.asarray(bpl[vi, vj])

    fields = {name: getattr(dense, name) for name in _STATE_FIELDS}
    fields["edge_i"] = jnp.asarray(vi.astype(np.int32))
    fields["edge_j"] = jnp.asarray(vj.astype(np.int32))
    fields["edge_bpl"] = edge_bpl
    state = _ElecState(**fields)
    return term, state, dense, vl


def elec_energy_on_verlet(term, state, coords):
    """fa_elec over the Verlet edges, as {ScoreType.fa_elec: [P]}. Differentiable."""
    return term._call_edge_list(coords, state)


def demo() -> None:
    """Assert elec-on-Verlet == jtmol dense elec (energy + gradient) on 1ubq, f64,
    plus a cutoff-completeness sweep."""
    jax.config.update("jax_enable_x64", True)
    import os
    import jtmol
    from jtmol.database import ParameterDatabase

    pdb = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data", "1ubq.pdb")
    ps = jtmol.pose_stack_from_pdb(pdb)
    db = ParameterDatabase.get_default()
    st = ScoreType.fa_elec

    term, state, dense_state, vl = setup_elec_on_verlet(ps, db)
    dense_term = term  # same beta2016-configured term for the dense reference
    max_dis = float(term.global_params.max_dis)
    print(f"[elec-on-verlet] 1ubq: {int(vl.pairs_i.size)} Verlet edges "
          f"(uncapped, cutoff=max_dis={max_dis})")

    # f32 residuals are native summation-order (informational); the exactness
    # assertion is in f64, where only reduction order between the dense [N,N] path
    # and the flat edge list separates them (~1e-8 energy, ~1e-6 grad on CPU).
    for dt, label in ((np.float32, "f32 (native, summation-order)"),
                      (np.float64, "f64 (edge == dense; residual is reduction-order)")):
        coords = jnp.asarray(np.asarray(ps.coords, dtype=dt))
        E_dense = dense_term(ps, dense_state, coords_override=coords)
        E_v = elec_energy_on_verlet(term, state, coords)
        ed = float(np.asarray(E_dense[st])[0]); ev = float(np.asarray(E_v[st])[0])
        rel = abs(ed - ev) / (abs(ed) + 1e-12)
        print(f"  --- {label} ---")
        print(f"    {st.name:10}: dense {ed:+.8f}  verlet {ev:+.8f}  rel {rel:.2e}")
        gd = jax.grad(lambda c: jnp.sum(dense_term(ps, dense_state, coords_override=c)[st]))(coords)
        gv = jax.grad(lambda c: jnp.sum(elec_energy_on_verlet(term, state, c)[st]))(coords)
        gmax = float(jnp.abs(gd - gv).max())
        print(f"    gradient max|dense-verlet| = {gmax:.2e}")
        if dt is np.float64:
            assert rel < 1e-6, f"f64 energy parity FAILED: rel {rel}"
            assert gmax < 1e-5, f"f64 gradient parity FAILED: {gmax}"

    # Cutoff-completeness sweep: the edge-list energy must NOT change as the cutoff
    # grows past max_dis. If max_dis dropped any nonzero pair, a larger radius would
    # re-include it and shift E by >> 1e-6; it does not, because elec is exactly 0
    # beyond max_dis so every extra edge contributes 0.
    print(f"  --- cutoff sweep (must be invariant above max_dis={max_dis}) ---")
    coords64 = jnp.asarray(np.asarray(ps.coords, dtype=np.float64))
    E_ref = None
    for cut in (max_dis, max_dis + 1.0, max_dis + 2.0):
        _t, s_c, _d, vl_c = setup_elec_on_verlet(ps, db, cutoff=cut)
        e_c = float(np.asarray(elec_energy_on_verlet(term, s_c, coords64)[st])[0])
        if E_ref is None:
            E_ref = e_c
        rel = abs(e_c - E_ref) / (abs(E_ref) + 1e-12)
        print(f"    cutoff {cut:.1f}: {int(vl_c.pairs_i.size):6d} edges  E {e_c:+.8f}  rel-to-min-cut {rel:.2e}")
        assert rel < 1e-6, f"cutoff {cut} changed the energy (rel {rel}) -> max_dis dropped a nonzero pair"

    print("\nelec-on-Verlet == dense elec to f64 machine precision (energy + gradient), "
          "and invariant to cutoff growth above max_dis. fa_elec on the uncapped "
          "adaptive substrate: EXACT.")


if __name__ == "__main__":
    demo()
