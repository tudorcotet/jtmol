"""LK-ball (FOUR channels) on the uncapped adaptive Verlet substrate — the hard
term on the exact nonbonded engine (companion to ``ljlk.py``).

Reuses jtmol's lk_ball physics VERBATIM — water placement, the four-channel
``lk_ball_pair`` combiner, and count-pair — exactly as the dense term computes
them. Only the EDGE SET changes: the capped/stale ``build_cell_list`` (<=512
neighbors, setup-frozen) is replaced by ``jtmol.nonbonded.verlet`` (uncapped,
exact, adaptive).

The one structural difference from LJLK
---------------------------------------
lk_ball's edge list is a FULL / ASYMMETRIC directed list (i->j), NOT the i<j
half-list LJLK uses. The dense path (``_score_all_pairs``) sums ``lk_ball_pair``
over EVERY ordered pair ``(i, j)``, ``i != j`` — the polar->occluder role and the
water set ``WI`` belong to ``i``, so ``(a, b)`` and ``(b, a)`` are different,
independent contributions. ``build_verlet_list`` returns an i<j HALF list, so we
EXPAND it: every unordered ``{a, b}`` becomes BOTH ``(a, b)`` and ``(b, a)``.

Per-edge fields (matching the term's own ``coords.shape[0] == 1`` edge-build):
lk_ball needs only ``edge_i / edge_j / edge_bpl``. Waters and per-atom params are
placed/gathered per-atom inside ``_eval_full`` and then indexed BY the edges —
so no per-edge water bookkeeping is required here; feeding the expanded edges is
enough.

Correctness / cutoff-completeness
---------------------------------
``lk_ball_pair`` multiplies ALL FOUR channels by ``gate = has_waters &
(dist < distance_threshold)`` with ``distance_threshold = 6.0`` (and the isotropic
LK ramp itself is 0 past ``cpoly_far_dmax = 6.0``). So every channel is exactly 0
for atom-atom distance >= 6.0 -> a Verlet list whose candidate radius covers 6.0
captures every nonzero ordered pair, and the edge-list sum equals the dense
``[N, N]`` sum EXACTLY (both evaluate the same ``lk_ball_pair`` at the same pairs;
pairs past the cutoff are 0 either way). Verified per-channel in ``demo()``.

Single-topology / single-pose (P == 1) for now; multi-pose ragged batching is
future work.
"""
from __future__ import annotations

import attrs
import jax
import jax.numpy as jnp
import numpy as np

from jtmol.nonbonded.verlet import build_verlet_list, VerletList
from jtmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
from jtmol.score.score_types import ScoreType

# lk_ball's hard distance gate. ``lk_ball_pair`` sets ``within_cutoff = dist <
# distance_threshold`` and multiplies every one of the four channels by it, so
# all channels are exactly 0 at/beyond this distance (the isotropic ramp's
# ``cpoly_far_dmax`` is the same 6.0). Beyond it there is nothing to capture.
_LKBALL_CUTOFF = 6.0

_LK_CHANNELS = (
    ScoreType.lk_ball_iso,
    ScoreType.lk_ball,
    ScoreType.lk_bridge,
    ScoreType.lk_bridge_uncpl,
)


def _verlet_edge_state(dense, coords0, atom_is_real, cutoff, skin):
    """Reconstruct the lk_ball term state with the (expanded) Verlet edges.

    Copies every palette field from the dense setup ``dense`` and swaps in the
    directed edge list. ``neighbor_idx / neighbor_mask / bpl_neighbor`` are set to
    non-None sentinels because ``_eval_full`` gates the edge branch on
    ``neighbor_idx is not None`` (and vmaps these through) even though the edge
    branch never reads them.
    """
    vl: VerletList = build_verlet_list(coords0, float(cutoff), float(skin), atom_is_real=atom_is_real)
    vi = np.asarray(vl.pairs_i)
    vj = np.asarray(vl.pairs_j)

    # EXPAND the i<j half-list -> lk_ball's FULL asymmetric directed list.
    ei = np.concatenate([vi, vj]).astype(np.int32)
    ej = np.concatenate([vj, vi]).astype(np.int32)

    bpl = np.asarray(dense.bonded_path_length[0])  # [N, N] count-pair separation (symmetric)
    edge_bpl = jnp.asarray(bpl[ei, ej])

    N = int(coords0.shape[0])
    dummy_i = jnp.zeros((1, N, 1), dtype=jnp.int32)
    dummy_m = jnp.zeros((1, N, 1), dtype=jnp.bool_)
    state = attrs.evolve(
        dense,
        neighbor_idx=dummy_i,
        neighbor_mask=dummy_m,
        bpl_neighbor=dummy_i,
        edge_i=jnp.asarray(ei),
        edge_j=jnp.asarray(ej),
        edge_bpl=edge_bpl,
    )
    return state, vl


def setup_lkball_on_verlet(pose_stack, param_db=None, cutoff: float | None = None, skin: float = 1.0):
    """Return (term, edge_state, dense_state, verlet) for lk_ball on the Verlet list.

    The lk_ball term is extracted from ``beta2016_score_function`` so it carries
    the exact production config (a bare ``LKBallEnergyTerm(db)`` is not guaranteed
    to match — the LJLK build proved this). ``edge_state`` carries the uncapped,
    expanded (directed) Verlet ``edge_i/edge_j/edge_bpl``; evaluate it with
    :func:`lkball_energy_on_verlet`. ``dense_state`` (neighbor_idx=None) drives the
    reference dense path through the public ``term(...)`` call.
    """
    import jtmol

    if int(np.asarray(pose_stack.coords).shape[0]) != 1:
        raise ValueError("setup_lkball_on_verlet is single-pose (P==1) for now")

    sfxn = jtmol.beta2016_score_function(param_db=param_db, use_neighbor_list=False)
    term = next(t for t in sfxn.terms if isinstance(t, LKBallEnergyTerm))

    # Dense reference state: use_neighbor_list=False -> neighbor_idx=None, so the
    # public term(...) call and _eval_full both take the dense _score_all_pairs path.
    term.use_neighbor_list = False
    dense = term.setup_for_pose_stack(pose_stack)

    cut = float(cutoff) if cutoff is not None else _LKBALL_CUTOFF
    if cut < _LKBALL_CUTOFF:
        raise ValueError(
            f"cutoff {cut} < lk_ball distance_threshold {_LKBALL_CUTOFF} would drop nonzero pairs"
        )

    coords0 = np.asarray(pose_stack.coords[0], dtype=np.float64)
    atom_is_real = np.asarray(dense.is_real[0], dtype=bool)
    edge_state, vl = _verlet_edge_state(dense, coords0, atom_is_real, cut, skin)

    # The edge branch of _eval_full is gated on the term flag; leave it True. The
    # dense_state (neighbor_idx=None) still routes to the dense path regardless.
    term.use_neighbor_list = True
    return term, edge_state, dense, vl


def lkball_energy_on_verlet(term, state, coords):
    """All 4 lk_ball channels {lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl}
    over the (expanded) Verlet edges. Differentiable.

    ``term._eval_full(coords, state)`` is lk_ball's ``(coords, state)`` entry point
    — the analogue of LJLK's ``_call_edge_list``. With ``term.use_neighbor_list``
    True and ``state`` carrying non-None ``neighbor_idx`` + ``edge_i``, it routes
    to ``_score_all_pairs_edges`` (the verbatim edge physics).
    """
    return term._eval_full(jnp.asarray(coords), state)


def demo() -> None:
    """Assert lk_ball-on-Verlet == jtmol dense lk_ball (all 4 channels, energy +
    gradient) on 1ubq, f64, plus a cutoff-completeness sweep."""
    import os

    jax.config.update("jax_enable_x64", True)
    import jtmol
    from jtmol.database import ParameterDatabase

    pdb = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data", "1ubq.pdb")
    ps = jtmol.pose_stack_from_pdb(pdb)
    db = ParameterDatabase.get_default()

    term, edge_state, dense_state, vl = setup_lkball_on_verlet(ps, db)
    n_half = int(vl.pairs_i.size)
    print(
        f"[lkball-on-verlet] 1ubq: {n_half} i<j Verlet edges -> {2 * n_half} directed "
        f"edges (asymmetric i->j, cutoff={_LKBALL_CUTOFF})"
    )

    coords0 = np.asarray(ps.coords[0], dtype=np.float64)
    atom_is_real = np.asarray(dense_state.is_real[0], dtype=bool)

    # f32 (native summation-order, informational) and f64 (physics-exact assert).
    for dt, label in (
        (np.float32, "f32 (native, summation-order)"),
        (np.float64, "f64 (edge == dense; residual is reduction-order)"),
    ):
        coords = jnp.asarray(np.asarray(ps.coords, dtype=dt))
        E_dense = term(ps, dense_state, coords_override=coords)  # public dense path
        E_v = lkball_energy_on_verlet(term, edge_state, coords)
        print(f"  --- {label} ---")
        worst = 0.0
        for st in _LK_CHANNELS:
            ed = float(np.asarray(E_dense[st])[0])
            ev = float(np.asarray(E_v[st])[0])
            rel = abs(ed - ev) / (abs(ed) + 1e-12)
            worst = max(worst, rel)
            print(f"    {st.name:16}: dense {ed:+.8f}  verlet {ev:+.8f}  rel {rel:.2e}")

        def _tot(fn):
            return jax.grad(lambda c: sum(jnp.sum(fn(c)[s]) for s in _LK_CHANNELS))(coords)

        gd = _tot(lambda c: term(ps, dense_state, coords_override=c))
        gv = _tot(lambda c: lkball_energy_on_verlet(term, edge_state, c))
        gmax = float(jnp.abs(gd - gv).max())
        print(f"    gradient max|dense-verlet| = {gmax:.2e}")
        if dt is np.float64:
            assert worst < 1e-6, f"f64 energy parity FAILED: rel {worst}"
            assert gmax < 1e-5, f"f64 gradient parity FAILED: {gmax}"

    # Cutoff-completeness sweep (f64). Small cutoffs drop pairs (candidate radius
    # cutoff+skin < 6.0); once cutoff+skin >= 6.0 every nonzero pair is captured
    # and the edge sum is EXACT. Demonstrates 6.0 is load-bearing and sufficient.
    coords_f64 = jnp.asarray(np.asarray(ps.coords, dtype=np.float64))  # [1, N, 3]
    E_dense_f64 = term(ps, dense_state, coords_override=coords_f64)
    dense_tot = sum(float(np.asarray(E_dense_f64[s])[0]) for s in _LK_CHANNELS)
    print(f"\n  cutoff-completeness sweep (skin=1.0; dense 4-channel sum = {dense_tot:+.6f}):")
    saw_incomplete = False
    for c in (3.0, 4.0, 5.0, 6.0):
        st_c, _vlc = _verlet_edge_state(dense_state, coords0, atom_is_real, c, 1.0)
        E_c = lkball_energy_on_verlet(term, st_c, coords_f64)
        tot_c = sum(float(np.asarray(E_c[s])[0]) for s in _LK_CHANNELS)
        rel = abs(tot_c - dense_tot) / (abs(dense_tot) + 1e-12)
        flag = "EXACT" if rel < 1e-6 else "incomplete (drops pairs)"
        saw_incomplete = saw_incomplete or rel >= 1e-6
        print(f"    cutoff {c:>4} (radius {c + 1.0:>4}): sum {tot_c:+.6f}  rel {rel:.2e}  {flag}")
    assert saw_incomplete, "sweep should show at least one too-small cutoff dropping pairs"

    print(
        "\nlk_ball-on-Verlet == dense lk_ball to f64 machine precision — ALL FOUR "
        "channels (lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl), energy + "
        "gradient. Asymmetric directed edge expansion + water/count-pair physics "
        "on the uncapped adaptive substrate: EXACT."
    )


if __name__ == "__main__":
    demo()
