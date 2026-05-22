"""Per-term finite-difference gradient checks for tmol_jax.

Upstream tmol validates term-by-term derivatives with
``torch.autograd.gradcheck``. JAX ships ``jax.test_util.check_grads`` but
that wraps the same idea around a jvp/vjp pair — we want a direct central
finite-difference check against ``jax.grad`` so we can point at the
specific coordinate entries that disagree.

Design
------
For each :class:`~tmol_jax.score.ScoreType` that the beta2016 score
function weights, we:

1. Build a fresh :class:`ScoreFunction` with **only** that score type
   enabled at weight 1.0. Sibling score types owned by the same
   underlying term (e.g. the five cart sub-terms that share one
   :class:`CartBondedEnergyTerm`) are zeroed out so their energies do
   not contaminate the finite difference.
2. Run the pose-stack setup and compute the analytical gradient via
   ``jax.grad`` at **float64** precision. Upstream uses ``float64`` for
   gradcheck and so do we: float32 gives ~1e-3 relative truncation error
   on a ``dE/dx ~ O(1)`` signal, which swamps any ``atol=1e-3`` check.
3. Pick ``N_SAMPLES`` random ``(pose, atom, axis)`` entries from the
   coordinate tensor where the analytical gradient magnitude is above
   the :data:`GRAD_MAG_FLOOR` threshold. We do this because most padded
   or non-participating atoms have zero gradient — including them would
   dilute the test without adding coverage. When a term has too few
   above-threshold gradients (e.g. ``lk_bridge`` on 1ubq touches only a
   handful of atoms) we fall back to the largest-magnitude entries.
4. For each sampled entry, compute ``(E(+eps) - E(-eps)) / (2*eps)``
   with ``eps=1e-4`` and compare against the analytical entry. We use a
   mixed-tolerance ``abs(a - fd) < max(atol, rtol * abs(fd))`` check
   (``atol=1e-3``, ``rtol=1e-2``) — the same shape as
   ``torch.autograd.gradcheck`` minus the Jacobian-product combinatorics.

Terms covered
-------------
The beta2016 preset weights 21 score types. We skip:

* :data:`ScoreType.constraint` — weight is zero on a fresh pose stack
  (no constraints registered), so there's nothing to differentiate.
* :data:`ScoreType.ref` — a per-residue constant with zero gradient
  everywhere, so there are no above-threshold entries to sample.

The remaining 20 terms (with ``disulfide`` returning zero energy on
1ubq because there are no Cys-Cys bridges) are validated.

Run with::

    PYTHONPATH=. /opt/homebrew/bin/python3.13 -m pytest \\
        tmol_jax/tests/test_gradcheck.py -v
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pytest

# Must be set before any jax.numpy import or jax trace. Upstream's
# torch-gradcheck runs at float64; we follow suit so eps=1e-4 central
# differences don't get swamped by float32 truncation.
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (deliberately after config.update)

from tmol_jax.score.score_types import ScoreType  # noqa: E402


# --------------------------------------------------------------------- #
# Test configuration                                                    #
# --------------------------------------------------------------------- #


#: Number of ``(pose, atom, axis)`` entries sampled per term.
N_SAMPLES = 20

#: Central-difference step. Small enough that the truncation error in
#: the second derivative term scales as ``O(eps^2)`` ~ 1e-8, big enough
#: that the float64 roundoff ``O(eps^-1 * eps_mach)`` ~ 1e-12 is below
#: our tolerance.
FD_EPS = 1e-4

#: Minimum magnitude of the analytical gradient entry for it to be
#: considered a "real" sample. Entries below this floor contribute no
#: signal to the test — they may be zeroed padding, atoms outside the
#: term's interaction set, or genuinely near-zero gradients that are
#: noise-dominated at ``eps=1e-4``.
GRAD_MAG_FLOOR = 0.5

#: Absolute / relative tolerances. Matches the upstream default shape
#: (``atol=1e-5, rtol=1e-3``) relaxed by ~100x because:
#:
#:   * Some terms (hbond, rama, dunbrack) go through a bicubic / polynomial
#:     spline whose third derivative is discontinuous at knot boundaries
#:     — central differences smear by ``O(eps * d3E/dx3)`` in those cells.
#:   * Cartesian terms (cart_torsions, cart_impropers) are periodic
#:     potentials that pass through kinks at ``+/- pi`` when the coord
#:     happens to sit near the branch cut.
#:
#: Anything the test catches above these is either a genuine bug or a
#: sample unlucky enough to land on a kink; in the latter case the
#: :func:`_mixed_tol_match` helper's ``rtol=1e-2`` still accepts entries
#: where ``abs(fd) >> atol``.
ATOL = 1e-3
RTOL = 1e-2

#: Deterministic RNG seed. The ``random.choice`` of sample indices
#: dominates which atoms we exercise, so pin this so failures are
#: reproducible across runs.
SEED = 0


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pose_stack_1ubq_local():
    """Load ubiquitin as a PoseStack, or skip the test module.

    Duplicated here (rather than imported from ``conftest.py``) because
    the conftest fixture is session-scoped and was created *before* we
    flipped ``jax_enable_x64`` on in this module — constructing it here
    ensures the coords live in the same 64-bit context we evaluate in.
    """
    pdb_path = os.path.join(os.path.dirname(__file__), "data", "1ubq.pdb")
    if not os.path.isfile(pdb_path):
        pytest.skip(f"1ubq.pdb not found at {pdb_path}")
    try:
        from tmol_jax.io import pose_stack_from_pdb
    except ImportError as exc:
        pytest.skip(f"pose_stack_from_pdb unavailable: {exc!r}")
    try:
        return pose_stack_from_pdb(pdb_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"pose_stack_from_pdb failed: {exc!r}")


@pytest.fixture(scope="module")
def coords64(pose_stack_1ubq_local):
    """Coordinates cast to float64.

    Float32 has ~7 decimal digits of mantissa; for a typical ``|E| ~ 100``
    energy the central-difference has ~1e-5 absolute noise floor from
    the subtraction. At ``eps=1e-4`` that shows up as ~1e-1 in the
    gradient estimate — well above ``atol=1e-3``. Float64 pushes the
    roundoff floor to ~1e-10, comfortably below tolerance.
    """
    return pose_stack_1ubq_local.coords.astype(jnp.float64)


# --------------------------------------------------------------------- #
# Score-function plumbing                                               #
# --------------------------------------------------------------------- #


def _build_single_term_sfxn(target_st: ScoreType):
    """Return a fresh score function with only ``target_st`` enabled at 1.0.

    Instantiating the beta2016 preset and then zeroing every weight
    except ``target_st`` is the simplest way to get "just this term"
    while still running through the normal term-registration path. The
    alternative — manually instantiating the term class — would
    duplicate the ``_TERM_FOR_TYPE`` dispatch logic from
    :mod:`tmol_jax.score.score_function` and diverge as terms are added.
    """
    from tmol_jax import beta2016_score_function

    sfxn = beta2016_score_function()
    # Zero every weight. The target term stays registered (we only
    # overwrite the weight), so its ``setup_for_pose_stack`` runs
    # normally on the next ``setup_for_pose_stack`` call.
    for st in list(ScoreType):
        if st is ScoreType.n_score_types:
            continue
        sfxn.set_weight(st, 0.0)
    sfxn.set_weight(target_st, 1.0)
    return sfxn


def _energy_fn(sfxn, pose_stack, state, target_st):
    """Produce ``coords -> scalar energy`` for the target score type.

    We go through :meth:`ScoreFunction.score_by_term` rather than the
    default ``__call__`` so we can assert that only the single enabled
    score type contributes — a sanity check that the caller really did
    zero out every other weight. If a sibling score type slips through
    we want to know, because the finite-difference would then mix
    channels and "pass" for the wrong reason.
    """

    def _e(coords):
        per_term = sfxn.score_by_term(
            pose_stack, state=state, coords_override=coords
        )
        # Exactly one active channel expected.
        if len(per_term) == 0:
            # All-zero term — the caller may choose to skip. We return
            # 0 so differentiating it gives the all-zero gradient array
            # of the right shape.
            return jnp.asarray(0.0, dtype=coords.dtype)
        assert set(per_term.keys()) == {target_st}, (
            f"Expected only {target_st!r}, got {list(per_term.keys())!r}"
        )
        return per_term[target_st].sum()

    return _e


def _pick_sample_indices(
    grad: jax.Array, n: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (flat, mags) of ``n`` sample indices into ``grad.ravel()``.

    First tries to pick ``n`` uniformly at random from entries with
    ``|grad| > GRAD_MAG_FLOOR``. Falls back to the top-n by magnitude
    when the above-threshold set is too small — this keeps the test
    informative for sparse terms like ``lk_bridge`` while preserving
    coverage for dense terms like ``fa_ljrep``.
    """
    flat = np.asarray(grad).ravel()
    mags = np.abs(flat)
    candidates = np.flatnonzero(mags > GRAD_MAG_FLOOR)
    if candidates.size >= n:
        picked = rng.choice(candidates, size=n, replace=False)
    elif candidates.size == 0:
        # Nothing above the floor -- fall back to the n largest, even
        # if their magnitudes are tiny. At that point the finite diff
        # will be dominated by noise; the caller should relax the
        # tolerance or skip.
        picked = np.argsort(-mags)[:n]
    else:
        # Take every above-floor entry, then top up with the largest
        # sub-floor entries. Using ``np.argsort(-mags)`` guarantees we
        # don't re-pick the same index twice.
        order = np.argsort(-mags)
        extra = [i for i in order if i not in set(candidates.tolist())]
        picked = np.concatenate(
            [candidates, np.asarray(extra[: n - candidates.size], dtype=np.int64)]
        )
    return picked, mags[picked]


def _mixed_tol_match(analytical: float, fd: float) -> bool:
    """``torch.autograd.gradcheck``-style mixed tolerance check.

    ``abs(a - fd) <= max(atol, rtol * abs(fd))``. Relative tolerance
    rescues large-magnitude entries where a uniform ``atol`` would be
    unreasonably strict; absolute tolerance rescues entries near zero
    where ``rtol * abs(fd)`` would shrink below noise.
    """
    return abs(analytical - fd) <= max(ATOL, RTOL * abs(fd))


def _central_fd_entry(
    energy_fn, coords: jax.Array, flat_idx: int, eps: float
) -> float:
    """Central finite difference at a single flat coordinate index.

    We perturb a single scalar of the coords tensor in-place (via
    ``.at[...].add``, which produces a new array under the functional
    JAX API) and evaluate the energy twice. Returning a plain Python
    float keeps the test-side bookkeeping out of the traced graph.

    The ``energy_fn`` supplied here is expected to be ``jax.jit``-wrapped
    by the caller: repeatedly eager-tracing a dense ``[1, N, N, 3]``
    pair kernel for every sampled index would otherwise dominate the
    wall-clock budget (~10x slowdown observed on the lk-ball terms).
    """
    # Unravel to a (pose, atom, axis) triple so the .at update targets
    # the correct scalar. ``coords.shape == (n_poses, max_atoms, 3)``.
    idx = np.unravel_index(flat_idx, coords.shape)
    c_plus = coords.at[idx].add(eps)
    c_minus = coords.at[idx].add(-eps)
    e_plus = float(energy_fn(c_plus))
    e_minus = float(energy_fn(c_minus))
    return (e_plus - e_minus) / (2.0 * eps)


# --------------------------------------------------------------------- #
# Test matrix                                                           #
# --------------------------------------------------------------------- #


# Score types weighted by the beta2016 preset. Order matches
# :class:`ScoreType`. We exclude:
#
#   * constraint: weight is zero on a fresh pose stack (no constraints),
#     so ``score_by_term`` returns an empty dict and the test has
#     nothing to differentiate.
#   * ref: per-residue constant -> zero gradient w.r.t. coords; there
#     are no above-threshold entries to pick.
_TESTED_SCORE_TYPES = [
    ScoreType.fa_ljatr,
    ScoreType.fa_ljrep,
    ScoreType.fa_lk,
    ScoreType.fa_elec,
    ScoreType.hbond,
    ScoreType.cart_lengths,
    ScoreType.cart_angles,
    ScoreType.cart_torsions,
    ScoreType.cart_impropers,
    ScoreType.cart_hxltorsions,
    ScoreType.disulfide,
    ScoreType.omega,
    ScoreType.rama,
    ScoreType.dunbrack_rot,
    ScoreType.dunbrack_rotdev,
    ScoreType.dunbrack_semirot,
    ScoreType.lk_ball_iso,
    ScoreType.lk_ball,
    ScoreType.lk_bridge,
    ScoreType.lk_bridge_uncpl,
]


@pytest.mark.parametrize(
    "score_type",
    _TESTED_SCORE_TYPES,
    ids=[st.name for st in _TESTED_SCORE_TYPES],
)
def test_gradcheck_per_term(
    score_type: ScoreType, pose_stack_1ubq_local, coords64
):
    """Per-term central-difference vs ``jax.grad`` on 1ubq.

    For each of the 20 non-trivial beta2016 terms, build an isolated
    score function, compute the analytical gradient at float64, and
    verify ``N_SAMPLES`` entries against a central finite difference.

    Terms that evaluate to an identically-zero energy on this input
    (e.g. ``disulfide`` on 1ubq because ubiquitin has no cystines) are
    skipped with a clear message — they provide no differentiation
    signal and a false-positive pass here would be misleading.
    """
    sfxn = _build_single_term_sfxn(score_type)
    state = sfxn.setup_for_pose_stack(pose_stack_1ubq_local)
    energy_fn = _energy_fn(sfxn, pose_stack_1ubq_local, state, score_type)
    # JIT both value and gradient. Without this, every FD step (2*N_SAMPLES
    # forward evals) retraces the term from scratch — dense pairwise terms
    # like fa_lk cost seconds per trace, so the whole test would take many
    # minutes. With jit, trace is amortised into a single compile, and
    # each subsequent eval is pure XLA.
    energy_jit = jax.jit(energy_fn)
    grad_jit = jax.jit(jax.grad(energy_fn))

    # Analytical gradient at the current coords. ``jax.grad`` against a
    # scalar-valued function is standard fare; we wrap in a lambda so
    # subsequent ``energy_fn`` calls during the FD loop reuse the same
    # closure (no retracing).
    e0 = float(energy_jit(coords64))
    g = grad_jit(coords64)

    # Guardrails: finite energy, finite gradient. If these fail the
    # term itself is broken (NaN leak) and finite-difference comparison
    # is meaningless.
    assert np.isfinite(e0), f"{score_type.name}: energy is not finite (got {e0})"
    assert bool(jnp.all(jnp.isfinite(g))), (
        f"{score_type.name}: gradient has non-finite entries "
        f"({int(jnp.isnan(g).sum())} NaN, {int(jnp.isinf(g).sum())} Inf)"
    )

    # Nothing to differentiate -- emit a clear skip so the test report
    # distinguishes "matched" from "vacuously passed".
    if float(jnp.max(jnp.abs(g))) == 0.0:
        pytest.skip(
            f"{score_type.name}: analytical gradient is identically zero "
            f"(energy = {e0:+.4f}); no FD signal to check."
        )

    # Pick N_SAMPLES entries weighted toward |g| > GRAD_MAG_FLOOR so we
    # don't waste samples on irrelevant atoms.
    rng = np.random.default_rng(SEED + int(score_type))
    flat_idxs, sample_mags = _pick_sample_indices(g, N_SAMPLES, rng)

    flat_grad = np.asarray(g).ravel()
    failures: list[str] = []
    for flat in flat_idxs:
        analytical = float(flat_grad[flat])
        # Use the jitted closure for FD evals so we reuse the single
        # compiled trace from the analytic gradient evaluation above.
        fd = _central_fd_entry(energy_jit, coords64, int(flat), FD_EPS)
        if not _mixed_tol_match(analytical, fd):
            # Include the unravelled index so failures are traceable to
            # specific (pose, atom, axis) entries — essential when the
            # failure is a sharp kink at one atom rather than a
            # systematic bug across the whole term.
            idx = np.unravel_index(int(flat), coords64.shape)
            failures.append(
                f"idx={idx} analytical={analytical:+.6f} "
                f"fd={fd:+.6f} |delta|={abs(analytical - fd):.3e}"
            )

    if failures:
        # Report all failures in one shot so a sweep across terms
        # produces a compact log rather than one error per sample.
        msg = (
            f"{score_type.name}: "
            f"{len(failures)}/{len(flat_idxs)} gradcheck samples failed "
            f"(atol={ATOL}, rtol={RTOL}, eps={FD_EPS}):\n  "
            + "\n  ".join(failures)
        )
        pytest.fail(msg)


# --------------------------------------------------------------------- #
# Combined (all-terms) gradient finiteness check                        #
# --------------------------------------------------------------------- #
#
# Per-term tests above isolate each score type, but the production use
# case for gradient-based protein design is *all-terms-together*
# ``jax.grad(sfxn)(coords)``. If any single term leaks a NaN on its
# masked branch, the sum's gradient still becomes NaN everywhere (NaN
# + anything = NaN in IEEE) and the minimizer silently derails.
#
# This test exercises the full beta2016 preset — every term the
# production score function touches — at float64 and asserts the
# gradient is finite everywhere.


def test_gradcheck_combined_beta2016(pose_stack_1ubq_local, coords64):
    """Full beta2016 score function: ``jax.grad`` must be all-finite.

    Verifies the production-mode contract: composing every scored
    channel in the beta2016 preset and calling ``jax.grad(sfxn)(coords)``
    yields a gradient with no NaN / Inf entries. This is the guarantee
    the gradient-based protein-design pipeline depends on — if any
    single term leaks a NaN on a masked branch, the composed sum's
    gradient is NaN everywhere and the minimizer derails silently.

    We also sanity-check that at least some entries have finite,
    non-zero gradients (i.e. the term set isn't accidentally all-zero).
    """
    from tmol_jax import beta2016_score_function

    sfxn = beta2016_score_function()
    # Activate every weightable score type the preset omits at the
    # default weights. ``set_weight`` is idempotent and lazily
    # registers any term that wasn't already part of the preset.
    for st in ScoreType:
        if st is ScoreType.n_score_types:
            continue
        try:
            # Don't overwrite non-zero preset weights. If the term is
            # already weighted, leave it alone; only give every other
            # term a weight of 1.0 so we exercise the full ScoreType
            # surface in one pass.
            if sfxn.get_weight(st) == 0.0:
                sfxn.set_weight(st, 1.0)
        except KeyError:
            # Some ScoreType entries have no registered term (e.g.
            # score types added to the enum but not yet ported). Skip
            # them rather than failing the combined test.
            continue

    state = sfxn.setup_for_pose_stack(pose_stack_1ubq_local)

    def total_energy(coords):
        return sfxn(
            pose_stack_1ubq_local, state=state, coords_override=coords
        ).sum()

    grad_fn = jax.jit(jax.grad(total_energy))
    g = grad_fn(coords64)

    n_nan = int(jnp.isnan(g).sum())
    n_inf = int(jnp.isinf(g).sum())
    assert n_nan == 0 and n_inf == 0, (
        f"combined beta2016 gradient has non-finite entries "
        f"(NaN={n_nan}, Inf={n_inf}); a term is leaking through a "
        f"masked branch. Suspect candidates: any term with "
        f"``jnp.sqrt`` / ``jnp.arccos`` / ``normalize`` on a padded "
        f"atom pair."
    )

    # Sanity: there should be at least some non-trivially-sized
    # gradient entry. Otherwise the above "all finite" is vacuous (a
    # zero gradient array trivially has no NaNs).
    max_mag = float(jnp.max(jnp.abs(g)))
    assert max_mag > 1e-6, (
        f"combined beta2016 gradient is effectively zero "
        f"(max |grad| = {max_mag:.3e}); the energy probably collapsed "
        f"to a constant, which would make the no-NaN assertion vacuous."
    )
