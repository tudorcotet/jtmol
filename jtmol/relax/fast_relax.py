"""FastRelax protocol — pure-Python JAX port of ``tmol.relax.fast_relax``.

FastRelax is a fixed-point "repack + ramped minimization" protocol used
across Rosetta/tmol to drive a pose into a nearby local minimum of the
energy surface. Each cycle repacks the rotamers once and then runs a
short sequence of minimizations at increasing ``fa_ljrep`` (Lennard-Jones
repulsion) weights. The low starting weight lets atoms pass through each
other just enough for sidechains to settle; ramping back up to the full
weight at the end restores the physical repulsion.

This port deliberately stays simple and eager:

* No best-pose tracking / accept-to-best step. Upstream keeps the
  lowest-energy PoseStack across repeats; in JAX we just return the
  final pose because (a) the common case is a single repeat in a
  differentiable pipeline, and (b) ``attr.evolve``-style mutation of a
  pytree is more awkward in JAX. If you need best-pose selection, wrap
  ``fast_relax`` in a loop and pick the argmin yourself.
* No kinematic minimization by default. We call
  :func:`jtmol.optimization.minimizers.run_min`, which under the
  simplified JAX optimizer layer is Cartesian LBFGS. The full kinematic
  (torsion-space) path is still available by passing ``min_fn`` manually.
* Ramp is a flat list of ``fa_ljrep`` fractions rather than the richer
  ``{"fa_rep_pack_frac", "fa_rep_min_frac", "cst_frac"}`` dicts used
  upstream. The four-fraction default
  ``[0.02, 0.25, 0.55, 1.0]`` matches the "classic" FastRelax ramp
  that most tmol callers expect.

The protocol:

    if constrain_relax_to_start:
        add_mainchain_coordinate_constraints(sigma=bb_cst_sigma)

    for cycle in range(n_cycles):
        (optional) pack_rotamers(pose, sfxn)
        for frac in ramp_schedule:
            sfxn.set_weight(fa_ljrep, frac * start_weight)
            pose = run_min(pose, sfxn)
        sfxn.set_weight(fa_ljrep, start_weight)   # restore

    return pose

Public entry points:

* :func:`fast_relax` — full protocol (pack + ramp minimize per cycle).
* :func:`fast_relax_no_pack` — skips the packer, just runs ramped
  minimization. Handy for tests and for pipelines that have their own
  sequence-design step.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional, Sequence

from jtmol.pose.pose_stack import PoseStack
from jtmol.score.score_function import ScoreFunction
from jtmol.score.score_types import ScoreType


# ---------------------------------------------------------------------- #
# Default ramp schedule.                                                 #
#                                                                        #
# These are the *fractions* that multiply the starting ``fa_ljrep``       #
# weight (typically 0.55 in beta_nov2016_cart). The four-point ramp      #
# matches the canonical FastRelax defaults Rosetta ships with; it's      #
# shallower at the start than MonomerRelax2019 (which begins at 0.040)   #
# but the final minimization still ends at full weight so the output     #
# pose is scored correctly.                                              #
# ---------------------------------------------------------------------- #
DEFAULT_RAMP_SCHEDULE: tuple[float, ...] = (0.02, 0.25, 0.55, 1.0)


# ---------------------------------------------------------------------- #
# Type aliases for clarity.                                              #
# ---------------------------------------------------------------------- #

# A minimization callable. Signature chosen to be compatible with both
# :func:`jtmol.optimization.minimizers.run_min` (Cartesian, the
# default) and :func:`run_kin_min` (kinematic). Extra kwargs are
# ignored by Cartesian minimizers.
MinFn = Callable[[PoseStack, ScoreFunction], PoseStack]


# Default min / pack hooks are resolved lazily inside the entry points so
# the sibling packages (``optimization.minimizers``, ``pack.packer``) are
# only imported when actually needed.


def _maybe_add_mainchain_constraints(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    constrain_relax_to_start: bool,
    bb_cst_sigma: float,
) -> PoseStack:
    """If enabled, attach harmonic coord constraints to every mainchain atom.

    We also make sure the constraint channel has a non-zero weight on
    the score function so the term actually contributes to the energy.
    If the caller already set a constraint weight, we leave it alone.
    """
    if not constrain_relax_to_start:
        return pose_stack

    from jtmol.score.constraint.utility import (
        create_mainchain_coordinate_constraints,
    )

    pose_stack = create_mainchain_coordinate_constraints(
        pose_stack, sigma=bb_cst_sigma
    )
    # Surface the constraint channel if the user hasn't already. The
    # upstream convention is "constraint weight starts at 1.0 when
    # constraints are applied"; we mirror that.
    if sfxn.get_weight(ScoreType.constraint) == 0.0:
        sfxn.set_weight(ScoreType.constraint, 1.0)
    return pose_stack


# ---------------------------------------------------------------------- #
# Internal helpers.                                                      #
# ---------------------------------------------------------------------- #


def _run_ramp(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ramp_schedule: Sequence[float],
    fa_ljrep_start: float,
    min_fn: MinFn,
    verbose: bool,
) -> PoseStack:
    """Run one ramp (minimize at each scaled ``fa_ljrep`` weight).

    The starting weight is restored at the end regardless of how the
    loop exits, so the caller's score function is left untouched.
    """
    try:
        for step_idx, frac in enumerate(ramp_schedule):
            scaled = float(frac) * fa_ljrep_start
            sfxn.set_weight(ScoreType.fa_ljrep, scaled)
            if verbose:
                print(
                    f"    ramp {step_idx + 1}/{len(ramp_schedule)}:"
                    f" fa_ljrep = {scaled:.4f} ({frac:.3f} x {fa_ljrep_start:.4f})"
                )
            t0 = time.perf_counter()
            pose_stack = min_fn(pose_stack, sfxn)
            if verbose:
                print(f"      minimize: {time.perf_counter() - t0:.2f}s")
    finally:
        sfxn.set_weight(ScoreType.fa_ljrep, fa_ljrep_start)
    return pose_stack


def _validate_schedule(schedule: Sequence[float]) -> tuple[float, ...]:
    """Coerce ``schedule`` to a tuple of floats and warn on odd shapes."""
    if len(schedule) == 0:
        raise ValueError("ramp_schedule must contain at least one entry.")
    out = tuple(float(x) for x in schedule)
    if abs(out[-1] - 1.0) > 1e-6:
        # The last step should normally be at the full weight so the
        # returned pose is scored against the real energy surface.
        warnings.warn(
            f"Final ramp step is {out[-1]:.3f} (expected 1.0); the"
            " returned pose will be at a scaled fa_ljrep weight.",
            stacklevel=3,
        )
    return out


# ---------------------------------------------------------------------- #
# Public entry points.                                                   #
# ---------------------------------------------------------------------- #


def fast_relax(
    pose_stack: PoseStack,
    score_function: ScoreFunction,
    n_cycles: int = 5,
    *,
    constrain_relax_to_start: bool = True,
    bb_cst_sigma: float = 0.5,
    ramp_schedule: Sequence[float] = DEFAULT_RAMP_SCHEDULE,
    pack_fn: Optional[Callable[[PoseStack, ScoreFunction], PoseStack]] = None,
    min_fn: Optional[MinFn] = None,
    verbose: bool = False,
) -> PoseStack:
    """Run the FastRelax protocol on ``pose_stack``.

    Parameters
    ----------
    pose_stack
        Input pose(s) to relax. The stack axis is preserved; all poses
        are relaxed in lockstep (shared JIT trace).
    score_function
        The :class:`~jtmol.score.ScoreFunction` used for both packing
        and minimization. ``fa_ljrep`` is temporarily rescaled during
        the ramp and restored after every cycle. If
        ``constrain_relax_to_start`` is True and the ``constraint``
        weight is zero, we set it to 1.0 so the added mainchain
        constraints actually contribute to the gradient.
    n_cycles
        Number of outer cycles (pack + ramped minimize). Default 5,
        matching upstream.
    constrain_relax_to_start
        If True, attach harmonic coordinate constraints to every
        mainchain atom (CA/N/C/O for canonical amino acids) keyed on
        the starting coordinates. This keeps the backbone near its
        input geometry during relaxation — useful for refining
        a predicted structure without drifting.
    bb_cst_sigma
        Standard deviation (Å) of the backbone coordinate constraints.
        Smaller = stiffer. Default 0.5 Å, matching upstream's
        ``create_mainchain_coordinate_constraints``.
    ramp_schedule
        Fractions of ``fa_ljrep_start`` to use at each minimization
        step within a cycle. Default is :data:`DEFAULT_RAMP_SCHEDULE`
        (``[0.02, 0.25, 0.55, 1.0]``). The last entry should normally
        be 1.0 so the returned pose is scored at the full weight.
    pack_fn
        Optional override for the rotamer packer. Must have signature
        ``(pose_stack, sfxn) -> PoseStack``. Default uses
        :func:`jtmol.pack.packer.pack_rotamers`.
    min_fn
        Optional override for the per-step minimizer. Must have
        signature ``(pose_stack, sfxn) -> PoseStack``. Default uses
        :func:`jtmol.optimization.minimizers.run_min` (Cartesian
        LBFGS). Pass a lambda that calls ``run_kin_min`` for
        internal-coordinate minimization.
    verbose
        Print per-cycle / per-ramp-step timing.

    Returns
    -------
    PoseStack
        The relaxed pose stack after ``n_cycles`` of pack + ramp
        minimize. ``fa_ljrep`` on ``score_function`` is restored to
        its starting value on exit (including via exception).

    Notes
    -----
    * The starting ``fa_ljrep`` weight is captured once from the input
      score function. If it is zero, we raise — running FastRelax with
      no repulsive LJ term would be meaningless.
    * This is a pure-Python loop over JAX calls; the inner work is
      jit-compilable through the minimizer and packer. We do not
      jit the outer loop itself because ``sfxn.set_weight`` mutates
      Python state and re-triggers tracing.
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    ramp = _validate_schedule(ramp_schedule)

    fa_ljrep_start = float(score_function.get_weight(ScoreType.fa_ljrep))
    if fa_ljrep_start == 0.0:
        raise ValueError(
            "score_function has fa_ljrep weight = 0; FastRelax needs a"
            " non-zero starting fa_ljrep to ramp. Set it via"
            " sfxn.set_weight(ScoreType.fa_ljrep, ...)."
        )

    if pack_fn is None:
        from jtmol.pack.packer import pack_rotamers as pack_fn
    if min_fn is None:
        from jtmol.optimization.minimizers import run_min as min_fn

    pose_stack = _maybe_add_mainchain_constraints(
        pose_stack, score_function, constrain_relax_to_start, bb_cst_sigma
    )

    if verbose:
        print(
            f"fast_relax: {n_cycles} cycles, ramp={list(ramp)},"
            f" fa_ljrep_start={fa_ljrep_start:.4f},"
            f" constrain_bb={constrain_relax_to_start} (sigma={bb_cst_sigma})"
        )

    try:
        for cycle_idx in range(n_cycles):
            if verbose:
                print(f"cycle {cycle_idx + 1}/{n_cycles}")

            # Repack rotamers at the starting weight. We reset fa_ljrep
            # just in case a previous iteration bailed mid-ramp.
            score_function.set_weight(ScoreType.fa_ljrep, fa_ljrep_start)
            t_pack = time.perf_counter()
            pose_stack = pack_fn(pose_stack, score_function)
            if verbose:
                print(f"  pack: {time.perf_counter() - t_pack:.2f}s")

            # Ramped minimization at increasing fa_ljrep weights.
            pose_stack = _run_ramp(
                pose_stack,
                score_function,
                ramp,
                fa_ljrep_start,
                min_fn,
                verbose,
            )
    finally:
        # Always restore fa_ljrep, even on exceptions.
        score_function.set_weight(ScoreType.fa_ljrep, fa_ljrep_start)

    return pose_stack


def fast_relax_no_pack(
    pose_stack: PoseStack,
    score_function: ScoreFunction,
    n_cycles: int = 5,
    *,
    constrain_relax_to_start: bool = True,
    bb_cst_sigma: float = 0.5,
    ramp_schedule: Sequence[float] = DEFAULT_RAMP_SCHEDULE,
    min_fn: Optional[MinFn] = None,
    verbose: bool = False,
) -> PoseStack:
    """FastRelax without rotamer packing — ramped minimization only.

    Identical to :func:`fast_relax` but skips the packer. Useful when:

    * The sidechains are already placed by an upstream step (e.g. an
      ML-predicted pose, or a dedicated packing pass) and you only
      want to minimize.
    * The rotamer packer is not available (e.g. while porting tests
      before the pack layer lands).
    * You want a fully differentiable relax — the packer is a
      discrete choice and not trivially differentiable, but the
      minimization path is.

    The same ``n_cycles`` outer loop is preserved so repeated ramps
    can still be done (each ramp still progresses the pose through
    the schedule), but no rotamer swaps happen in between.

    See :func:`fast_relax` for a description of the parameters.
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    ramp = _validate_schedule(ramp_schedule)

    fa_ljrep_start = float(score_function.get_weight(ScoreType.fa_ljrep))
    if fa_ljrep_start == 0.0:
        raise ValueError(
            "score_function has fa_ljrep weight = 0; FastRelax needs a"
            " non-zero starting fa_ljrep to ramp."
        )

    if min_fn is None:
        from jtmol.optimization.minimizers import run_min as min_fn

    pose_stack = _maybe_add_mainchain_constraints(
        pose_stack, score_function, constrain_relax_to_start, bb_cst_sigma
    )

    if verbose:
        print(
            f"fast_relax_no_pack: {n_cycles} cycles, ramp={list(ramp)},"
            f" fa_ljrep_start={fa_ljrep_start:.4f},"
            f" constrain_bb={constrain_relax_to_start} (sigma={bb_cst_sigma})"
        )

    try:
        for cycle_idx in range(n_cycles):
            if verbose:
                print(f"cycle {cycle_idx + 1}/{n_cycles}")
            pose_stack = _run_ramp(
                pose_stack,
                score_function,
                ramp,
                fa_ljrep_start,
                min_fn,
                verbose,
            )
    finally:
        score_function.set_weight(ScoreType.fa_ljrep, fa_ljrep_start)

    return pose_stack


__all__ = [
    "DEFAULT_RAMP_SCHEDULE",
    "fast_relax",
    "fast_relax_no_pack",
]
