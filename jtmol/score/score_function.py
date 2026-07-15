"""Score function: sum of weighted :class:`EnergyTerm` s.

Upstream reference: :mod:`tmol.score.score_function`. The torch version
maintains an ordered list of nn.Module terms, a weight tensor, and three
"dirty" flags for the sorted term caches. The JAX port is deliberately
simpler:

* Weights are a plain ``dict[ScoreType, float]``. They are Python floats
  (static under ``jit``), so changing a weight does not invalidate the
  traced energy graph — only changing which term is registered does.
* Terms are a ``list[EnergyTerm]`` in insertion order. We don't maintain
  separate one-body / two-body / multi-body buckets because the JAX
  design collapses all body counts into a single ``__call__(pose_stack,
  term_state, coords)`` signature.
* Registration is lazy: the first time :meth:`set_weight` mentions a
  score type that no currently-registered term provides, we look up the
  import path in :data:`_TERM_FOR_TYPE`, import the module, and
  instantiate the term.

The score function produces three levels of output:

    * :meth:`setup_for_pose_stack`  — precompute per-pose state.
    * ``__call__``                  — scalar-per-pose weighted total.
    * :meth:`score_by_term`         — dict of weighted per-score-type
                                      energies (useful for debugging and
                                      for the "score by term" REST API).

All three accept an optional ``coords_override`` that is threaded into
every term so ``jax.grad(sfxn)(coords)`` works out of the box.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any, Callable, Optional

import attrs
import jax
import jax.numpy as jnp

from jtmol.score.energy_term import EnergyTerm
from jtmol.score.score_types import ScoreType


# ---------------------------------------------------------------------- #
# Term registry.                                                         #
#                                                                        #
# Maps each ScoreType to the import path of the EnergyTerm class that   #
# produces it, using the convention "module.path:ClassName".            #
#                                                                        #
# A single EnergyTerm may appear under multiple score-type keys (for     #
# example, LJLKEnergyTerm covers fa_ljatr, fa_ljrep, and fa_lk). The     #
# ScoreFunction uses this table to lazily import and instantiate terms   #
# when a weight is first set for a score type they cover.                #
# ---------------------------------------------------------------------- #

_TERM_FOR_TYPE: dict[ScoreType, str] = {
    # LJ / LK (jtmol.score.ljlk.ljlk_energy_term:LJLKEnergyTerm)
    ScoreType.fa_ljatr: "jtmol.score.ljlk.ljlk_energy_term:LJLKEnergyTerm",
    ScoreType.fa_ljrep: "jtmol.score.ljlk.ljlk_energy_term:LJLKEnergyTerm",
    ScoreType.fa_lk: "jtmol.score.ljlk.ljlk_energy_term:LJLKEnergyTerm",
    # Electrostatics
    ScoreType.fa_elec: "jtmol.score.elec.elec_energy_term:ElecEnergyTerm",
    # Hydrogen bonds
    ScoreType.hbond: "jtmol.score.hbond.hbond_energy_term:HBondEnergyTerm",
    # Cartesian bonded (5 sub-terms share a single CartBondedEnergyTerm)
    ScoreType.cart_lengths: "jtmol.score.cartbonded.cartbonded_energy_term:CartBondedEnergyTerm",
    ScoreType.cart_angles: "jtmol.score.cartbonded.cartbonded_energy_term:CartBondedEnergyTerm",
    ScoreType.cart_torsions: "jtmol.score.cartbonded.cartbonded_energy_term:CartBondedEnergyTerm",
    ScoreType.cart_impropers: "jtmol.score.cartbonded.cartbonded_energy_term:CartBondedEnergyTerm",
    ScoreType.cart_hxltorsions: "jtmol.score.cartbonded.cartbonded_energy_term:CartBondedEnergyTerm",
    # Constraints
    ScoreType.constraint: "jtmol.score.constraint.constraint_energy_term:ConstraintEnergyTerm",
    # Disulfide
    ScoreType.disulfide: "jtmol.score.disulfide.disulfide_energy_term:DisulfideEnergyTerm",
    # Backbone torsions (rama + omega share one term)
    ScoreType.omega: "jtmol.score.backbone_torsion.backbone_torsion_energy_term:BackboneTorsionEnergyTerm",
    ScoreType.rama: "jtmol.score.backbone_torsion.backbone_torsion_energy_term:BackboneTorsionEnergyTerm",
    # Dunbrack (3 sub-terms)
    ScoreType.dunbrack_rot: "jtmol.score.dunbrack.dunbrack_energy_term:DunbrackEnergyTerm",
    ScoreType.dunbrack_rotdev: "jtmol.score.dunbrack.dunbrack_energy_term:DunbrackEnergyTerm",
    ScoreType.dunbrack_semirot: "jtmol.score.dunbrack.dunbrack_energy_term:DunbrackEnergyTerm",
    # LK ball (4 sub-terms)
    ScoreType.lk_ball_iso: "jtmol.score.lk_ball.lk_ball_energy_term:LKBallEnergyTerm",
    ScoreType.lk_ball: "jtmol.score.lk_ball.lk_ball_energy_term:LKBallEnergyTerm",
    ScoreType.lk_bridge: "jtmol.score.lk_ball.lk_ball_energy_term:LKBallEnergyTerm",
    ScoreType.lk_bridge_uncpl: "jtmol.score.lk_ball.lk_ball_energy_term:LKBallEnergyTerm",
    # Reference energies
    ScoreType.ref: "jtmol.score.ref.ref_energy_term:RefEnergyTerm",
    # Generic bonded torsions + harmonic bond-stretch / bond-angle
    ScoreType.gen_torsions: "jtmol.score.genbonded.genbonded_energy_term:GenBondedEnergyTerm",
    ScoreType.gen_bond_length: "jtmol.score.genbonded.genbonded_energy_term:GenBondedEnergyTerm",
    ScoreType.gen_bond_angle: "jtmol.score.genbonded.genbonded_energy_term:GenBondedEnergyTerm",
}


def _resolve(path: str):
    """Turn a ``"pkg.mod:Name"`` string into the referenced object."""
    module_path, _, attr = path.partition(":")
    if not attr:
        raise ValueError(
            f"Invalid term import path {path!r}: expected 'module:ClassName'."
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# ---------------------------------------------------------------------- #
# Per-pose term state bag.                                               #
# ---------------------------------------------------------------------- #


@attrs.define(slots=False)
class ScoreFunctionState:
    """Bag of per-term, per-pose-stack precomputed state.

    Produced by :meth:`ScoreFunction.setup_for_pose_stack` and threaded
    through subsequent score calls. We register it as a JAX pytree so
    it can flow through ``jit``, ``vmap``, and ``scan`` transparently;
    ``term_states`` is a list of pytrees (one per registered term, in
    the same order as :attr:`ScoreFunction._terms`).
    """

    term_states: list[Any]


def _sfs_flatten(s: ScoreFunctionState):
    return (s.term_states,), None


def _sfs_unflatten(aux, children):
    (term_states,) = children
    return ScoreFunctionState(term_states=list(term_states))


jax.tree_util.register_pytree_node(ScoreFunctionState, _sfs_flatten, _sfs_unflatten)


# ---------------------------------------------------------------------- #
# Score function.                                                        #
# ---------------------------------------------------------------------- #


class ScoreFunction:
    """Weighted sum of :class:`EnergyTerm`s.

    Build one by creating an empty score function against a
    :class:`~jtmol.database.ParameterDatabase` and then calling
    :meth:`set_weight` for each desired channel, or use the
    :func:`~jtmol.score.beta2016_score_function` preset for the
    full beta_nov2016_cart weights.

    The score function is immutable with respect to ``jit``: the set of
    registered terms and the weights are captured into the traced graph
    when you first call it. If you later want to change weights or add a
    term, re-run ``jax.jit`` (or call ``sfxn`` fresh in eager mode).

    Parameters
    ----------
    param_db
        Parameter database shared between all energy terms.
    """

    def __init__(self, param_db: Any):
        self._param_db = param_db
        self._weights: dict[ScoreType, float] = {}
        self._terms: list[EnergyTerm] = []
        # st -> index into self._terms, for fast "is this covered?" checks
        # and for locating the owning term when a weight is zeroed.
        self._term_for_st: dict[ScoreType, int] = {}
        # Caches of pre-jitted closures, keyed by ``id(pose_stack)``. Each
        # entry holds up to three variants (``value``, ``grad``,
        # ``value_and_grad``) so repeated calls with the same pose stack
        # don't recompile. See :meth:`jitted` / :meth:`jitted_grad` /
        # :meth:`jitted_value_and_grad`.
        self._jit_cache: dict[int, dict[str, Callable]] = {}

    # ------------------------------------------------------------------ #
    # Weight management.                                                 #
    # ------------------------------------------------------------------ #

    def set_weight(self, st: ScoreType, weight: float) -> None:
        """Set the weight for score type ``st``.

        If no currently registered term produces ``st``, look the term
        up in :data:`_TERM_FOR_TYPE`, import it, and add it. Setting a
        weight to ``0.0`` does *not* remove the term — the term stays
        registered so its sibling score types continue to work.
        """
        if not isinstance(st, ScoreType):
            raise TypeError(f"Expected ScoreType, got {type(st).__name__}")
        if st not in self._term_for_st:
            self._retrieve_term_for_score_type(st)
        self._weights[st] = float(weight)

    def get_weight(self, st: ScoreType) -> float:
        """Return the current weight for ``st`` (``0.0`` if unset)."""
        return float(self._weights.get(st, 0.0))

    def _retrieve_term_for_score_type(self, st: ScoreType) -> None:
        """Lazily import and register the term that produces ``st``."""
        if st not in _TERM_FOR_TYPE:
            raise KeyError(
                f"No term registered in jtmol.score.score_function for {st!r}. "
                f"Update _TERM_FOR_TYPE if you added a new EnergyTerm."
            )
        term_cls = _resolve(_TERM_FOR_TYPE[st])
        term = term_cls(self._param_db)
        # All terms must expose ``score_types_provided`` per the EnergyTerm
        # base class contract. If a new term doesn't, fix the term — don't
        # widen the dispatcher.
        provided = list(term.score_types_provided())
        assert st in provided, (
            f"Term {term_cls.__name__} does not provide {st!r}; "
            f"it provides {list(provided)}."
        )
        term_index = len(self._terms)
        self._terms.append(term)
        for tst in provided:
            # Register sibling score types too so future set_weight calls on
            # them don't spuriously re-import and re-instantiate the term.
            if tst not in self._term_for_st:
                self._term_for_st[tst] = term_index

    # ------------------------------------------------------------------ #
    # Introspection.                                                     #
    # ------------------------------------------------------------------ #

    @property
    def param_db(self) -> Any:
        return self._param_db

    @property
    def terms(self) -> list[EnergyTerm]:
        """Read-only view of the registered terms, in insertion order."""
        return list(self._terms)

    @property
    def weights(self) -> dict[ScoreType, float]:
        """Shallow copy of the weights dict."""
        return dict(self._weights)

    def active_score_types(self) -> list[ScoreType]:
        """Score types with a non-zero weight, in ScoreType enum order."""
        return [
            st
            for st in ScoreType
            if st is not ScoreType.n_score_types and self._weights.get(st, 0.0) != 0.0
        ]

    # ------------------------------------------------------------------ #
    # Setup and evaluation.                                              #
    # ------------------------------------------------------------------ #

    def setup_for_pose_stack(self, pose_stack: Any) -> ScoreFunctionState:
        """Run each term's :meth:`EnergyTerm.setup_for_pose_stack`.

        Returns a :class:`ScoreFunctionState` that must be threaded
        into later calls of :meth:`__call__` / :meth:`score_by_term`.
        Pass ``state=None`` to have the caller auto-run setup each call
        (fine in eager mode, wasteful under jit).
        """
        self._reject_topology_only_covalent_state(pose_stack)
        pbt = getattr(pose_stack, "packed_block_types", None)
        term_states = [t.setup_for_pose_stack(pose_stack, pbt) for t in self._terms]
        return ScoreFunctionState(term_states=term_states)

    @staticmethod
    def _reject_topology_only_covalent_state(pose_stack: Any) -> None:
        pbt = getattr(pose_stack, "packed_block_types", None)
        topology_only_reason = getattr(
            pbt, "covalent_topology_only_reason", None
        )
        if topology_only_reason:
            raise ValueError(
                "cannot score a topology-only covalent state: "
                f"{topology_only_reason}. Supply a backend that consumes "
                "state-specific bond orders, bonded parameters, protonation, "
                "and partial charges; do not interpret this graph as a TI "
                "free energy."
            )

    def __call__(
        self,
        pose_stack: Any,
        state: Optional[ScoreFunctionState] = None,
        coords_override: Optional[jax.Array] = None,
        check_finite: bool = False,
    ) -> jax.Array:
        """Return the total weighted energy as ``float32[n_poses]``.

        If ``state`` is ``None`` we auto-run :meth:`setup_for_pose_stack`.
        This is convenient for one-shot scoring, but prefer the
        explicit-state form inside a jit to avoid redoing the setup on
        every trace.

        ``check_finite`` (eager-only debug guard, default off so the hot path
        pays nothing): if set, RAISE when the total is non-finite. A NaN in the
        input coords or a broken/unbuilt atom poisons a whole term's score and
        gradient to NaN; this fails loudly at the source rather than letting a
        NaN propagate silently downstream. It deliberately does NOT mask the NaN
        to 0 -- a silent floor would hide the broken pose and let a minimizer
        wander off it. Do not pass this under ``jit`` (the boolean check syncs).
        """
        per_term = self.score_by_term(
            pose_stack, state=state, coords_override=coords_override
        )
        if not per_term:
            # No active terms — return zero-per-pose.
            coords = (
                coords_override
                if coords_override is not None
                else pose_stack.coords
            )
            n_poses = coords.shape[0]
            return jnp.zeros((n_poses,), dtype=jnp.float32)

        total = None
        for value in per_term.values():
            total = value if total is None else total + value
        if check_finite and not bool(jnp.all(jnp.isfinite(total))):
            raise ValueError(
                "non-finite score (NaN/Inf): a NaN/Inf coord or a broken/unbuilt "
                "atom is poisoning a term. Fix the input structure -- do not mask "
                "the NaN (that hides the broken pose). check_finite is eager-only."
            )
        return total

    def score_by_term(
        self,
        pose_stack: Any,
        state: Optional[ScoreFunctionState] = None,
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        """Return ``{ScoreType: weighted_energy[n_poses]}``.

        Only score types with a non-zero weight are included. Zero-weighted
        score types covered by a registered term are still evaluated by
        that term internally — they just aren't returned here, because
        multiplying by zero would waste memory without affecting the sum.
        """
        self._reject_topology_only_covalent_state(pose_stack)
        if state is None:
            state = self.setup_for_pose_stack(pose_stack)

        result: dict[ScoreType, jax.Array] = {}
        for term, term_state in zip(self._terms, state.term_states):
            unweighted = term(
                pose_stack, term_state, coords_override=coords_override
            )
            for st, energy in unweighted.items():
                w = self._weights.get(st, 0.0)
                if w == 0.0:
                    # Skip zero-weighted channels. They remain unreported so
                    # downstream code that iterates over score_by_term sees
                    # only the active terms. The total energy is unchanged.
                    continue
                result[st] = w * energy
        return result

    # ------------------------------------------------------------------ #
    # Pre-jitted convenience wrappers.                                   #
    #                                                                    #
    # For repeated calls with the same pose-stack shape (L-BFGS inner    #
    # loop, Monte Carlo sampling, ML training) callers want a            #
    # pre-compiled ``coords -> energy`` (or gradient) callable so every  #
    # iteration amortises over a single trace. These helpers use         #
    # ``jax.jit`` + ``functools.partial`` to close over the pose stack   #
    # and term state as pytrees, then cache the result on ``self`` keyed #
    # by ``id(pose_stack)`` so calling ``jitted`` twice with the same    #
    # pose stack is free.                                                #
    # ------------------------------------------------------------------ #

    def _jit_cache_entry(self, pose_stack: Any) -> dict[str, Callable]:
        """Return (or create) the per-pose-stack cache bucket."""
        key = id(pose_stack)
        entry = self._jit_cache.get(key)
        if entry is None:
            entry = {}
            self._jit_cache[key] = entry
        return entry

    def jitted(
        self,
        pose_stack: Any,
        state: Optional[ScoreFunctionState] = None,
    ) -> Callable[[jax.Array], jax.Array]:
        """Return a ``jax.jit``-compiled ``coords -> energy[n_poses]``.

        The returned callable takes a coords array with the same shape as
        ``pose_stack.coords`` and returns per-pose weighted energies.
        ``pose_stack`` and the precomputed ``state`` are captured into
        the traced graph, so the first call compiles and subsequent calls
        reuse the cached trace.

        If ``state`` is ``None``, we run :meth:`setup_for_pose_stack`
        once up-front so the setup never enters the traced graph.

        Results are cached on ``self`` keyed by ``id(pose_stack)``:
        calling ``jitted`` again with the same pose stack returns the
        already-compiled closure without retracing.
        """
        entry = self._jit_cache_entry(pose_stack)
        cached = entry.get("value")
        if cached is not None:
            return cached

        if state is None:
            state = self.setup_for_pose_stack(pose_stack)

        def _score(ps: Any, st: ScoreFunctionState, coords: jax.Array) -> jax.Array:
            return self(ps, state=st, coords_override=coords)

        closure = jax.jit(functools.partial(_score, pose_stack, state))
        entry["value"] = closure
        return closure

    def jitted_grad(
        self,
        pose_stack: Any,
        state: Optional[ScoreFunctionState] = None,
    ) -> Callable[[jax.Array], jax.Array]:
        """Return a ``jax.jit``-compiled gradient ``coords -> d(sum_E)/d(coords)``.

        The gradient has the same shape as ``pose_stack.coords`` (i.e.
        ``[n_poses, max_n_atoms_per_pose, 3]``). We differentiate the
        *sum* of per-pose energies — for single-pose stacks this is the
        usual scalar gradient; for multi-pose stacks each pose
        contributes additively (which is what every minimizer, sampler,
        and ML loss in upstream actually wants).
        """
        entry = self._jit_cache_entry(pose_stack)
        cached = entry.get("grad")
        if cached is not None:
            return cached

        if state is None:
            state = self.setup_for_pose_stack(pose_stack)

        def _sum_energy(
            ps: Any, st: ScoreFunctionState, coords: jax.Array
        ) -> jax.Array:
            return jnp.sum(self(ps, state=st, coords_override=coords))

        grad_fn = jax.grad(_sum_energy, argnums=2)
        closure = jax.jit(functools.partial(grad_fn, pose_stack, state))
        entry["grad"] = closure
        return closure

    def jitted_value_and_grad(
        self,
        pose_stack: Any,
        state: Optional[ScoreFunctionState] = None,
    ) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
        """Return a ``jax.jit``-compiled fused ``(energy, gradient)``.

        Returns a closure ``vg(coords) -> (energy[n_poses], grad_coords)``
        where ``grad_coords`` has the same shape as ``coords``. Backed by
        ``jax.value_and_grad`` on the summed energy, so it's the right
        shape for L-BFGS / Adam / any gradient-based driver. The *value*
        is returned per-pose (not summed) because downstream code often
        wants both total and per-pose energies without a second trace.
        """
        entry = self._jit_cache_entry(pose_stack)
        cached = entry.get("value_and_grad")
        if cached is not None:
            return cached

        if state is None:
            state = self.setup_for_pose_stack(pose_stack)

        def _energy_and_sum(
            ps: Any, st: ScoreFunctionState, coords: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            per_pose = self(ps, state=st, coords_override=coords)
            return jnp.sum(per_pose), per_pose

        # ``has_aux=True`` lets us propagate per-pose energies alongside
        # the scalar-for-grad so the gradient is taken against the sum
        # while the caller still gets the full ``[n_poses]`` vector.
        vg = jax.value_and_grad(_energy_and_sum, argnums=2, has_aux=True)

        def _apply(
            ps: Any, st: ScoreFunctionState, coords: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            (_, per_pose), grad = vg(ps, st, coords)
            return per_pose, grad

        closure = jax.jit(functools.partial(_apply, pose_stack, state))
        entry["value_and_grad"] = closure
        return closure


__all__ = ["ScoreFunction", "ScoreFunctionState"]
