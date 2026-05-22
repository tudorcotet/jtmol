"""Abstract base class for a single energy term.

Upstream reference: :mod:`tmol.score.energy_term`. That torch version is
built around the torch nn.Module machinery — ``setup_block_type``,
``setup_packed_block_types``, ``setup_poses``, and per-call rendering
into a ``torch.nn.Module`` that the minimizer can autograd through.

In JAX we collapse all of that into two calls:

* :meth:`setup_for_pose_stack` runs *once* per :class:`PoseStack` and
  produces a pytree ("term state") of precomputed, conformation-independent
  arrays. These are the equivalent of upstream's block-type / packed
  block-type / pose-stack annotations, merged into a single object so the
  caller never has to worry about the three-stage setup sequence.

* ``__call__`` is the hot path. It runs inside ``jit``/``grad`` and must
  return a dict ``{ScoreType: jax.Array[n_poses]}`` of *unweighted*
  per-score-type energies. The :class:`ScoreFunction` applies the weights
  and sums across terms.

The ``coords_override`` argument is key for autograd: the optimizer wants
``grad`` with respect to coordinates, but ``PoseStack`` pytrees flatten
into many leaves. Rather than forcing every minimizer to reconstruct a
pose stack around a candidate coords array, every term accepts an
optional override that is substituted for ``pose_stack.coords`` inside
the energy evaluation.

Subclasses should override:

    * ``score_types_provided``        — required, static list
    * ``setup_for_pose_stack``        — optional, defaults to empty dict
    * ``__call__``                    — required; pure function of
                                        (pose_stack, term_state, coords).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import jax

from tmol_jax.score.score_types import ScoreType


class EnergyTerm(ABC):
    """Base class for one energy term (LJLK, Hbond, Rama, Ref, ...).

    Parameters
    ----------
    param_db
        A :class:`tmol_jax.database.ParameterDatabase`. Stored by
        reference on ``self.param_db``; subclasses use it in
        :meth:`setup_for_pose_stack` to look up tabulated parameters
        (LJ well depth, Dunbrack rotamer grids, cart bond spring
        constants, etc.).

    Notes
    -----
    The ``param_db`` must stay alive for as long as the term is used.
    Subclasses should treat it as read-only.
    """

    def __init__(self, param_db: Any):
        self.param_db = param_db

    # ------------------------------------------------------------------ #
    # Metadata.                                                          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def score_types_provided(self) -> list[ScoreType]:
        """Return the list of score types this term can compute.

        The order here is merely documentary — :meth:`__call__` returns
        a dict keyed by :class:`ScoreType`, so the order in which a term
        reports its channels has no effect on the weighted total.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Setup.                                                             #
    # ------------------------------------------------------------------ #

    def setup_for_pose_stack(self, pose_stack: Any, pbt: Any = None) -> Any:
        """Precompute conformation-independent arrays for a pose stack.

        This is the JAX analogue of upstream tmol's three-stage block-type
        / packed-block-type / pose-stack setup. The returned value must be
        a pytree (dict, attrs ``pytree_dataclass``, tuple of jax.Arrays,
        ...) so it survives ``jit`` / ``vmap`` / ``pytree.map``.

        Parameters
        ----------
        pose_stack
            The :class:`PoseStack` we will eventually score.
        pbt
            Optional :class:`PackedBlockTypes` override. Defaults to
            ``pose_stack.packed_block_types`` when ``None``. This is
            useful in tests and in the factor-graph path where the
            palette is shared across many pose stacks.

        Returns
        -------
        pytree
            Any JAX pytree. The default implementation returns an empty
            dict — terms that depend only on :class:`PoseStack` arrays
            already exposed via standard fields (e.g. :class:`RefEnergyTerm`,
            which only needs ``block_type_ind``) can leave this alone.
        """
        del pose_stack, pbt
        return {}

    # ------------------------------------------------------------------ #
    # Evaluation (hot path, runs inside jit).                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def __call__(
        self,
        pose_stack: Any,
        term_state: Any,
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        """Compute unweighted per-score-type energies.

        Parameters
        ----------
        pose_stack
            The pose stack whose conformation we are scoring. In the
            ``coords_override`` branch below, the rest of the pose stack
            (block types, offsets, connections, ...) is still read from
            here; only the coordinate array is substituted.
        term_state
            Whatever :meth:`setup_for_pose_stack` returned for this
            pose stack. The term may close over it or branch on static
            shapes inside it, but must not mutate it.
        coords_override
            Optional ``float32[n_poses, max_n_atoms, 3]`` array that
            replaces ``pose_stack.coords`` for this evaluation. Used by
            the minimizer so that ``grad(fn)(coords)`` can trace cleanly
            without reconstructing a :class:`PoseStack` pytree.

        Returns
        -------
        dict[ScoreType, jax.Array]
            Mapping from each score type in
            :meth:`score_types_provided` to a ``float32`` array of shape
            ``[n_poses]``. The :class:`ScoreFunction` multiplies each
            entry by its weight and sums across terms to produce the
            total.
        """
        raise NotImplementedError


__all__ = ["EnergyTerm"]
