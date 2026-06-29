"""tmol_jax — pure-JAX port of tmol (Rosetta beta_nov2016_cart).

This top-level module re-exports the small, stable surface that mirrors the
upstream `tmol.__init__`. See the README for the overall map.
"""

from __future__ import annotations

__version__ = "0.1.0"


def _lazy(name: str):
    """Resolve a top-level re-export lazily so a single broken submodule
    doesn't prevent `import tmol_jax` from succeeding."""

    def _get():
        if name == "pose_stack_from_pdb":
            from tmol_jax.io.pdb_parsing import pose_stack_from_pdb

            return pose_stack_from_pdb
        if name == "beta2016_score_function":
            from tmol_jax.score import beta2016_score_function

            return beta2016_score_function
        if name == "ScoreFunction":
            from tmol_jax.score.score_function import ScoreFunction

            return ScoreFunction
        if name == "ScoreType":
            from tmol_jax.score.score_types import ScoreType

            return ScoreType
        if name == "PoseStack":
            from tmol_jax.pose.pose_stack import PoseStack

            return PoseStack
        if name == "PackedBlockTypes":
            from tmol_jax.pose.packed_block_types import PackedBlockTypes

            return PackedBlockTypes
        if name == "ParameterDatabase":
            from tmol_jax.database import ParameterDatabase

            return ParameterDatabase
        if name == "run_min":
            from tmol_jax.optimization.minimizers import run_min

            return run_min
        if name == "run_cart_min":
            from tmol_jax.optimization.minimizers import run_cart_min

            return run_cart_min
        if name == "run_kin_min":
            from tmol_jax.optimization.minimizers import run_kin_min

            return run_kin_min
        if name == "write_pose_stack_pdb":
            from tmol_jax.io.write_pose_stack_pdb import write_pose_stack_pdb

            return write_pose_stack_pdb
        if name == "fast_relax":
            from tmol_jax.relax.fast_relax import fast_relax

            return fast_relax
        if name == "FoldForest":
            from tmol_jax.kinematics.fold_forest import FoldForest

            return FoldForest
        if name == "EdgeType":
            from tmol_jax.kinematics.fold_forest import EdgeType

            return EdgeType
        if name == "MoveMap":
            from tmol_jax.kinematics.move_map import MoveMap

            return MoveMap
        if name == "CartesianMoveMap":
            from tmol_jax.kinematics.move_map import CartesianMoveMap

            return CartesianMoveMap
        raise AttributeError(name)

    return _get


_EXPORTS = [
    "pose_stack_from_pdb",
    "beta2016_score_function",
    "ScoreFunction",
    "ScoreType",
    "PoseStack",
    "PackedBlockTypes",
    "ParameterDatabase",
    "run_min",
    "run_cart_min",
    "run_kin_min",
    "write_pose_stack_pdb",
    "fast_relax",
    "FoldForest",
    "EdgeType",
    "MoveMap",
    "CartesianMoveMap",
]


def __getattr__(name):
    if name in _EXPORTS:
        return _lazy(name)()
    raise AttributeError(f"module 'tmol_jax' has no attribute {name!r}")


__all__ = _EXPORTS
