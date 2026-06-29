"""tmol_jax — pure-JAX port of tmol (Rosetta beta_nov2016_cart).

This top-level module re-exports the small, stable surface that mirrors the
upstream `tmol.__init__`. See the README for the overall map.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# ``name -> (submodule, attr)``. Resolved on demand so one broken submodule
# does not break ``import tmol_jax``.
_EXPORTS: dict[str, tuple[str, str]] = {
    "pose_stack_from_pdb": ("tmol_jax.io.pdb_parsing", "pose_stack_from_pdb"),
    "beta2016_score_function": ("tmol_jax.score", "beta2016_score_function"),
    "ScoreFunction": ("tmol_jax.score.score_function", "ScoreFunction"),
    "ScoreType": ("tmol_jax.score.score_types", "ScoreType"),
    "PoseStack": ("tmol_jax.pose.pose_stack", "PoseStack"),
    "PackedBlockTypes": ("tmol_jax.pose.packed_block_types", "PackedBlockTypes"),
    "ParameterDatabase": ("tmol_jax.database", "ParameterDatabase"),
    "run_min": ("tmol_jax.optimization.minimizers", "run_min"),
    "run_cart_min": ("tmol_jax.optimization.minimizers", "run_cart_min"),
    "run_kin_min": ("tmol_jax.optimization.minimizers", "run_kin_min"),
    "write_pose_stack_pdb": ("tmol_jax.io.write_pose_stack_pdb", "write_pose_stack_pdb"),
    "fast_relax": ("tmol_jax.relax.fast_relax", "fast_relax"),
    "FoldForest": ("tmol_jax.kinematics.fold_forest", "FoldForest"),
    "EdgeType": ("tmol_jax.kinematics.fold_forest", "EdgeType"),
    "MoveMap": ("tmol_jax.kinematics.move_map", "MoveMap"),
    "CartesianMoveMap": ("tmol_jax.kinematics.move_map", "CartesianMoveMap"),
}


def __getattr__(name: str):
    try:
        modpath, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'tmol_jax' has no attribute {name!r}") from None
    return getattr(importlib.import_module(modpath), attr)


__all__ = list(_EXPORTS)
