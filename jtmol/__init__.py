"""jtmol — pure-JAX port of tmol (Rosetta beta_nov2016_cart).

This top-level module re-exports the small, stable surface that mirrors the
upstream `tmol.__init__`. See the README for the overall map.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# ``name -> (submodule, attr)``. Resolved on demand so one broken submodule
# does not break ``import jtmol``.
_EXPORTS: dict[str, tuple[str, str]] = {
    "pose_stack_from_pdb": ("jtmol.io.pdb_parsing", "pose_stack_from_pdb"),
    "beta2016_score_function": ("jtmol.score", "beta2016_score_function"),
    "beta_jan25_score_function": ("jtmol.score", "beta_jan25_score_function"),
    "ScoreFunction": ("jtmol.score.score_function", "ScoreFunction"),
    "ScoreType": ("jtmol.score.score_types", "ScoreType"),
    "PoseStack": ("jtmol.pose.pose_stack", "PoseStack"),
    "PackedBlockTypes": ("jtmol.pose.packed_block_types", "PackedBlockTypes"),
    "ParameterDatabase": ("jtmol.database", "ParameterDatabase"),
    "run_min": ("jtmol.optimization.minimizers", "run_min"),
    "run_cart_min": ("jtmol.optimization.minimizers", "run_cart_min"),
    "run_kin_min": ("jtmol.optimization.minimizers", "run_kin_min"),
    "write_pose_stack_pdb": ("jtmol.io.write_pose_stack_pdb", "write_pose_stack_pdb"),
    "fast_relax": ("jtmol.relax.fast_relax", "fast_relax"),
    "FoldForest": ("jtmol.kinematics.fold_forest", "FoldForest"),
    "EdgeType": ("jtmol.kinematics.fold_forest", "EdgeType"),
    "MoveMap": ("jtmol.kinematics.move_map", "MoveMap"),
    "CartesianMoveMap": ("jtmol.kinematics.move_map", "CartesianMoveMap"),
}


def __getattr__(name: str):
    try:
        modpath, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'jtmol' has no attribute {name!r}") from None
    return getattr(importlib.import_module(modpath), attr)


__all__ = list(_EXPORTS)
