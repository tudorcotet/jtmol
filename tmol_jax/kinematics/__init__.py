"""JAX-native port of :mod:`tmol.kinematics`.

Re-exports the core public API — datatypes, fold forest, forward /
inverse kinematics, and move maps — so downstream modules can use
``from tmol_jax.kinematics import KinForest, ...`` without reaching
into submodules.
"""

from .datatypes import (
    KinForest,
    KinDOF,
    BondDOF,
    JumpDOF,
    KinematicModuleData,
    NodeType,
)
from .fold_forest import FoldForest, EdgeType
from .operations import forward_kin, inverse_kin, inverseKin
from .move_map import MoveMap, CartesianMoveMap
from .scan_ordering import construct_kin_module_data


__all__ = [
    "KinForest",
    "KinDOF",
    "BondDOF",
    "JumpDOF",
    "KinematicModuleData",
    "NodeType",
    "FoldForest",
    "EdgeType",
    "forward_kin",
    "inverse_kin",
    "inverseKin",
    "MoveMap",
    "CartesianMoveMap",
    "construct_kin_module_data",
]
