"""Shared math primitives for jtmol scoring.

Pure JAX ports of the C++ utilities in ``tmol/score/common``.
Derivatives are obtained via ``jax.grad`` — only value variants are
exposed here.
"""

from jtmol.score.common.count_pair import (
    MAX_COUNTPAIR_SEPARATION,
    connectivity_weight,
    pair_bonded_path_length_direct,
    pairwise_bonded_path_length,
    pose_atom_block_index,
    pose_atom_real_mask,
    scatter_per_atom,
)
from jtmol.score.common.cubic_hermite_polynomial import (
    interpolate,
    interpolate_t,
    interpolate_to_zero,
    interpolate_to_zero_t,
)
from jtmol.score.common.geom import (
    cos_interior_angle,
    dihedral_angle,
    distance,
    normalize,
    pt_interior_angle,
)
from jtmol.score.common.neighbor_list import (
    build_cell_list,
    gather_neighbor_values,
    gather_pair_matrix,
    pairwise_sum_with_nbrs,
)
from jtmol.score.common.polynomial import bound_poly, poly_v

__all__ = [
    "MAX_COUNTPAIR_SEPARATION",
    "bound_poly",
    "build_cell_list",
    "connectivity_weight",
    "cos_interior_angle",
    "dihedral_angle",
    "distance",
    "gather_neighbor_values",
    "gather_pair_matrix",
    "interpolate",
    "interpolate_t",
    "interpolate_to_zero",
    "interpolate_to_zero_t",
    "normalize",
    "pair_bonded_path_length_direct",
    "pairwise_bonded_path_length",
    "pairwise_sum_with_nbrs",
    "pose_atom_block_index",
    "pose_atom_real_mask",
    "scatter_per_atom",
    "poly_v",
    "pt_interior_angle",
]
