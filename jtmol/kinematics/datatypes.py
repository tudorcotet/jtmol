"""Kinematic forest datatypes ported to JAX.

This module is a JAX-native port of ``tmol.kinematics.datatypes``. All
tensor-backed containers are exposed as ``pytree_dataclass`` instances so
they flow cleanly through ``jit``, ``vmap``, ``grad``, and ``lax.scan``.
The PyTorch/C++ references in the upstream file are gone; array data is
stored as ``jax.Array`` (typically ``int32`` indices or ``float64`` DOF
values) with ``numpy`` used only for authoring default values.
"""

from __future__ import annotations

import enum

import attrs
import jax.numpy as jnp
import numpy as np

from jtmol.types import Array, pytree_dataclass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(enum.IntEnum):
    """KinForest node types."""

    root = 0
    jump = 1
    bond = 2


class BondDOFTypes(enum.IntEnum):
    """Indices of bond dof types within KinDOF.raw."""

    phi_p = 0
    theta = 1
    d = 2
    phi_c = 3


class JumpDOFTypes(enum.IntEnum):
    """Indices of jump dof types within KinDOF.raw."""

    RBx = 0
    RBy = 1
    RBz = 2
    RBdel_alpha = 3
    RBdel_beta = 4
    RBdel_gamma = 5
    RBalpha = 6
    RBbeta = 7
    RBgamma = 8


n_movable_bond_dof_types = 4
n_movable_jump_dof_types = 6


# ---------------------------------------------------------------------------
# KinForest
# ---------------------------------------------------------------------------


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class KinForest:
    """A collection of atom-level kinematic trees, each of which can be processed
    in parallel.

    A kinematic description of a collection of atom locations, each atom location
    corresponding to a node within a tree. The root of each tree in this forest
    is built from a jump from the global reference frame at the origin. (The
    global reference frame will later be treated as a node in the forest,
    effectively linking all the trees, but this is a minor technical detail; best
    to think of this as several independent trees than a single tree). Every
    other node corresponds to a derived orientation, with an atomic coordinate at
    the center of the frame.

    Each node in the tree is connected by one of two "node types":

    1) Jump nodes, representing an arbitrary rigid body transform between two
    reference frames via six degrees of freedom, 3 translational and
    3 rotational.

    2) Bond nodes, representing the relationships between two atom reference
    frames via three bond degrees of freedom: the translation from the parent
    to the child along the bond axis (bond length, d), the rotation from the
    grand-parent-to-parent bond axis to the bond axis (an improper bond
    angle, theta), and the rotation about the grand-parent-to-parent bond axis
    (bond torsion, phi). Bond nodes include an additional, redundant,
    degree of freedom representing concerted rotation of all downstream atoms
    about the parent-to-self bond. These DOFs are used to represent
    the torsions that alter the location of several children. For example,
    chi1 is represented as the 4th DOF for the CB atom of LEU. A rotation
    about the CA-->CB bond axis will spin CG, HB1 and HB2. In this scheme,
    the phi DOF would be 0 for CG, 120 for HB1 and 240 for HB2. This differs
    from the Rosetta3 implementation of downstream-dihedral propagation
    where Chi1 would live as the phi DOF of CG, and CG's rotation would
    carry forward to HB1 and HB2 (requiring that CG be the first child of
    CB).

    The atoms in the `KinForest` have their own order that is distinct
    from the ordering in the target (e.g. a PoseStack) where there might
    be gaps between sets of atoms (e.g. because each Pose in the stack
    has a different number of atoms, so a contiguous block of atom indices
    from 0-100 might have a gap before the next contiguous block begins
    at 150). When working with a `KinForest`, remembering what order
    and array's indices is in (the kin-forest order (KFO) or the target
    order (TO)) and what a value/index read out of an array represents (is
    the index an index in KFO or TO?) is *very* challenging. The documentation
    for these arrays includes whether the arrays are indexed in KFO or TO
    and whether the values they hold are KFO or TO indices.

    The `KinForest` data structure itself is frozen and can not be modified post
    construction. `KinForests` are not typically built directly by users,
    but rather are constructed as part of building the `KinModuleData` class
    by `construct_kin_module_data` that lives in scan_ordering.py.

    Indices::
        id       = the TO index in KFO; i.e. kin_forest_order_2_target_order
        doftype  = NodeType for each node (root, jump, bond), indexed in KFO
        parent   = KFO index of the parent, in KFO
        frame_x  = KFO index of self, in KFO
        frame_y  = KFO index of parent, in KFO
        frame_z  = KFO index of grandparent, in KFO
    """

    id: Array  # int32[N]
    doftype: Array  # int32[N]
    parent: Array  # int32[N]
    frame_x: Array  # int32[N]
    frame_y: Array  # int32[N]
    frame_z: Array  # int32[N]

    @classmethod
    def node(
        cls,
        id: int,
        doftype: int,
        parent: int,
        frame_x: int,
        frame_y: int,
        frame_z: int,
    ) -> "KinForest":
        """Construct a single-node KinForest from scalar values."""
        mk = lambda v: jnp.asarray(np.asarray([int(v)], dtype=np.int32), dtype=jnp.int32)
        return cls(
            id=mk(id),
            doftype=mk(int(doftype)),
            parent=mk(parent),
            frame_x=mk(frame_x),
            frame_y=mk(frame_y),
            frame_z=mk(frame_z),
        )

    @classmethod
    def root_node(cls) -> "KinForest":
        """The global/root kinematic node at KinForest[0]."""
        return cls.node(
            id=-1,
            doftype=int(NodeType.root),
            parent=0,
            frame_x=0,
            frame_y=0,
            frame_z=0,
        )


# ---------------------------------------------------------------------------
# Scan data / module data
# ---------------------------------------------------------------------------


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class KinForestScanData:
    """Segmented-scan description for one direction of KinForest traversal."""

    nodes: Array  # int32[N]
    scans: Array  # int32[N]
    gens: Array  # int32[M]


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class KinematicModuleData:
    """Container bundling a KinForest with the auxiliary tensors needed to run
    forward / backward kinematics over a pose-stack-shaped fold forest."""

    forest: KinForest
    scan_data_fw: KinForestScanData
    scan_data_bw: KinForestScanData

    # Extra tensors describing the fold forest in its entirety.
    block_in_and_first_out: Array  # int32[n_poses, n_blocks]
    keep_atom_fixed: Array  # bool[n_poses, n_atoms]
    pose_stack_atom_for_jump: Array  # int32[n_poses, max_n_jumps, 2]
    pose_stack_atom_for_root_jump: Array  # int32[n_poses, 2]


# ---------------------------------------------------------------------------
# DOF containers
# ---------------------------------------------------------------------------


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class BondDOF:
    """A bond-DOF view of a KinDOF.raw slice."""

    raw: Array  # float64[..., 4]

    @property
    def phi_p(self):
        return self.raw[..., BondDOFTypes.phi_p]

    @property
    def theta(self):
        return self.raw[..., BondDOFTypes.theta]

    @property
    def d(self):
        return self.raw[..., BondDOFTypes.d]

    @property
    def phi_c(self):
        return self.raw[..., BondDOFTypes.phi_c]


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class JumpDOF:
    """A jump-DOF view of a KinDOF.raw slice."""

    raw: Array  # float64[..., 9]

    @property
    def RBx(self):
        return self.raw[..., JumpDOFTypes.RBx]

    @property
    def RBy(self):
        return self.raw[..., JumpDOFTypes.RBy]

    @property
    def RBz(self):
        return self.raw[..., JumpDOFTypes.RBz]

    @property
    def RBdel_alpha(self):
        return self.raw[..., JumpDOFTypes.RBdel_alpha]

    @property
    def RBdel_beta(self):
        return self.raw[..., JumpDOFTypes.RBdel_beta]

    @property
    def RBdel_gamma(self):
        return self.raw[..., JumpDOFTypes.RBdel_gamma]

    @property
    def RBalpha(self):
        return self.raw[..., JumpDOFTypes.RBalpha]

    @property
    def RBbeta(self):
        return self.raw[..., JumpDOFTypes.RBbeta]

    @property
    def RBgamma(self):
        return self.raw[..., JumpDOFTypes.RBgamma]


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class KinDOF:
    """Internal coordinate data.

    The KinDOF data structure holds two logical views: the "raw" view, a
    sparsely populated [N, 9] tensor of DOF values, and a set of named
    property accessors providing access to specific entries within this
    array. This is logically equivalent to a C union datatype: the
    interpretation of an entry in the DOF buffer depends on the type of
    the corresponding KinForest entry.
    """

    raw: Array  # float64[N, 9]

    @property
    def bond(self) -> BondDOF:
        return BondDOF(raw=self.raw[..., :4])

    @property
    def jump(self) -> JumpDOF:
        return JumpDOF(raw=self.raw[..., :9])

    def clone(self) -> "KinDOF":
        """Return a shallow copy. JAX arrays are immutable, so this is a no-op
        wrapper that preserves the PyTorch-era API."""
        return KinDOF(raw=self.raw)


__all__ = [
    "NodeType",
    "BondDOFTypes",
    "JumpDOFTypes",
    "n_movable_bond_dof_types",
    "n_movable_jump_dof_types",
    "KinForest",
    "KinForestScanData",
    "KinematicModuleData",
    "BondDOF",
    "JumpDOF",
    "KinDOF",
]
