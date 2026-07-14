"""Pure-JAX forward and inverse kinematics for KinForest trees.

This module is a JAX port of the C++ / CUDA routines in
``tmol/kinematics/compiled/common.hh`` (the mathematical ground truth) and
the Python wrappers in ``tmol/kinematics/operations.py``. The math is
preserved verbatim; only the execution model changes:

* All transforms are built as plain ``jnp`` 4x4 matrices.
* ``forward_kin`` composes transforms down the tree *generation by
  generation* using ``jax.lax.scan``: a pre-computed BFS-depth
  partitioning groups atoms so all atoms sharing a generation have
  parents in an earlier generation, and therefore can be refolded in
  parallel via ``vmap`` once the previous generation's HTs are in hand.
  When no pre-computed scan data is available, the implementation falls
  back to a sequential ``jax.lax.fori_loop`` — still correct because
  ``parent[i] < i`` for every non-root ``i``, but with no per-generation
  parallelism.
* ``inverse_kin`` is fully vectorized with ``vmap`` since every atom's
  frame and local HT can be computed independently once coordinates are
  available.

**On derivatives.** The upstream C++ implements explicit ``bondDerivatives`` /
``jumpDerivatives`` routines using the Abe-Go trick so that a hand-written
backward pass can avoid re-materializing rotation Jacobians. In JAX we get
that same speedup automatically through ``jax.grad`` / ``jax.vjp``: the
forward HT composition is differentiable end-to-end, so no manual adjoint
code is needed. We intentionally do **not** port ``bondDerivatives`` /
``jumpDerivatives`` — they would be redundant and would require us to run
``jax.lax.scan`` backwards with a bespoke segmented scan. ``jax.grad`` does
the right thing over ``forward_kin`` out of the box.

Conventions (matching the upstream C++ exactly):

* Row-vector homogeneous transforms. A ``4x4`` HT stores its three basis
  vectors in rows 0-2 (columns 0-2) and its origin in row 3 (columns 0-2).
  The last column is always ``[0, 0, 0, 1]^T``. A point ``p`` transforms as
  ``p' = [p, 1] @ HT``; composition of transforms is left-to-right, matching
  the ``HTs[i] = HTs[i] * HTs[parent]`` line in the C++ forward pass.
* The translation of node ``i`` is therefore ``HTs[i, 3, :3]``.
* DOF indexing mirrors ``common.hh``:
  - ``BondDOF``: ``[phi_p, theta, d, phi_c]`` at raw indices ``0..3``.
  - ``JumpDOF``: ``[RBx, RBy, RBz, RBdel_alpha, RBdel_beta, RBdel_gamma,
    RBalpha, RBbeta, RBgamma]`` at raw indices ``0..8``.
* ``NodeType`` values: ``ROOT=0``, ``JUMP=1``, ``BOND=2``.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jtmol.kinematics.datatypes import KinDOF, KinForest, NodeType

# ---------------------------------------------------------------------------
# NodeType integer codes — materialized as Python ints so they can serve as
# ``lax.switch`` branch indices inside a traced function.
# ---------------------------------------------------------------------------

_ROOT = int(NodeType.root)
_JUMP = int(NodeType.jump)
_BOND = int(NodeType.bond)

# Numerical tolerance for singularity detection. Matches the 1e-6 threshold
# used in common.hh's invBondTransform / invJumpTransform.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Local-frame DOF -> 4x4 homogeneous transforms
# ---------------------------------------------------------------------------


def bondTransform(dof4: jnp.ndarray) -> jnp.ndarray:
    """Convert a 4-vector of bond DOFs to a 4x4 homogeneous transform.

    Mirrors ``common::bondTransform`` exactly. The local frame composition is
    ``rot(phi_p, x) @ rot(theta, z) @ trans(d, x) @ rot(phi_c, x)``.

    Parameters
    ----------
    dof4 : array [4]
        ``[phi_p, theta, d, phi_c]``.

    Returns
    -------
    HT : array [4, 4] (float64)
    """
    dof4 = jnp.asarray(dof4, dtype=jnp.float64)
    phi_p, theta, d, phi_c = dof4[0], dof4[1], dof4[2], dof4[3]
    cpp = jnp.cos(phi_p)
    spp = jnp.sin(phi_p)
    cpc = jnp.cos(phi_c)
    spc = jnp.sin(phi_c)
    cth = jnp.cos(theta)
    sth = jnp.sin(theta)

    # HT(row, col) assignments copied verbatim from common.hh lines 175-190.
    HT = jnp.array(
        [
            [cth,             cpp * sth,                    spp * sth,                   0.0],
            [-cpc * sth,      cpc * cpp * cth - spc * spp,  cpp * spc + cpc * cth * spp, 0.0],
            [spc * sth,       -cpp * cth * spc - cpc * spp, cpc * cpp - cth * spc * spp, 0.0],
            [d * cth,         d * cpp * sth,                d * spp * sth,               1.0],
        ],
        dtype=jnp.float64,
    )
    return HT


def _euler_zyx_delta(si, sj, sk, ci, cj, ck) -> jnp.ndarray:
    """Build the rotation block used by jumpTransform (both HTdelta and
    HTglobal share the same 3x3 structure)."""
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    # Rows 0..2, columns 0..2 of the 4x4 transform, filled per common.hh.
    return jnp.array(
        [
            [cj * ck,        cj * sk,       -sj,    0.0],
            [sj * sc - cs,   sj * ss + cc,  cj * si, 0.0],
            [sj * cc + ss,   sj * cs - sc,  cj * ci, 0.0],
        ],
        dtype=jnp.float64,
    )


def jumpTransform(dof9: jnp.ndarray) -> jnp.ndarray:
    """Convert a 9-vector of jump DOFs to a 4x4 homogeneous transform.

    Mirrors ``common::jumpTransform``. Composition is::

        HT = HTglobal(RBalpha, RBbeta, RBgamma) @ HTdelta(RBdel_*, translation)

    where both rotation blocks are ZYX Euler angles.

    Parameters
    ----------
    dof9 : array [9]
        ``[RBx, RBy, RBz, RBdel_alpha, RBdel_beta, RBdel_gamma,
        RBalpha, RBbeta, RBgamma]``.

    Returns
    -------
    HT : array [4, 4] (float64)
    """
    dof9 = jnp.asarray(dof9, dtype=jnp.float64)
    RBx, RBy, RBz = dof9[0], dof9[1], dof9[2]
    rda, rdb, rdg = dof9[3], dof9[4], dof9[5]
    ra, rb, rg = dof9[6], dof9[7], dof9[8]

    # HTdelta: the small-rotation correction + translation, bottom row is (RBx, RBy, RBz, 1).
    delta_rot = _euler_zyx_delta(
        jnp.sin(rda), jnp.sin(rdb), jnp.sin(rdg),
        jnp.cos(rda), jnp.cos(rdb), jnp.cos(rdg),
    )
    HTdelta = jnp.concatenate(
        [delta_rot, jnp.array([[RBx, RBy, RBz, 1.0]], dtype=jnp.float64)],
        axis=0,
    )

    # HTglobal: the "user" rotation, no translation.
    global_rot = _euler_zyx_delta(
        jnp.sin(ra), jnp.sin(rb), jnp.sin(rg),
        jnp.cos(ra), jnp.cos(rb), jnp.cos(rg),
    )
    HTglobal = jnp.concatenate(
        [global_rot, jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float64)],
        axis=0,
    )

    # Row-vector convention: order matches ``HTglobal * HTdelta`` in common.hh.
    return HTglobal @ HTdelta


def _identity_ht() -> jnp.ndarray:
    return jnp.eye(4, dtype=jnp.float64)


def _zero_bond_dof_to_ht(_dof: jnp.ndarray) -> jnp.ndarray:
    """Root placeholder: always identity, regardless of DOF contents."""
    return _identity_ht()


def _local_transform(dof9: jnp.ndarray, doftype: jnp.ndarray) -> jnp.ndarray:
    """Dispatch a 9-vector of DOFs to the correct local-frame transform based
    on ``doftype``. ``lax.switch`` keeps this jit-compatible without Python
    branches."""
    return jax.lax.switch(
        doftype,
        (
            _zero_bond_dof_to_ht,           # ROOT  -> identity
            jumpTransform,                  # JUMP  -> full 9-DOF transform
            lambda d: bondTransform(d[:4]), # BOND  -> only the first 4 DOFs matter
        ),
        dof9,
    )


# ---------------------------------------------------------------------------
# Inverse transforms: HT -> DOFs
# ---------------------------------------------------------------------------


def invBondTransform(HT: jnp.ndarray) -> jnp.ndarray:
    """Recover ``[phi_p, theta, d, phi_c]`` from a local bond HT.

    Mirrors ``common::invBondTransform``. Handles the ``theta == 0``
    singularity by assigning all x-axis rotation to ``phi_c`` (matching the
    C++ convention).

    Parameters
    ----------
    HT : array [4, 4]
        Local homogeneous transform (i.e. ``HT[i] @ ht_inv(HT[parent])`` for
        a non-root node).

    Returns
    -------
    dof4 : array [4]
        ``[phi_p, theta, d, phi_c]``.
    """
    HT = jnp.asarray(HT, dtype=jnp.float64)

    # Non-singular (theta != 0) branch:
    phip_reg = jnp.arctan2(HT[0, 2], HT[0, 1])
    phic_reg = jnp.arctan2(-HT[2, 0], -HT[1, 0])
    theta_reg = jnp.arctan2(jnp.sqrt(HT[1, 0] ** 2 + HT[2, 0] ** 2), HT[0, 0])

    # Singular (theta ≈ 0) branch: phi_p := 0, theta := 0, phi_c := atan2(M(1,2), M(1,1)).
    phip_sing = jnp.asarray(0.0, dtype=jnp.float64)
    theta_sing = jnp.asarray(0.0, dtype=jnp.float64)
    phic_sing = jnp.arctan2(HT[1, 2], HT[1, 1])

    is_singular = jnp.abs(HT[0, 0] - 1.0) < _EPS
    phi_p = jnp.where(is_singular, phip_sing, phip_reg)
    theta = jnp.where(is_singular, theta_sing, theta_reg)
    phi_c = jnp.where(is_singular, phic_sing, phic_reg)

    # d = norm of the bottom-left 1x3 (origin of the child frame in parent coords).
    d = jnp.linalg.norm(HT[3, :3])

    return jnp.stack([phi_p, theta, d, phi_c]).astype(jnp.float64)


def invJumpTransform(HT: jnp.ndarray) -> jnp.ndarray:
    """Recover the 9-vector of jump DOFs from a local jump HT.

    Mirrors ``common::invJumpTransform``. The RBdel components are always
    returned as zero (the redundancy between RB and RBdel is resolved by
    putting all the rotation into RBalpha/beta/gamma). Handles the
    ``cos(beta) ≈ 0`` gimbal-lock singularity by putting all rotation into
    alpha.

    Parameters
    ----------
    HT : array [4, 4]

    Returns
    -------
    dof9 : array [9]
    """
    HT = jnp.asarray(HT, dtype=jnp.float64)

    RBx, RBy, RBz = HT[3, 0], HT[3, 1], HT[3, 2]

    cy = jnp.sqrt(HT[0, 0] ** 2 + HT[0, 1] ** 2)

    # Non-singular branch:
    alpha_reg = jnp.arctan2(HT[1, 2], HT[2, 2])
    beta_reg = jnp.arctan2(-HT[0, 2], cy)
    gamma_reg = jnp.arctan2(HT[0, 1], HT[0, 0])

    # Gimbal-lock branch: put all rotation into alpha.
    alpha_sing = jnp.arctan2(-HT[2, 1], HT[1, 1])
    beta_sing = jnp.arctan2(-HT[0, 2], cy)
    gamma_sing = jnp.asarray(0.0, dtype=jnp.float64)

    is_singular = cy < _EPS
    alpha = jnp.where(is_singular, alpha_sing, alpha_reg)
    beta = jnp.where(is_singular, beta_sing, beta_reg)
    gamma = jnp.where(is_singular, gamma_sing, gamma_reg)

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return jnp.stack([RBx, RBy, RBz, zero, zero, zero, alpha, beta, gamma]).astype(
        jnp.float64
    )


# ---------------------------------------------------------------------------
# Homogeneous transform utilities
# ---------------------------------------------------------------------------


def _ht_inv(HT: jnp.ndarray) -> jnp.ndarray:
    """Inverse of a row-vector homogeneous transform, assuming the 3x3 block
    is a proper rotation (so the inverse is the transpose).

    Mirrors ``common::ht_inv``::

        htinv.topLeftCorner(3, 3) = R^T
        htinv.bottomLeftCorner(1, 3) = -t @ R^T
        htinv.col(3) = [0, 0, 0, 1]^T
    """
    R = HT[:3, :3]
    t = HT[3, :3]
    R_T = R.T
    t_inv = -t @ R_T
    out = jnp.zeros((4, 4), dtype=jnp.float64)
    out = out.at[:3, :3].set(R_T)
    out = out.at[3, :3].set(t_inv)
    out = out.at[3, 3].set(1.0)
    return out


def _hts_from_frames(ori: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """Build a homogeneous transform from an origin plus three frame atoms.

    Mirrors ``common::hts_from_frames``::

        xaxis = normalize(a - b)
        zaxis = normalize(xaxis x (c - a))
        yaxis = normalize(zaxis x xaxis)

    and stores them as the three rotation rows with ``ori`` as the last row.
    """
    xaxis = a - b
    xaxis = xaxis / jnp.linalg.norm(xaxis)
    zaxis = jnp.cross(xaxis, c - a)
    zaxis = zaxis / jnp.linalg.norm(zaxis)
    yaxis = jnp.cross(zaxis, xaxis)
    yaxis = yaxis / jnp.linalg.norm(yaxis)

    ht = jnp.zeros((4, 4), dtype=jnp.float64)
    ht = ht.at[0, :3].set(xaxis)
    ht = ht.at[1, :3].set(yaxis)
    ht = ht.at[2, :3].set(zaxis)
    ht = ht.at[3, :3].set(ori)
    ht = ht.at[3, 3].set(1.0)
    return ht


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def _forward_kin_sequential_impl(
    parent: jnp.ndarray,
    doftype: jnp.ndarray,
    kindof_raw: jnp.ndarray,
    root_transform: jnp.ndarray,
) -> jnp.ndarray:
    """Sequential fallback forward-kinematics routine.

    Builds each local HT then composes it with its parent's accumulated HT
    using a ``lax.fori_loop``. Because a ``KinForest`` is topologically
    ordered (``parent[i] < i``), a single sequential pass suffices. Used
    when no generational scan data is available (e.g. when ``parent`` is
    itself traced and we cannot pre-compute BFS depths).
    """
    n = kindof_raw.shape[0]

    # Pre-build every node's local (relative-to-parent) transform with a
    # vmapped switch over the per-node doftype.
    local_hts = jax.vmap(_local_transform)(kindof_raw, doftype)  # [N, 4, 4]

    # Seed the output buffer: node 0 is the forest root and gets the root
    # transform; every other slot starts at zero and will be overwritten.
    HTs_init = jnp.zeros((n, 4, 4), dtype=jnp.float64)
    HTs_init = HTs_init.at[0].set(root_transform)

    def step(i, HTs):
        # Local-frame transform for node i.
        local_i = local_hts[i]
        # Row-vector convention: HTs[i] = local_i @ HTs[parent].
        parent_ht = HTs[parent[i]]
        return HTs.at[i].set(local_i @ parent_ht)

    HTs = jax.lax.fori_loop(1, n, step, HTs_init)
    return HTs


def _forward_kin_generational_impl(
    parent: jnp.ndarray,
    doftype: jnp.ndarray,
    kindof_raw: jnp.ndarray,
    gen_atoms: jnp.ndarray,  # int32[n_gens, max_gen_size]
    gen_valid: jnp.ndarray,  # bool[n_gens, max_gen_size]
    root_transform: jnp.ndarray,
) -> jnp.ndarray:
    """Generational forward-kinematics routine using ``lax.scan``.

    At each scan step we receive one generation's worth of atom indices
    (padded to ``max_gen_size``) plus a boolean validity mask, gather
    each atom's local HT and its parent's accumulated HT, compose them,
    and scatter back into the HT buffer. Padding rows gather the root's
    HT (index 0), recompose it with the root's own local HT, and write
    back into slot 0 — but because the composed value is masked against
    the existing HTs[0] via ``jnp.where(valid, ...)``, the root's HT is
    left untouched.

    Within a generation every atom has its parent in an earlier
    generation, so the composition is safely parallelizable.
    """
    n = kindof_raw.shape[0]

    # Pre-build every node's local (relative-to-parent) transform.
    local_hts = jax.vmap(_local_transform)(kindof_raw, doftype)  # [N, 4, 4]

    HTs_init = jnp.zeros((n, 4, 4), dtype=jnp.float64)
    HTs_init = HTs_init.at[0].set(root_transform)

    def gen_step(HTs, gen_info):
        atoms_g, valid_g = gen_info  # [max_gen_size], [max_gen_size]
        parents_g = parent[atoms_g]  # [max_gen_size]

        # Gather local HTs and parent HTs for every atom in this generation.
        locals_g = local_hts[atoms_g]  # [max_gen_size, 4, 4]
        parent_hts = HTs[parents_g]    # [max_gen_size, 4, 4]

        # Compose local @ parent — row-vector convention.
        new_hts = jnp.einsum("gij,gjk->gik", locals_g, parent_hts)

        # Mask out padded rows so they don't clobber real HTs on scatter.
        existing = HTs[atoms_g]
        to_write = jnp.where(
            valid_g[:, None, None], new_hts, existing
        )
        HTs_new = HTs.at[atoms_g].set(to_write)
        return HTs_new, None

    HTs, _ = jax.lax.scan(gen_step, HTs_init, (gen_atoms, gen_valid))
    return HTs


def _maybe_build_gen_layout(
    parent_array,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute BFS-depth-based generational layout from a concrete parent
    array. Returns ``None`` if ``parent_array`` is abstract/traced, in
    which case the caller should fall back to the sequential path.
    """
    # Defer the import so we don't pay for it on every call; scan_ordering
    # pulls in a fair bit of numpy machinery.
    from jtmol.kinematics.scan_ordering import (
        _compute_bfs_depth,
        build_padded_gen_layout,
    )

    try:
        parent_np = np.asarray(parent_array, dtype=np.int32)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
        TypeError,
    ):
        return None

    n = parent_np.shape[0]
    if n == 0:
        return (
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 0), dtype=bool),
        )

    depth = _compute_bfs_depth(parent_np)
    # Sort atoms into generation-major / KFO-minor order, then build the
    # per-generation boundary table and the padded 2D layout.
    order = np.argsort(depth, kind="stable").astype(np.int32)
    sorted_depth = depth[order]
    max_depth = int(sorted_depth[-1])
    n_gens = max_depth + 1

    gens = np.zeros((n_gens + 1, 2), dtype=np.int32)
    for g in range(n_gens + 1):
        idx = int(np.searchsorted(sorted_depth, g, side="left"))
        gens[g, 0] = idx
        gens[g, 1] = idx
    gens[n_gens, 0] = n
    gens[n_gens, 1] = n

    gen_atoms, gen_valid = build_padded_gen_layout(order, gens)
    return gen_atoms, gen_valid


def _forward_kin_dispatch(
    kinforest: KinForest,
    kindof_raw: jnp.ndarray,
    root_transform: Optional[jnp.ndarray],
    gen_atoms: Optional[jnp.ndarray],
    gen_valid: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Shared body for ``forward_kin`` / ``forward_kin_hts``.

    Materializes the root transform, promotes inputs to the right
    dtypes, and picks the generational path when ``gen_atoms`` /
    ``gen_valid`` are provided (concrete). Returns the full HT stack.
    """
    if root_transform is None:
        root_transform_arr = _identity_ht()
    else:
        root_transform_arr = jnp.asarray(root_transform, dtype=jnp.float64)

    parent = jnp.asarray(kinforest.parent, dtype=jnp.int32)
    doftype = jnp.asarray(kinforest.doftype, dtype=jnp.int32)
    kindof_raw_j = jnp.asarray(kindof_raw, dtype=jnp.float64)

    if gen_atoms is not None and gen_valid is not None:
        return _forward_kin_generational_impl(
            parent,
            doftype,
            kindof_raw_j,
            jnp.asarray(gen_atoms, dtype=jnp.int32),
            jnp.asarray(gen_valid, dtype=jnp.bool_),
            root_transform_arr,
        )
    return _forward_kin_sequential_impl(
        parent, doftype, kindof_raw_j, root_transform_arr
    )


# JIT-compiled core dispatcher. ``gen_atoms`` / ``gen_valid`` come in as
# JAX arrays (or None); the arg structure is static per traced call.
_forward_kin_dispatch_jit = jax.jit(
    _forward_kin_dispatch, static_argnames=()
)


def forward_kin(
    kinforest: KinForest,
    kindof_raw: jnp.ndarray,
    root_transform: jnp.ndarray | None = None,
    *,
    scan_data: Optional["jtmol.kinematics.datatypes.KinForestScanData"] = None,
) -> jnp.ndarray:
    """DOFs -> Cartesian coordinates.

    Walks the ``KinForest`` generation by generation, composing each node's
    local transform onto its parent's accumulated transform. When the
    caller-supplied ``scan_data`` is absent and ``kinforest.parent`` is a
    concrete array, this function precomputes a BFS-depth partitioning
    and uses ``jax.lax.scan`` over generations so atoms sharing a
    generation are refolded in parallel via a single einsum.

    When ``kinforest.parent`` is traced (e.g. ``forward_kin`` is called
    inside a user-level ``jit``), the generational layout cannot be
    computed at trace time and the routine falls back to the sequential
    ``jax.lax.fori_loop`` path — still correct, just not
    generation-parallel.

    Parameters
    ----------
    kinforest : KinForest
        The tree. Only ``parent`` and ``doftype`` are read.
    kindof_raw : array [N, 9], float64
        DOFs for every node in KFO order. The first 4 columns are used for
        BOND nodes; all 9 for JUMP nodes; ROOT entries are ignored.
    root_transform : array [4, 4] or None
        The HT assigned to the forest root (node 0). Defaults to the 4x4
        identity, matching the C++ ``HomogeneousTransform::Identity()``.
    scan_data : KinForestScanData, optional
        Pre-built forward scan data (e.g. from
        :func:`jtmol.kinematics.scan_ordering.construct_kin_module_data`).
        When supplied, bypasses the internal BFS-depth computation and
        runs the generational scan directly.

    Returns
    -------
    coords : array [N, 3], float64
        Atomic coordinates in KFO order.
    """
    gen_atoms, gen_valid = _resolve_gen_layout(kinforest, scan_data)
    HTs = _forward_kin_dispatch_jit(
        kinforest, kindof_raw, root_transform, gen_atoms, gen_valid
    )
    return HTs[:, 3, :3]


def forward_kin_hts(
    kinforest: KinForest,
    kindof_raw: jnp.ndarray,
    root_transform: jnp.ndarray | None = None,
    *,
    scan_data: Optional["jtmol.kinematics.datatypes.KinForestScanData"] = None,
) -> jnp.ndarray:
    """Same as :func:`forward_kin` but returns the full HT stack ``[N, 4, 4]``.

    Useful for derivative pipelines or when downstream code needs the local
    basis at each atom.
    """
    gen_atoms, gen_valid = _resolve_gen_layout(kinforest, scan_data)
    return _forward_kin_dispatch_jit(
        kinforest, kindof_raw, root_transform, gen_atoms, gen_valid
    )


def _resolve_gen_layout(
    kinforest: KinForest,
    scan_data,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Return ``(gen_atoms, gen_valid)`` suitable for the generational
    scan, or ``(None, None)`` to fall back to the sequential path.

    Resolution order:

    1. If ``scan_data`` is supplied, build the padded layout from its
       ``nodes`` / ``gens`` arrays.
    2. Otherwise attempt to build a BFS-depth layout from
       ``kinforest.parent``. If that array is still abstract (traced),
       return ``(None, None)`` so the caller falls back to the
       sequential fori_loop.
    """
    if scan_data is not None:
        # Convert to numpy; scan_data.nodes / .gens are small int arrays
        # and are cheap to fetch.
        try:
            nodes_np = np.asarray(scan_data.nodes, dtype=np.int32)
            gens_np = np.asarray(scan_data.gens, dtype=np.int32)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
            TypeError,
        ):
            # Can't materialize under a parent-jit; fall back.
            return None, None
        from jtmol.kinematics.scan_ordering import build_padded_gen_layout

        gen_atoms, gen_valid = build_padded_gen_layout(nodes_np, gens_np)
        return jnp.asarray(gen_atoms, dtype=jnp.int32), jnp.asarray(
            gen_valid, dtype=jnp.bool_
        )

    layout = _maybe_build_gen_layout(kinforest.parent)
    if layout is None:
        return None, None
    gen_atoms, gen_valid = layout
    if gen_atoms.shape[0] == 0:
        # Empty forest: fall back to sequential to avoid pathological
        # scan shapes (n_gens == 0 arrays throw downstream).
        return None, None
    return jnp.asarray(gen_atoms, dtype=jnp.int32), jnp.asarray(
        gen_valid, dtype=jnp.bool_
    )


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------


def _select_inverse(lclHT: jnp.ndarray, doftype: jnp.ndarray) -> jnp.ndarray:
    """Dispatch a local HT to the correct inverse transform by doftype.

    Returns a 9-vector; the BOND branch pads its 4-vector with zeros, and
    the ROOT branch is all zeros (matching ``dofs[i] = KintreeDof::Constant(0)``
    in ``InverseKinDispatch`` for ROOT nodes).
    """

    def _root(_ht):
        return jnp.zeros(9, dtype=jnp.float64)

    def _jump(ht):
        return invJumpTransform(ht)

    def _bond(ht):
        dof4 = invBondTransform(ht)
        return jnp.concatenate([dof4, jnp.zeros(5, dtype=jnp.float64)])

    return jax.lax.switch(doftype, (_root, _jump, _bond), lclHT)


def _inverse_kin_impl(
    coords: jnp.ndarray,
    parent: jnp.ndarray,
    frame_x: jnp.ndarray,
    frame_y: jnp.ndarray,
    frame_z: jnp.ndarray,
    doftype: jnp.ndarray,
) -> jnp.ndarray:
    """Inner (traced) inverse-kinematics routine. Mirrors the two-kernel
    structure of ``InverseKinDispatch::f`` in ``compiled.cpu.cpp``:

    1. ``hts_from_frames`` for every non-root node (node 0 is identity).
    2. ``lclHT = HTs[i] @ ht_inv(HTs[parent[i]])`` and then
       ``invBondTransform`` / ``invJumpTransform`` per doftype.

    Both passes are embarrassingly parallel across nodes, so we use ``vmap``.
    """
    # --- Pass 1: build global HT for every atom ---
    # Gather frame-atom positions. Invalid frame indices on the root node
    # still produce finite numbers because we overwrite HTs[0] with identity
    # below; the root's frame contribution is never consumed downstream.
    a = coords[frame_x]
    b = coords[frame_y]
    c = coords[frame_z]
    HTs = jax.vmap(_hts_from_frames)(coords, a, b, c)
    # Force node 0 (the forest root) to identity to match the C++ behavior.
    HTs = HTs.at[0].set(_identity_ht())

    # --- Pass 2: local HT and inverse transform per node ---
    parent_HTs = HTs[parent]
    parent_inv = jax.vmap(_ht_inv)(parent_HTs)
    # Row-vector convention: lclHT = HTs @ ht_inv(HTs_parent).
    lcl_HTs = jnp.einsum("nij,njk->nik", HTs, parent_inv)
    dofs = jax.vmap(_select_inverse)(lcl_HTs, doftype)
    return dofs


@partial(jax.jit, static_argnames=())
def inverse_kin(kinforest: KinForest, coords: jnp.ndarray) -> KinDOF:
    """Cartesian coordinates -> DOFs.

    For each atom: build the atom's frame HT from its frame_x/y/z neighbors
    via Gram-Schmidt, compute the local transform relative to its parent's
    frame, and recover the DOF 9-vector (bond or jump) from that local HT.

    Parameters
    ----------
    kinforest : KinForest
        Provides ``parent``, ``frame_x``, ``frame_y``, ``frame_z``, and
        ``doftype``.
    coords : array [N, 3]
        Atomic coordinates in KFO order.

    Returns
    -------
    KinDOF
        With ``raw`` of shape ``[N, 9]`` and dtype float64.
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)
    parent = jnp.asarray(kinforest.parent, dtype=jnp.int32)
    frame_x = jnp.asarray(kinforest.frame_x, dtype=jnp.int32)
    frame_y = jnp.asarray(kinforest.frame_y, dtype=jnp.int32)
    frame_z = jnp.asarray(kinforest.frame_z, dtype=jnp.int32)
    doftype = jnp.asarray(kinforest.doftype, dtype=jnp.int32)

    raw = _inverse_kin_impl(coords, parent, frame_x, frame_y, frame_z, doftype)
    return KinDOF(raw=raw)


# Upstream uses the camelCase name ``inverseKin`` as the public wrapper in
# ``tmol/kinematics/operations.py``. Preserve it for drop-in compatibility.
inverseKin = inverse_kin


__all__ = [
    "forward_kin",
    "forward_kin_hts",
    "inverse_kin",
    "inverseKin",
    "bondTransform",
    "jumpTransform",
    "invBondTransform",
    "invJumpTransform",
]
