"""Cell-list neighbor finder for short-range non-bonded potentials.

The dense-pair path in ``LJLKEnergyTerm`` materialises a
``[P, N, N]`` distance matrix. For small systems (~1200 atoms / 1UBQ)
that's fine; for ~5000 atoms the matrix is ~25M pairs and memory
becomes the dominant cost — every precomputed auxiliary (bonded path
length, parameter broadcasts, LJ/LK per-region branches) inherits the
same quadratic footprint and grad-tape memory blows up quickly.

Upstream tmol addresses this with tiled shared-memory CUDA kernels; in
pure JAX the equivalent optimisation is a classic Verlet / cell-list
style neighbour finder. For a ~6 A cutoff the expected neighbour count
per atom is bounded (~60-150 for dense protein cores), so the fast-path
scales as ``O(N * k_max)`` rather than ``O(N^2)``.

Design
======

1. **Host-side construction.** Cell assignment is pure-numpy bucketing
   into a 3D spatial grid. The grid cell size matches the cutoff, so
   the set of candidate neighbours for any atom is the union of its
   own cell plus the 26 neighbour cells (3x3x3). Running the build
   on the host keeps the JAX trace small and avoids a
   dynamic-shape / ``jnp.unique`` dance that would otherwise
   recompile on every call.

2. **Padded output layout.** We return a dense
   ``[P, N, max_nbr]`` index buffer. This is the friendliest layout for
   JAX: a straightforward ``take_along_axis`` gathers per-atom neighbour
   params, and a boolean mask zeroes padded slots before the sum. We
   pad (or clip if the true count exceeds ``max_nbr_soft_cap``) rather
   than using segment_sum so that the compiled hot loop has static
   shapes — recompilation on every score call would swamp any speedup.

3. **Half-list by default.** Non-bonded sums in Rosetta-style score
   functions are over the upper triangle (``i < j``). The cell-list
   builder respects that convention so the downstream summation
   doesn't need an additional symmetrise / divide-by-two step and
   no atom pair is counted twice.

The functions below are designed to slot in behind an
``use_neighbor_list`` flag on :class:`LJLKEnergyTerm` (and any other
non-bonded term that follows the same ``i, j`` pair contract). The
dense path is kept verbatim as the default so bit-exact parity with
upstream is preserved.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------- #
# Build                                                                    #
# ---------------------------------------------------------------------- #


def build_cell_list(
    coords,
    cutoff: float = 6.0,
    atom_is_real=None,
    half_list: bool = True,
    max_nbr_soft_cap: int = 512,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a padded per-atom neighbour list from 3D coordinates.

    Parameters
    ----------
    coords
        ``float32[n_poses, max_n_atoms_per_pose, 3]``. Either a numpy
        array or a ``jax.Array``; we pull it onto the host because cell
        bucketing is trivially parallelised on CPU and this is a
        one-time setup step (or recomputable whenever the geometry
        drifts enough to invalidate the current list).
    cutoff
        Interaction cutoff in Angstroms. Defaults to 6.0, which matches
        the far cpoly cutoff used by both LJ and LK.
    atom_is_real
        Optional ``bool[P, N]`` mask of real atoms. Padded atoms are
        excluded from the neighbour list entirely.
    half_list
        If ``True`` (default) keep only ``j > i`` pairs so the consumer
        doesn't double-count. Set ``False`` for potentials that sum
        asymmetric contributions (e.g. LK is symmetric but its naive
        i->j decomposition isn't).
    max_nbr_soft_cap
        Upper bound on per-atom neighbour count used to pre-size the
        padded output. We size the actual ``max_nbr`` axis to the true
        maximum over all atoms and clip at this cap to keep memory
        bounded on pathological densely-packed systems.

    Returns
    -------
    ``(neighbor_idx, n_neighbors, valid_mask)``

    * ``neighbor_idx``: ``int32[P, N, max_nbr]``. For atom ``i`` in
      pose ``p`` the first ``n_neighbors[p, i]`` entries are the
      indices of atoms ``j`` within ``cutoff`` (and ``j > i`` if
      ``half_list``). Remaining slots are ``0`` (dummy; combined with
      the mask they contribute nothing).
    * ``n_neighbors``: ``int32[P, N]``. True neighbour count per atom.
    * ``valid_mask``: ``bool[P, N, max_nbr]``. True on real neighbour
      entries, False on padded slots.

    Notes
    -----
    The grid cell size equals ``cutoff`` — the canonical choice that
    minimises candidate checks (at ~14x fewer pairs compared to the
    dense case on a well-distributed system) while keeping the
    3x3x3 stencil trivially correct.
    """
    coords_np = np.asarray(coords, dtype=np.float32)
    if coords_np.ndim != 3 or coords_np.shape[-1] != 3:
        raise ValueError(
            f"coords must be [P, N, 3], got shape {coords_np.shape}"
        )
    P, N, _ = coords_np.shape

    if atom_is_real is None:
        real_np = np.ones((P, N), dtype=bool)
    else:
        real_np = np.asarray(atom_is_real, dtype=bool)
        if real_np.shape != (P, N):
            raise ValueError(
                f"atom_is_real must be [P, N] = ({P}, {N}), got {real_np.shape}"
            )

    cutoff_sq = float(cutoff) * float(cutoff)

    # First pass: collect raw (variable-length) neighbour lists per pose
    # so we can compute the true padded width before we allocate. For
    # typical proteins (~50-150 nbrs per atom at 6 A), materialising the
    # lists as Python lists of numpy arrays is cheap.
    per_pose_lists: list[list[np.ndarray]] = []
    per_pose_counts: list[np.ndarray] = []
    max_nbr_seen = 0

    for p in range(P):
        xyz = coords_np[p]
        real = real_np[p]
        real_idx = np.nonzero(real)[0].astype(np.int32)
        if real_idx.size == 0:
            per_pose_lists.append([np.empty(0, dtype=np.int32) for _ in range(N)])
            per_pose_counts.append(np.zeros(N, dtype=np.int32))
            continue

        real_xyz = xyz[real_idx]

        # Grid origin: min corner of real atoms' bounding box. Using the
        # real-atom bounding box (rather than a fixed origin) keeps the
        # cell count tight on small systems and avoids overflow on
        # systems that happen to live far from the origin.
        mn = real_xyz.min(axis=0)
        # Add a 1-cell halo on each side so the edge-cell stencil always
        # lives inside the grid; simplifies indexing.
        cell_size = float(cutoff)
        # Number of cells along each axis.
        grid_dims = np.maximum(
            np.ceil((real_xyz.max(axis=0) - mn) / cell_size).astype(np.int32) + 1,
            1,
        )
        Gx, Gy, Gz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

        # Bucket: for each real atom, compute (cx, cy, cz) cell index.
        cell_xyz = np.floor((real_xyz - mn) / cell_size).astype(np.int32)
        # Clamp into the grid (shouldn't trigger but defensive against
        # floating-point ceil quirks).
        cell_xyz[:, 0] = np.clip(cell_xyz[:, 0], 0, Gx - 1)
        cell_xyz[:, 1] = np.clip(cell_xyz[:, 1], 0, Gy - 1)
        cell_xyz[:, 2] = np.clip(cell_xyz[:, 2], 0, Gz - 1)
        cell_flat = (
            cell_xyz[:, 0] * (Gy * Gz)
            + cell_xyz[:, 1] * Gz
            + cell_xyz[:, 2]
        ).astype(np.int64)

        # Sort atoms by cell so each cell becomes a contiguous range;
        # we then build per-cell atom-index lists with a single
        # searchsorted pass.
        order = np.argsort(cell_flat, kind="stable")
        cell_sorted = cell_flat[order]
        atoms_sorted = real_idx[order]  # original atom indices
        xyz_sorted = real_xyz[order]

        # Boundaries of each cell in the sorted list.
        n_cells = Gx * Gy * Gz
        cell_start = np.searchsorted(
            cell_sorted, np.arange(n_cells, dtype=np.int64), side="left"
        )
        cell_end = np.searchsorted(
            cell_sorted, np.arange(n_cells, dtype=np.int64), side="right"
        )

        # For each real atom, scan the 3x3x3 neighbour-cell stencil.
        nbr_lists: list[np.ndarray] = [np.empty(0, dtype=np.int32)] * N
        counts = np.zeros(N, dtype=np.int32)

        # Precompute stencil flat offsets (27 entries). We index cells by
        # (cx, cy, cz) but want to avoid Python loops in the hot inner
        # loop — we inline the 3x3x3 offsets directly.
        stencil = np.array(
            [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            ],
            dtype=np.int32,
        )

        for k in range(atoms_sorted.shape[0]):
            i_atom = int(atoms_sorted[k])
            xi = xyz_sorted[k]
            cxi, cyi, czi = cell_xyz[order][k]

            # Collect candidate atom indices from the 27-cell stencil.
            candidates = []
            for dx, dy, dz in stencil:
                nx = cxi + dx
                ny = cyi + dy
                nz = czi + dz
                if nx < 0 or nx >= Gx or ny < 0 or ny >= Gy or nz < 0 or nz >= Gz:
                    continue
                nc = int(nx * (Gy * Gz) + ny * Gz + nz)
                lo = int(cell_start[nc])
                hi = int(cell_end[nc])
                if hi > lo:
                    candidates.append((lo, hi))

            if not candidates:
                continue

            # Flatten candidate atom indices and xyz in one go.
            ranges = [np.arange(lo, hi) for lo, hi in candidates]
            cand_idx = np.concatenate(ranges)
            cand_atoms = atoms_sorted[cand_idx]
            cand_xyz = xyz_sorted[cand_idx]

            # Distance filter.
            d = cand_xyz - xi
            d_sq = (d * d).sum(axis=-1)
            within = d_sq <= cutoff_sq

            # Half-list filter: only keep j > i (avoid double counting).
            if half_list:
                within &= cand_atoms > i_atom
            else:
                # Always exclude self.
                within &= cand_atoms != i_atom

            nbrs = cand_atoms[within].astype(np.int32)
            # Bound per-atom count by the soft cap (trims the unlikely
            # worst-case from a pathological system).
            if nbrs.size > max_nbr_soft_cap:
                # Keep the spatially closest ones.
                kept_sq = d_sq[within]
                keep_order = np.argsort(kept_sq)[:max_nbr_soft_cap]
                nbrs = nbrs[keep_order]

            nbr_lists[i_atom] = nbrs
            counts[i_atom] = nbrs.size
            if nbrs.size > max_nbr_seen:
                max_nbr_seen = int(nbrs.size)

        per_pose_lists.append(nbr_lists)
        per_pose_counts.append(counts)

    # Allocate padded outputs. ``max_nbr_seen == 0`` means no
    # interactions anywhere; pad to width 1 so downstream shapes remain
    # well-defined.
    max_nbr = max(max_nbr_seen, 1)
    neighbor_idx = np.zeros((P, N, max_nbr), dtype=np.int32)
    valid_mask = np.zeros((P, N, max_nbr), dtype=bool)

    for p in range(P):
        nbr_lists = per_pose_lists[p]
        for i_atom in range(N):
            nbrs = nbr_lists[i_atom]
            k = int(nbrs.size)
            if k:
                neighbor_idx[p, i_atom, :k] = nbrs
                valid_mask[p, i_atom, :k] = True

    n_neighbors = np.stack(per_pose_counts, axis=0).astype(np.int32)

    return (
        jnp.asarray(neighbor_idx),
        jnp.asarray(n_neighbors),
        jnp.asarray(valid_mask),
    )


# ---------------------------------------------------------------------- #
# Gather helpers                                                           #
# ---------------------------------------------------------------------- #


def gather_neighbor_values(values: jax.Array, nbr_idx: jax.Array) -> jax.Array:
    """Gather per-atom values to ``[P, N, max_nbr, ...]`` along the neighbour axis.

    ``values`` can be any ``[P, N, *trailing]`` pytree leaf; we use
    ``take_along_axis`` on the atom axis after broadcasting the
    neighbour index. For plain scalar-per-atom parameters this is a
    two-line operation; for 3-vector coords we reshape the index once.
    """
    v = jnp.asarray(values)
    P, N = v.shape[0], v.shape[1]
    max_nbr = nbr_idx.shape[-1]

    if v.ndim == 2:
        # [P, N] -> [P, N, max_nbr]
        return jnp.take_along_axis(v[:, :, None], nbr_idx[:, :, :], axis=1)[..., 0] \
            if False else _take_2d(v, nbr_idx)
    if v.ndim == 3:
        # [P, N, D] -> [P, N, max_nbr, D]
        return _take_3d(v, nbr_idx)
    raise ValueError(f"gather_neighbor_values: unsupported rank {v.ndim}")


def _take_2d(v: jax.Array, nbr_idx: jax.Array) -> jax.Array:
    """Gather ``v[P, N]`` -> ``[P, N, max_nbr]`` using neighbour indices.

    Uses a vmap over poses of ``jnp.take`` along the atom axis; the
    per-pose gather is the simplest thing that fuses cleanly under jit.
    """
    # nbr_idx: [P, N, max_nbr]. We want out[p, i, k] = v[p, nbr_idx[p, i, k]].
    return jax.vmap(lambda vp, ip: vp[ip], in_axes=(0, 0))(v, nbr_idx)


def _take_3d(v: jax.Array, nbr_idx: jax.Array) -> jax.Array:
    """Gather ``v[P, N, D]`` -> ``[P, N, max_nbr, D]`` using neighbour indices."""
    return jax.vmap(lambda vp, ip: vp[ip], in_axes=(0, 0))(v, nbr_idx)


def gather_pair_matrix(
    matrix: jax.Array, nbr_idx: jax.Array
) -> jax.Array:
    """Gather a dense ``[P, N, N]`` pair matrix to ``[P, N, max_nbr]``.

    Needed for things like the bonded-path-length tensor: it's still
    cheap to build the ``[P, N, N]`` int32 matrix (4 * N^2 bytes; 25 MB
    at 5k atoms) but once the energy evaluation itself is neighbour-list
    based we should also compress this tensor down to the same layout
    to avoid the dense-pair memory cost inside the hot loop.
    """
    # matrix: [P, N, N], nbr_idx: [P, N, max_nbr]
    # out[p, i, k] = matrix[p, i, nbr_idx[p, i, k]]
    def _per_pose(mp, ip):
        # mp: [N, N], ip: [N, max_nbr]
        # Use take_along_axis on the inner N axis.
        return jnp.take_along_axis(mp, ip, axis=1)

    return jax.vmap(_per_pose, in_axes=(0, 0))(matrix, nbr_idx)


# ---------------------------------------------------------------------- #
# Pairwise sum                                                             #
# ---------------------------------------------------------------------- #


def pairwise_sum_with_nbrs(
    coords: jax.Array,
    params_i: Any,
    params_j: Any,
    nbr_idx: jax.Array,
    nbr_mask: jax.Array,
    pair_fn: Callable[[jax.Array, jax.Array, Any, Any], jax.Array],
) -> jax.Array:
    """Sum ``pair_fn`` over the neighbour list per pose.

    This is the analogue of the dense
    ``jnp.where(mask, pair_fn(dist, params_i, params_j), 0.0).sum((-1, -2))``
    pattern used by ``LJLKEnergyTerm``, specialised to the neighbour
    list layout. No ``[P, N, N]`` intermediate is ever materialised.

    Parameters
    ----------
    coords
        ``[P, N, 3]`` coordinates.
    params_i
        A pytree-like object whose leaves are either ``[P, N]`` or
        ``[P, N, ...]`` arrays (the per-atom LJ/LK params). We expand
        them to ``[P, N, 1, ...]`` on entry so they broadcast against
        the gathered neighbour axis.
    params_j
        Same shape contract as ``params_i``. Each leaf is gathered
        along the neighbour axis to ``[P, N, max_nbr, ...]``.
    nbr_idx
        ``int32[P, N, max_nbr]``.
    nbr_mask
        ``bool[P, N, max_nbr]``.
    pair_fn
        ``pair_fn(coords_i, coords_j, params_i_exp, params_j_gath) ->
        [P, N, max_nbr]`` per-pair scalar value (already weighted by
        any bonded-path-length factor). See ``lj_atr_rep`` /
        ``lk_isotropic_pair`` for the expected signature; the
        ``pair_fn`` here is typically a small lambda that wires those
        upstream potentials to the neighbour-list layout.

    Returns
    -------
    ``float32[P]`` — the summed pair energy per pose.
    """
    # Gather neighbour coords. [P, N, max_nbr, 3]
    coords_j = _take_3d(coords, nbr_idx)
    # [P, N, 1, 3] so it broadcasts against coords_j.
    coords_i = coords[:, :, None, :]

    # Expand params_i along a new neighbour axis; gather params_j along
    # the atom axis.
    expanded_i = _expand_params_i(params_i)
    gathered_j = _gather_params_j(params_j, nbr_idx)

    V = pair_fn(coords_i, coords_j, expanded_i, gathered_j)  # [P, N, max_nbr]
    V = jnp.where(nbr_mask, V, jnp.zeros_like(V))
    return V.sum(axis=(-1, -2))


def _expand_params_i(params: Any) -> Any:
    """Expand each leaf of ``params`` to ``[P, N, 1, ...]``.

    Mirrors the broadcasting used by the dense path: we want atom-``i``
    parameters to broadcast against the neighbour axis. We walk the
    attributes of a simple object (struct-of-arrays pattern) rather
    than using ``jax.tree_util.tree_map`` so the caller doesn't have
    to register anything as a pytree.
    """
    return _map_leaves(params, lambda a: a[:, :, None])


def _gather_params_j(params: Any, nbr_idx: jax.Array) -> Any:
    """Gather each leaf of ``params`` along the atom axis using ``nbr_idx``.

    Scalar-per-atom arrays become ``[P, N, max_nbr]``; vector-per-atom
    arrays (if any) become ``[P, N, max_nbr, D]``.
    """
    def _do(a):
        a = jnp.asarray(a)
        if a.ndim == 2:
            return _take_2d(a, nbr_idx)
        if a.ndim == 3:
            return _take_3d(a, nbr_idx)
        raise ValueError(
            f"neighbor-list gather: unsupported param rank {a.ndim}"
        )

    return _map_leaves(params, _do)


def _map_leaves(params: Any, fn: Callable[[jax.Array], jax.Array]) -> Any:
    """Apply ``fn`` to every attribute leaf on ``params`` in-place-ish.

    ``params`` is expected to be a simple attrs-style object whose
    fields are all ``jax.Array``. We return a new ``SimpleNamespace``
    so the caller doesn't need to care about the concrete type; any
    attribute access in ``pair_fn`` (``params_i.lj_radius``, etc.)
    will work.
    """
    from types import SimpleNamespace

    out = SimpleNamespace()
    # If params is a dict, iterate directly; otherwise introspect.
    if isinstance(params, dict):
        for k, v in params.items():
            setattr(out, k, fn(v))
        return out

    # attrs / SimpleNamespace / anything with __dict__ or __slots__.
    if hasattr(params, "__dict__") and params.__dict__:
        items = params.__dict__.items()
    else:
        # Fall back to ``dir``-style scan, skipping dunder/private and
        # callable attributes.
        items = [
            (name, getattr(params, name))
            for name in dir(params)
            if not name.startswith("_")
            and not callable(getattr(params, name, None))
        ]

    for k, v in items:
        setattr(out, k, fn(v))
    return out


__all__ = [
    "build_cell_list",
    "gather_neighbor_values",
    "gather_pair_matrix",
    "pairwise_sum_with_nbrs",
]
