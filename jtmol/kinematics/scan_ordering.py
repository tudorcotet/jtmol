"""Scan-ordering / generational-segmented-scan construction, JAX port.

Upstream reference: ``tmol/kinematics/scan_ordering.py``.

Upstream's ``scan_ordering`` module does two very different things in one
file:

1. Given a ``KinForest`` (atom-level parent pointers), partition the forest
   into *generational scan paths*: depth-indexed chains of atoms that can
   be processed in parallel with a GPU segmented-scan kernel. Upstream's
   ``get_scans`` / ``KinForestScanOrdering.calculate_from_kinforest`` do
   this with a numba-jitted numpy routine.

2. Given a ``FoldForest`` plus a ``PoseStack`` (block-level topology),
   *build* the atom-level ``KinForest`` by weaving per-block-type scan-path
   segments together, then compute its generational ordering. Upstream's
   ``construct_kin_module_data_for_pose`` plus the huge
   ``_annotate_block_type_with_gen_scan_path_segs`` block-type machinery
   do this; most of the per-pose work runs inside C++/CUDA kernels
   reached through ``tmol.kinematics.compiled``.

This JAX port implements both (1) with full per-generation parallelism and
(2) with a simpler BFS-depth-based partitioning that delegates the
atom-level KinForest construction to :meth:`FoldForest.build_kinforest`.

Three scan modes
----------------

* ``"bfs_depth"`` (default) — atoms grouped by BFS depth from the root,
  one scan per atom within a generation. For a KinForest with parent
  invariant ``parent[i] <= i``, atoms at depth ``d`` have parents at
  depth ``d-1``, so all atoms in generation ``d`` can be refolded in
  parallel once generation ``d-1`` is complete. This maps cleanly onto
  ``jax.lax.scan`` over generations with ``jax.vmap`` over the atoms
  within a generation.

* ``"full"`` — faithful port of upstream's DFS-with-descendant-priority
  algorithm (``get_scans``). Uses long "primary paths" (a.k.a. scan
  segments) within each generation so the generation count is tiny
  (typically 4-10 for a full pose stack) but each segment is sequential.
  Suitable for a real segmented-scan GPU kernel, not yet wired into the
  JAX forward kinematics.

* ``"degenerate"`` — one generation containing every atom, one scan
  starting at 0. Matches the sequential ``fori_loop`` implementation and
  is kept as an escape hatch.

All three modes return ``KinForestScanData`` with identical field
structure and identical dtypes, so a caller can swap between them freely.

In addition to ``KinForestScanData`` (flat ``nodes``/``scans``/``gens``),
:func:`build_padded_gen_layout` converts the flat layout into
``(gen_atoms, gen_valid)`` 2D arrays (padded per generation) that
``jtmol.kinematics.operations.forward_kin`` consumes inside a
``jax.lax.scan`` over generations.

NOTE: The full block-type-annotation machinery from upstream
(``_annotate_block_type_with_gen_scan_path_segs`` and friends) is *not*
ported here. Those functions build per-block-type scan-path segments
that are consumed by the C++ kernel ``get_kinforest_scans_from_stencils2``
when weaving together the atom-level KinForest for a whole PoseStack.
This port instead delegates KinForest assembly to
:meth:`FoldForest.build_kinforest`, a simpler per-pose walker that
produces a topologically-valid atom-level KinForest (every atom has a
parent earlier in KFO) but does not optimize the generation count.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from jtmol.kinematics.datatypes import (
    KinForest,
    KinForestScanData,
    KinematicModuleData,
)


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

_INT = np.int32


# ---------------------------------------------------------------------------
# Child-list construction (full port of upstream's `get_children`)
# ---------------------------------------------------------------------------


def _get_children(parents: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a CSR-style child list from a parent-pointer array.

    Parameters
    ----------
    parents : int32[N]
        ``parents[i]`` is the KFO index of atom ``i``'s parent; root atoms
        satisfy ``parents[i] == i``. Every non-root atom must satisfy
        ``parents[i] < i`` (the KinForest topological-ordering invariant).

    Returns
    -------
    n_descendents : int32[N]
        Total number of descendants of each node (inclusive of self).
    child_list_span : int32[N, 2]
        ``(start, stop)`` slice into ``child_list`` for each parent.
    child_list : int32[N]
        Flat concatenation of per-parent child indices, referenced by
        ``child_list_span``.

    This is a faithful numpy port of upstream's numba-jitted ``get_children``.
    """

    parents = np.ascontiguousarray(parents, dtype=_INT)
    nelts = parents.shape[0]

    # Pass 1: count immediate children per parent.
    n_immediate_children = np.zeros(nelts, dtype=_INT)
    for i in range(nelts):
        p = int(parents[i])
        if p > i:
            raise ValueError(
                f"Invalid kinematic tree ordering at atom {i}: parent index "
                f"{p} >= child index {i}."
            )
        if p == i:  # root
            continue
        n_immediate_children[p] += 1

    # Pass 2: mark CSR spans.
    child_list_span = np.empty((nelts, 2), dtype=_INT)
    child_list_span[0, 0] = 0
    child_list_span[0, 1] = n_immediate_children[0]
    for i in range(1, nelts):
        child_list_span[i, 0] = child_list_span[i - 1, 1]
        child_list_span[i, 1] = child_list_span[i, 0] + n_immediate_children[i]

    # Pass 3: fill the child list in reverse (so children stay in ascending
    # index order) and tally descendant counts bottom-up.
    child_list = np.full(nelts, -1, dtype=_INT)
    n_descendents = np.ones(nelts, dtype=_INT)
    remaining = n_immediate_children.copy()
    for i in range(nelts - 1, 0, -1):
        p = int(parents[i])
        if p == i:
            continue
        n_descendents[p] += n_descendents[i]
        slot = int(child_list_span[p, 0]) + int(remaining[p]) - 1
        child_list[slot] = i
        remaining[p] -= 1

    return n_descendents, child_list_span, child_list


# ---------------------------------------------------------------------------
# Generational scan construction (full port of upstream's `get_scans`)
# ---------------------------------------------------------------------------


def get_scans(
    parents: np.ndarray, roots: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition a KinForest into generational scan paths.

    Faithful numpy port of upstream's numba-jitted ``get_scans``. See that
    function's docstring for the high-level idea; in brief:

    * Start from ``roots`` (generation 0).
    * Each active root extends a scan through the child with the greatest
      descendant count (the "primary child"). Every other child becomes
      the root of a new scan in the next generation.
    * Scans within a generation are independent and can be processed by
      a segmented-scan kernel; generation ``k`` depends only on
      generations ``0..k-1``.

    Parameters
    ----------
    parents : int32[N]
        KinForest parent-pointer array (KFO indexed, KFO valued).
    roots : int32[R]
        KFO indices of the initial roots (generation-0 entry points).

    Returns
    -------
    nodes : int32[K]
        Concatenated node indices for every scan, in traversal order.
    scan_starts : int32[S]
        Indices into ``nodes`` where each individual scan begins.
    gen_starts : int32[G + 1, 2]
        ``gen_starts[g, 0]`` indexes into ``nodes`` where generation
        ``g`` starts; ``gen_starts[g, 1]`` indexes into ``scan_starts``.
        A sentinel row at ``gen_starts[-1]`` holds the total ``(len(nodes),
        len(scan_starts))`` so downstream users can slice without branching.
    """

    parents = np.ascontiguousarray(parents, dtype=_INT)
    roots = np.ascontiguousarray(roots, dtype=_INT)

    nelts = parents.shape[0]
    if nelts == 0:
        return (
            np.zeros(0, dtype=_INT),
            np.zeros(0, dtype=_INT),
            np.zeros((1, 2), dtype=_INT),
        )

    n_descendents, child_list_span, child_list = _get_children(parents)

    # Over-allocate upper bounds; trim at the end.
    nodes = np.full(4 * nelts, -1, dtype=_INT)
    scan_starts = np.full(nelts, -1, dtype=_INT)
    gen_starts = np.full((nelts + 1, 2), -1, dtype=_INT)

    gen_idx = 0
    scan_idx = 0
    node_idx = 0

    active_front = np.full(4 * nelts, -1, dtype=_INT)
    n_active_front = roots.shape[0]
    active_front[:n_active_front] = roots

    marked = np.zeros(nelts, dtype=_INT)
    # The conventional global root at index 0 is pre-marked so we never try
    # to root a scan from it -- it is already handled via the `roots` input.
    marked[0] = 1

    while not np.all(marked):
        gen_starts[gen_idx, 0] = node_idx
        gen_starts[gen_idx, 1] = scan_idx

        for i in range(n_active_front):
            cur_root = int(active_front[i])

            start, stop = (
                int(child_list_span[cur_root, 0]),
                int(child_list_span[cur_root, 1]),
            )
            for j in range(start, stop):
                expanded = int(child_list[j])
                if marked[expanded] != 0:
                    # Already drawn in an earlier generation's primary path.
                    continue

                # Open a new scan starting at cur_root, extended into
                # `expanded` and as far as we can go along primary children.
                scan_starts[scan_idx] = node_idx - int(gen_starts[gen_idx, 0])
                scan_idx += 1

                nodes[node_idx] = cur_root
                node_idx += 1

                while True:
                    marked[expanded] = 1
                    nodes[node_idx] = expanded
                    node_idx += 1

                    e_start = int(child_list_span[expanded, 0])
                    e_stop = int(child_list_span[expanded, 1])
                    if e_start == e_stop:
                        # Leaf reached.
                        break

                    # Extend to the child with the most descendants.
                    best = int(child_list[e_start])
                    for k in range(e_start + 1, e_stop):
                        cand = int(child_list[k])
                        if n_descendents[cand] > n_descendents[best]:
                            best = cand
                    expanded = best

        # Promote every node emitted this generation to the next active
        # front. Nodes with only a primary child will be filtered out by
        # the "already marked" guard when expanded next round.
        if int(gen_starts[gen_idx, 1]) < scan_idx:
            last_gen_start = int(gen_starts[gen_idx, 0])
            active_front.fill(-1)
            n_active_front = node_idx - last_gen_start
            active_front[:n_active_front] = nodes[last_gen_start:node_idx]

        gen_idx += 1

    # Sentinel row: total lengths, makes downstream slicing branchless.
    gen_starts[gen_idx, 0] = node_idx
    gen_starts[gen_idx, 1] = scan_idx
    gen_idx += 1

    return (
        np.ascontiguousarray(nodes[:node_idx]),
        np.ascontiguousarray(scan_starts[:scan_idx]),
        np.ascontiguousarray(gen_starts[:gen_idx]),
    )


# ---------------------------------------------------------------------------
# Forward / backward scan-data construction
# ---------------------------------------------------------------------------


def _reverse_scan_data(
    nodes: np.ndarray, scan_starts: np.ndarray, gen_starts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip a forward scan into a backward (leaves-to-root) scan.

    Faithful numpy port of the reverse-scan bookkeeping inside
    ``KinForestScanOrdering.calculate_from_kinforest``.
    """

    n_nodes = nodes.shape[0]
    n_scans = scan_starts.shape[0]

    nodes_r = np.ascontiguousarray(np.flipud(nodes))

    gen_starts_r = np.zeros_like(gen_starts)
    # The first (generation) row stays at zero; later rows are the
    # complements of the forward gen boundaries, read from the tail.
    if gen_starts.shape[0] > 1:
        gen_starts_r[1:, 0] = n_nodes - np.flipud(gen_starts[:-1, 0])
        gen_starts_r[1:, 1] = n_scans - np.flipud(gen_starts[:-1, 1])

    n_gens = gen_starts.shape[0] - 1
    scan_starts_r = np.zeros_like(scan_starts)
    for i in range(n_gens):
        g_start = int(gen_starts_r[i, 1])
        g_stop = int(gen_starts_r[i + 1, 1])
        nodes_i = int(gen_starts_r[i + 1, 0]) - int(gen_starts_r[i, 0])
        # Window of forward scan_starts that corresponds to this reversed gen.
        fwd_window = scan_starts[(n_scans - g_stop) : (n_scans - g_start)]
        scan_i = nodes_i - np.ascontiguousarray(np.flipud(fwd_window))
        # First scan of the generation always starts at 0.
        scan_starts_r[g_start] = 0
        # Remaining scan starts are the reversed offsets, shifted by one.
        if g_stop - g_start > 1:
            scan_starts_r[g_start + 1 : g_stop] = scan_i[:-1]

    return nodes_r, scan_starts_r, gen_starts_r


def build_scan_ordering_for_kinforest(
    kinforest: KinForest, roots: Optional[Sequence[int]] = None
) -> Tuple[KinForestScanData, KinForestScanData]:
    """Produce the forward/backward generational scan data for a KinForest.

    Upstream equivalent: ``KinForestScanOrdering.calculate_from_kinforest``
    (returning the ``forward_scan_paths`` / ``backward_scan_paths`` pair).

    Parameters
    ----------
    kinforest : KinForest
        An already-built atom-level kinforest. Its ``parent`` array must
        satisfy the KFO invariant ``parent[i] <= i``.
    roots : sequence of int, optional
        KFO indices of the initial scan roots. Defaults to ``[0]`` (the
        conventional global root), matching upstream.

    Returns
    -------
    forward, backward : KinForestScanData
        Scan data for root->leaves (forward kinematics) and leaves->root
        (derivative summation) passes respectively. All fields are
        ``jnp.int32`` arrays and live in the default JAX device.
    """

    if roots is None:
        roots_arr = np.array([0], dtype=_INT)
    else:
        roots_arr = np.asarray(roots, dtype=_INT)

    # Pull parent into a plain numpy array (construction stays in numpy).
    parents = np.asarray(kinforest.parent, dtype=_INT)

    nodes_f, scans_f, gens_f = get_scans(parents, roots_arr)
    nodes_b, scans_b, gens_b = _reverse_scan_data(nodes_f, scans_f, gens_f)

    forward = KinForestScanData(
        nodes=jnp.asarray(nodes_f, dtype=jnp.int32),
        scans=jnp.asarray(scans_f, dtype=jnp.int32),
        gens=jnp.asarray(gens_f, dtype=jnp.int32),
    )
    backward = KinForestScanData(
        nodes=jnp.asarray(nodes_b, dtype=jnp.int32),
        scans=jnp.asarray(scans_b, dtype=jnp.int32),
        gens=jnp.asarray(gens_b, dtype=jnp.int32),
    )
    return forward, backward


# ---------------------------------------------------------------------------
# Degenerate (single-generation) scan data — the default for jtmol.
# ---------------------------------------------------------------------------


def construct_degenerate_scan_data(
    n_nodes: int,
) -> Tuple[KinForestScanData, KinForestScanData]:
    """Build one-generation / one-scan scan data suitable for a sequential
    ``jax.lax.fori_loop`` forward-kinematics implementation.

    Structure:

    * ``nodes`` = ``arange(n_nodes)`` (forward) / reversed (backward).
    * ``scans`` = ``[0]`` — a single scan spanning the whole generation.
    * ``gens`` = ``[[0, 0], [n_nodes, 1]]`` — one real generation plus the
      sentinel row carrying the totals.

    This is correct because the KinForest invariant ``parent[i] <= i``
    guarantees that a left-to-right linear scan visits every parent before
    its children, so no generational partitioning is required. The same
    sentinel structure as the full port is preserved so downstream code
    (kinematics kernels, serializers) can treat both modes identically.

    NOTE: This is the "degenerate" simplification called out in the module
    docstring. For a true parallel segmented-scan implementation, swap
    this call for :func:`build_scan_ordering_for_kinforest`.
    """

    n = int(n_nodes)
    if n < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n}")

    if n == 0:
        empty_nodes = jnp.zeros(0, dtype=jnp.int32)
        empty_scans = jnp.zeros(0, dtype=jnp.int32)
        empty_gens = jnp.zeros((1, 2), dtype=jnp.int32)
        empty = KinForestScanData(
            nodes=empty_nodes, scans=empty_scans, gens=empty_gens
        )
        return empty, empty

    nodes_f = np.arange(n, dtype=_INT)
    nodes_b = np.ascontiguousarray(np.flipud(nodes_f))
    scans = np.zeros(1, dtype=_INT)
    # One generation plus the sentinel row: shape (2, 2).
    gens = np.array([[0, 0], [n, 1]], dtype=_INT)

    forward = KinForestScanData(
        nodes=jnp.asarray(nodes_f, dtype=jnp.int32),
        scans=jnp.asarray(scans, dtype=jnp.int32),
        gens=jnp.asarray(gens, dtype=jnp.int32),
    )
    backward = KinForestScanData(
        nodes=jnp.asarray(nodes_b, dtype=jnp.int32),
        scans=jnp.asarray(scans, dtype=jnp.int32),
        gens=jnp.asarray(gens, dtype=jnp.int32),
    )
    return forward, backward


# ---------------------------------------------------------------------------
# Optional BFS-depth generational layout (degenerate-plus)
# ---------------------------------------------------------------------------


def _compute_bfs_depth(parents: np.ndarray) -> np.ndarray:
    """Compute the BFS-depth of every atom from its parent pointers.

    Since the KinForest invariant guarantees ``parent[i] <= i``, a single
    left-to-right linear sweep is enough: every parent is already
    processed by the time we visit its child.
    """
    parents = np.ascontiguousarray(parents, dtype=_INT)
    n = parents.shape[0]
    depth = np.zeros(n, dtype=_INT)
    for i in range(n):
        p = int(parents[i])
        if p == i:
            depth[i] = 0
        else:
            depth[i] = depth[p] + 1
    return depth


def construct_bfs_depth_scan_data(
    parents: np.ndarray, roots: Optional[Sequence[int]] = None
) -> Tuple[KinForestScanData, KinForestScanData]:
    """Partition atoms by BFS depth from the root(s); one scan per atom.

    Each atom sits at its BFS-depth generation; every atom in generation
    ``d`` is treated as its own independent "scan start" because its
    parent — the only data dependency it has — lives in an earlier
    generation ``d-1 < d``. This layout is therefore suitable for
    generation-parallel forward / inverse kinematics: within a generation
    all atoms can be updated independently via ``jax.vmap`` (or an einsum
    over the gathered parent HTs), and successive generations are walked
    via ``jax.lax.scan``.

    Returns
    -------
    forward : KinForestScanData
        ``nodes`` lists atoms in generation-major / KFO-minor order.
        ``scans`` holds the in-generation offsets of each scan start
        (``[0, 1, 2, ..., gen_size-1]`` for each generation, since every
        atom is its own scan). ``gens`` is the standard
        ``[n_gens + 1, 2]`` table whose two columns index ``nodes`` and
        ``scans`` respectively, with the sentinel final row carrying the
        totals.
    backward : KinForestScanData
        Same partitioning reversed (leaves-to-root), suitable for
        derivative summation / inverse kinematics scans.

    Notes
    -----
    This differs from the full segmented-scan layout produced by
    :func:`get_scans` in that each generation-wise atom is its own
    scan (length 1) rather than being chained into longer primary
    paths. The tradeoff is many shallow generations vs. few deep ones:
    this layout maps cleanly onto JAX's ``vmap`` + ``scan`` combo, while
    upstream's deep primary paths are better suited to a CUDA segmented-
    scan kernel.
    """

    parents = np.ascontiguousarray(parents, dtype=_INT)
    n = parents.shape[0]
    if n == 0:
        empty, _ = construct_degenerate_scan_data(0)
        return empty, empty

    depth = _compute_bfs_depth(parents)

    # Group atoms by depth, preserving KFO index order within each group.
    # ``argsort(kind='stable')`` ensures KFO order is kept within a depth.
    order = np.argsort(depth, kind="stable").astype(_INT)
    sorted_depth = depth[order]

    # Compute generation boundaries via searchsorted over the sorted depth.
    max_depth = int(sorted_depth[-1])
    n_gens = max_depth + 1

    # For each generation g, the "nodes" range is the contiguous slice of
    # `order` where sorted_depth == g. `gens[g, 0]` points into `nodes`;
    # `gens[g, 1]` points into `scans`. Each atom is its own scan, so the
    # per-gen scan count equals the per-gen node count, and `scans`
    # entries are ``0, 1, 2, ...`` within each generation.
    gen_starts = np.zeros((n_gens + 1, 2), dtype=_INT)
    for g in range(n_gens + 1):
        idx = int(np.searchsorted(sorted_depth, g, side="left"))
        gen_starts[g, 0] = idx
        gen_starts[g, 1] = idx  # scans run 1:1 with nodes here

    # Sentinel row: total lengths.
    gen_starts[n_gens, 0] = n
    gen_starts[n_gens, 1] = n

    # Build scan starts: within each generation, scans go 0, 1, 2, ...
    scan_starts = np.zeros(n, dtype=_INT)
    for g in range(n_gens):
        lo = int(gen_starts[g, 0])
        hi = int(gen_starts[g + 1, 0])
        scan_starts[lo:hi] = np.arange(hi - lo, dtype=_INT)

    # Reverse direction: walk from leaves to root.
    nodes_r, scan_starts_r, gen_starts_r = _reverse_scan_data(
        order, scan_starts, gen_starts
    )

    forward = KinForestScanData(
        nodes=jnp.asarray(order, dtype=jnp.int32),
        scans=jnp.asarray(scan_starts, dtype=jnp.int32),
        gens=jnp.asarray(gen_starts, dtype=jnp.int32),
    )
    backward = KinForestScanData(
        nodes=jnp.asarray(nodes_r, dtype=jnp.int32),
        scans=jnp.asarray(scan_starts_r, dtype=jnp.int32),
        gens=jnp.asarray(gen_starts_r, dtype=jnp.int32),
    )
    return forward, backward


# ---------------------------------------------------------------------------
# Padded (n_gens, max_gen_size) layout for jax.lax.scan + vmap
# ---------------------------------------------------------------------------


def build_padded_gen_layout(
    nodes: np.ndarray, gens: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a flat ``(nodes, gens)`` scan layout into padded 2D arrays.

    Parameters
    ----------
    nodes : int32[N]
        Atoms listed in generation-major order.
    gens : int32[n_gens + 1, 2]
        Per-generation boundaries into ``nodes``.

    Returns
    -------
    gen_atoms : int32[n_gens, max_gen_size]
        Per-generation atom index array. Padded slots contain ``0`` (a
        safe index that always points to the forest root).
    gen_valid : bool[n_gens, max_gen_size]
        True where ``gen_atoms[g, i]`` is a real atom, False for padding.

    Notes
    -----
    This layout is consumed by forward / inverse kinematics kernels that
    use ``jax.lax.scan`` over generations: each scan step pulls the
    atoms in one generation, gathers their parent transforms from the
    HT buffer, composes, and scatters back — with padding masked out so
    the forest root's HT is preserved.
    """
    nodes = np.ascontiguousarray(nodes, dtype=_INT)
    gens = np.ascontiguousarray(gens, dtype=_INT)
    n_gens = int(gens.shape[0]) - 1
    if n_gens <= 0:
        return (
            np.zeros((0, 0), dtype=_INT),
            np.zeros((0, 0), dtype=bool),
        )

    # Determine max generation size so we can pre-allocate the padded 2D
    # layout.
    gen_sizes = np.diff(gens[:, 0]).astype(_INT)
    max_gen = int(gen_sizes.max()) if n_gens > 0 else 0

    gen_atoms = np.zeros((n_gens, max_gen), dtype=_INT)
    gen_valid = np.zeros((n_gens, max_gen), dtype=bool)
    for g in range(n_gens):
        lo = int(gens[g, 0])
        hi = int(gens[g + 1, 0])
        size = hi - lo
        if size > 0:
            gen_atoms[g, :size] = nodes[lo:hi]
            gen_valid[g, :size] = True
    return gen_atoms, gen_valid


# ---------------------------------------------------------------------------
# Public entry point: build a KinematicModuleData from a FoldForest+PoseStack
# ---------------------------------------------------------------------------


def construct_kin_module_data(
    fold_forest,
    pose_stack,
    move_map=None,
    *,
    kinforest: Optional[KinForest] = None,
    scan_mode: str = "bfs_depth",
) -> KinematicModuleData:
    """Build a :class:`KinematicModuleData` for a ``(fold_forest, pose_stack)``
    pair.

    Upstream reference: ``construct_kin_module_data_for_pose`` in
    ``tmol/kinematics/scan_ordering.py``.

    Parameters
    ----------
    fold_forest : FoldForest
        Pose-level fold-forest edges (polymer, jump, root-jump).
    pose_stack : PoseStack
        The pose stack whose atoms will be woven into a KinForest.
    move_map : MoveMap, optional
        A move map restricting which DOFs are free to move. Accepted for
        API parity with upstream; currently unused in the scan-ordering
        construction — move-map application happens later, in the DOF
        layer.
    kinforest : KinForest, optional
        Escape hatch: if the atom-level KinForest has already been built
        (for example, by a dedicated kinforest-builder module), pass it
        directly and this function only computes the scan data and
        auxiliary fold-forest bookkeeping.
    scan_mode : {"bfs_depth", "full", "degenerate"}, default "bfs_depth"
        Which scan-ordering strategy to use:

        * ``"bfs_depth"`` — atoms grouped by BFS depth from the root.
          Each atom in a generation is its own scan start. This layout
          gives per-generation parallelism in JAX via
          ``lax.scan + vmap``. Recommended default.
        * ``"full"`` — faithful port of upstream's DFS-with-descendant-
          priority algorithm. Produces longer primary paths and fewer
          generations; suitable for a real segmented-scan GPU kernel.
        * ``"degenerate"`` — one generation containing every atom, one
          scan starting at 0. Kept for backward compatibility with the
          sequential ``fori_loop`` path.

    Returns
    -------
    KinematicModuleData
        Bundles the atom-level KinForest together with forward/backward
        scan data and the fold-forest auxiliary tensors
        (``block_in_and_first_out``, ``keep_atom_fixed``,
        ``pose_stack_atom_for_jump``, ``pose_stack_atom_for_root_jump``).

    Notes
    -----
    The KinForest-from-FoldForest assembly in upstream lives partly in
    the compiled ops (``calculate_ff_edge_delays``,
    ``get_kfo_indices_for_atoms``, ``get_kfo_atom_parents``,
    ``get_id_and_frame_xyz``, ``get_kinforest_scans_from_stencils2``)
    and partly in the heavyweight Python annotator
    ``_annotate_block_type_with_gen_scan_path_segs``. This JAX port
    instead delegates KinForest construction to
    :meth:`FoldForest.build_kinforest`, a simpler per-pose walker that
    emits each residue's atoms in icoor-ancestor order and connects
    successive residues via polymer / jump edges.
    """

    if scan_mode not in ("degenerate", "bfs_depth", "full"):
        raise ValueError(
            f"Unknown scan_mode {scan_mode!r}; "
            "expected one of 'bfs_depth', 'full', 'degenerate'."
        )

    # ------------------------------------------------------------------
    # Resolve the KinForest.
    # ------------------------------------------------------------------
    if kinforest is None:
        kinforest = _try_resolve_kinforest(fold_forest, pose_stack)

    if kinforest is None and fold_forest is not None and pose_stack is not None:
        # FoldForest carries a `build_kinforest` method that produces the
        # per-pose atom-level KinForest by walking polymer / jump edges.
        build_fn = getattr(fold_forest, "build_kinforest", None)
        if callable(build_fn):
            kinforest = build_fn(pose_stack)

    if kinforest is None:
        raise ValueError(
            "construct_kin_module_data: could not resolve a KinForest. "
            "Pass one explicitly via the `kinforest` kwarg, or ensure the "
            "supplied `fold_forest` has a callable `build_kinforest` method."
        )

    # ------------------------------------------------------------------
    # Scan data.
    # ------------------------------------------------------------------
    n_nodes = int(np.asarray(kinforest.parent).shape[0])

    if scan_mode == "degenerate":
        forward, backward = construct_degenerate_scan_data(n_nodes)
    elif scan_mode == "bfs_depth":
        parents = np.asarray(kinforest.parent, dtype=_INT)
        forward, backward = construct_bfs_depth_scan_data(parents)
    else:  # "full"
        forward, backward = build_scan_ordering_for_kinforest(kinforest)

    # ------------------------------------------------------------------
    # Fold-forest auxiliary tensors.
    # ------------------------------------------------------------------
    (
        block_in_and_first_out,
        keep_atom_fixed,
        pose_stack_atom_for_jump,
        pose_stack_atom_for_root_jump,
    ) = _build_fold_forest_aux_tensors(fold_forest, pose_stack, move_map)

    return KinematicModuleData(
        forest=kinforest,
        scan_data_fw=forward,
        scan_data_bw=backward,
        block_in_and_first_out=block_in_and_first_out,
        keep_atom_fixed=keep_atom_fixed,
        pose_stack_atom_for_jump=pose_stack_atom_for_jump,
        pose_stack_atom_for_root_jump=pose_stack_atom_for_root_jump,
    )


# ---------------------------------------------------------------------------
# Helpers for the public entry point
# ---------------------------------------------------------------------------


def _try_resolve_kinforest(fold_forest, pose_stack) -> Optional[KinForest]:
    """Best-effort lookup of a pre-built KinForest on either object.

    Other modules in jtmol (the kinforest builder, caching layers,
    tests) may attach a ``kinforest`` attribute to a ``PoseStack`` or
    ``FoldForest``. This helper sniffs the common locations so
    ``construct_kin_module_data`` can be called with just the two
    high-level objects once a builder is available.
    """

    for obj in (pose_stack, fold_forest):
        if obj is None:
            continue
        for attr in ("kinforest", "_kinforest", "kin_forest"):
            cand = getattr(obj, attr, None)
            if isinstance(cand, KinForest):
                return cand
    return None


def _build_fold_forest_aux_tensors(
    fold_forest, pose_stack, move_map
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Produce the four auxiliary tensors stored alongside scan data.

    These tensors describe block-level bookkeeping for the fold forest
    that downstream kinematics code consumes (move-map gating, jump atom
    lookup, etc.). Until the full block-type-annotation pipeline is
    ported, they are produced as zero-filled placeholders with the
    correct shapes and dtypes, which is enough for the scan-ordering
    code path to work.

    NOTE: Zeros are safe sentinels here because the forward / inverse
    kinematics operations consume only ``kinforest`` and ``scan_data_fw`` /
    ``scan_data_bw``. Move-map gating and jump-DOF bookkeeping will
    consult these tensors once the matching upstream pipeline lands.
    """

    # Try to read shapes from the pose_stack (JAX port) without depending
    # on any particular upstream attribute layout.
    n_poses = _safe_get(pose_stack, "n_poses", default=1)
    max_n_blocks = _safe_get(pose_stack, "max_n_blocks", default=0)
    max_n_atoms_per_pose = _safe_get(pose_stack, "max_n_atoms_per_pose", default=0)
    max_n_jumps = _safe_get(fold_forest, "max_n_edges", default=0)

    block_in_and_first_out = jnp.zeros(
        (int(n_poses), int(max_n_blocks), 2), dtype=jnp.int32
    )
    keep_atom_fixed = jnp.zeros(
        (int(n_poses), int(max_n_atoms_per_pose)), dtype=jnp.bool_
    )
    pose_stack_atom_for_jump = jnp.full(
        (int(n_poses), int(max(1, max_n_jumps)), 2), -1, dtype=jnp.int32
    )
    pose_stack_atom_for_root_jump = jnp.full(
        (int(n_poses), 2), -1, dtype=jnp.int32
    )

    return (
        block_in_and_first_out,
        keep_atom_fixed,
        pose_stack_atom_for_jump,
        pose_stack_atom_for_root_jump,
    )


def _safe_get(obj, attr, default):
    """Return ``getattr(obj, attr)`` if it exists and is readable."""
    if obj is None:
        return default
    try:
        val = getattr(obj, attr)
    except Exception:
        return default
    return val if val is not None else default


# ---------------------------------------------------------------------------
# Roadmap (what is still missing relative to upstream)
# ---------------------------------------------------------------------------
#
# DONE: Per-generation BFS-depth partitioning (default scan mode) and
# integration with a ``jax.lax.scan`` forward-kinematics kernel.
#
# DONE: KinForest assembly from a ``(fold_forest, pose_stack)`` pair via
# :meth:`FoldForest.build_kinforest`.
#
# TODO: Port the optimized block-type-aware KinForest builder. Upstream
# splits this between the compiled ops under ``tmol.kinematics.compiled``
# (pure C++: ``calculate_ff_edge_delays``, ``get_kfo_indices_for_atoms``,
# ``get_kfo_atom_parents``, ``get_children``, ``get_id_and_frame_xyz``,
# ``get_kinforest_scans_from_stencils2``, ``get_jump_atom_indices``,
# ``get_block_parent_connectivity_from_toposort``) and the block-type
# annotator ``_annotate_block_type_with_gen_scan_path_segs``. The JAX
# equivalent would weave the per-block-type primary-path segments into
# a KinForest with many fewer generations than the simple BFS
# ``build_kinforest`` produces, letting the generational ``lax.scan``
# contract further. A good landing spot is a new
# ``jtmol.kinematics.kinforest_builder`` module.
#
# TODO: Port ``_mark_kin_uaids_controlling_torsions`` and friends to
# populate per-block-type torsion bookkeeping that the move-map layer
# needs.
#
# TODO: Wire in the move-map gating once
# ``jtmol.kinematics.move_map.MoveMap`` exposes its per-DOF boolean
# mask -- that signal flows into ``keep_atom_fixed`` here.


__all__ = [
    "get_scans",
    "build_scan_ordering_for_kinforest",
    "construct_degenerate_scan_data",
    "construct_bfs_depth_scan_data",
    "construct_kin_module_data",
    "build_padded_gen_layout",
]
