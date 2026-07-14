"""Fold-forest data structure and construction, ported to pure JAX.

Upstream reference: ``tmol/kinematics/fold_forest.py`` and
``tmol/kinematics/check_fold_forest.py``. Both files are merged here into a
single module, since the JAX port drops the numba-JIT-ed validation helpers
in favour of pure numpy loops (fold-forest construction is a one-shot setup
step, not a per-minimization-step hot path).

Overview
--------
A :class:`FoldForest` describes, for every pose in a :class:`PoseStack`, a
rooted spanning tree over the pose's blocks. The tree encodes which blocks
are connected by polymeric bonds (modeled as variable-length "polymer
edges"), which blocks are attached to a parent block by a rigid-body
"jump", and which block of each tree connects back to the global reference
frame through a "root jump".

The forest is pose-stack-batched and padded to a common ``max_n_edges``
along axis 1. All array fields are ``jax.Array``; numpy is used internally
during construction.

Simplifications vs upstream
--------------------------
* Only ``polymer``, ``jump``, and ``root_jump`` edges are emitted by the
  constructors. ``chemical`` edges are defined for API parity but are never
  produced by :meth:`FoldForest.from_pose_stack`.
* Validation is performed by a plain-numpy :meth:`FoldForest.validate`
  method that raises ``ValueError`` on broken forests. It does not attempt
  to reproduce the per-error string accumulation of upstream verbatim — a
  single multi-line message is raised describing every issue found.
* :meth:`FoldForest.build_kinforest` produces a *simplified* atom-level
  :class:`KinForest`. It walks each polymer edge in block order, emits
  each residue's atoms in icoor-tree order (parents come from
  ``RefinedResidueType.icoors_ancestors``), connects successive residues
  via a peptide bond ``N <- C_prev`` (first atom of block ``i+1`` becomes
  a child of the last backbone atom of block ``i`` when both define
  ``up``/``down`` connections), and plants each chain at the global frame
  with a single root-jump node. Jumps between blocks are emitted as a
  single jump node to the first atom of the downstream block. This is a
  *coarse* scaffold suitable for sanity-checking downstream code; a
  faithful port of the scan-ordering builder lives in
  ``jtmol.kinematics.scan_ordering`` (not ported here).
"""

from __future__ import annotations

import enum
from typing import Any, List, Optional, Tuple

import attrs
import jax.numpy as jnp
import numpy as np

from jtmol.types import Array, as_jnp, as_np, pytree_dataclass


# ---------------------------------------------------------------------------
# Edge-type enumeration
# ---------------------------------------------------------------------------


class EdgeType(enum.IntEnum):
    """Kinds of edges that can appear in a :class:`FoldForest`.

    Values are fixed by the pose-stack protocol:

    * ``polymer`` (0): a contiguous run of polymer-bonded blocks.
    * ``jump`` (1): a rigid-body jump between two blocks within a pose.
    * ``chemical`` (2): a chemical-bond edge (e.g. disulfide). Reserved;
      not emitted by the default constructors in this port.
    * ``root_jump`` (3): the edge rooting a tree at the global frame.

    Note: this differs from the upstream ``tmol.kinematics.fold_forest``
    enum, which defines ``root_jump`` as ``2`` and omits ``chemical``.
    The extension here matches the planned successor schema.
    """

    polymer = 0
    jump = 1
    chemical = 2
    root_jump = 3


# ---------------------------------------------------------------------------
# Helpers — pure numpy
# ---------------------------------------------------------------------------


def _pbt_up_down_conn_arrays(pbt: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-block-type up/down connection indices from a
    ``PackedBlockTypes``.

    Upstream stores ``up_conn_inds`` / ``down_conn_inds`` directly on
    ``PackedBlockTypes``; the JAX port does not, so we derive them from
    the ``active_block_types`` tuple. Missing attributes default to ``-1``.
    """
    up_list: List[int] = []
    down_list: List[int] = []
    for rt in pbt.active_block_types:
        up_list.append(int(getattr(rt, "up_connection_ind", -1)))
        down_list.append(int(getattr(rt, "down_connection_ind", -1)))
    up = np.asarray(up_list, dtype=np.int64)
    down = np.asarray(down_list, dtype=np.int64)
    return up, down


def _pose_chain_id(
    bti_p: np.ndarray,
    irc_p: np.ndarray,
    up_c: np.ndarray,
    down_c: np.ndarray,
) -> np.ndarray:
    """Derive a per-residue ``chain_id`` array from polymer connectivity.

    The JAX port's :class:`PoseStack` does not carry a ``chain_id`` field
    (unlike upstream), so we synthesize one: every polymer bond joins
    residues into the same chain, and chain breaks start a new chain.

    Two residues share a chain id iff they are connected by a chain of
    valid polymer bonds (up/down with matching connection slots).
    """
    n_res = len(bti_p)
    chain_id = np.full(n_res, -1, dtype=np.int64)
    real_mask = bti_p >= 0

    # Build a union-find over polymer-successor edges.
    parent = np.arange(n_res, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    rows = np.arange(n_res)
    bt_safe = np.where(real_mask, bti_p, 0)
    uc_r = up_c[bt_safe]
    uc_safe = np.maximum(uc_r, 0)
    succ_raw = irc_p[rows, uc_safe, 0]
    succ_cs = irc_p[rows, uc_safe, 1]
    s_safe = np.maximum(succ_raw, 0)
    dc_s = down_c[np.maximum(bti_p[s_safe], 0)]
    valid = (
        real_mask
        & (uc_r >= 0)
        & (succ_raw >= 0)
        & real_mask[s_safe]
        & (succ_cs == dc_s)
    )
    for r in np.where(valid)[0]:
        union(int(r), int(succ_raw[r]))

    # Assign chain ids by first-seen root, in residue-index order.
    next_cid = 0
    root_to_cid: dict = {}
    for r in range(n_res):
        if not real_mask[r]:
            continue
        root = find(int(r))
        if root not in root_to_cid:
            root_to_cid[root] = next_cid
            next_cid += 1
        chain_id[r] = root_to_cid[root]
    return chain_id


def _build_pose_fold_forest(
    bti_p: np.ndarray,
    irc_p: np.ndarray,
    up_c: np.ndarray,
    down_c: np.ndarray,
    chain_id_p: np.ndarray,
) -> List[List[int]]:
    """Build fold-forest edges for a single pose from polymer connectivity.

    Direct numpy port of the upstream ``_build_pose_fold_forest`` helper,
    with the edge-type codes from this module's :class:`EdgeType`. Returns
    a list of ``[type, start, end, jump_idx]`` integer lists (no array
    padding — padding happens in the caller).

    The resulting graph is a disjoint union of simple paths (linear
    chains) and simple cycles (C→N cyclisation). Non-polymer connections
    (disulfides, etc.) are ignored; cyclic polymers are broken at the bond
    entering the lowest-index residue to keep the forest a valid tree.
    """
    n_res = len(bti_p)
    rows = np.arange(n_res)

    real_mask = bti_p >= 0
    real_res = np.where(real_mask)[0]

    if len(real_res) == 0:
        return []

    bt_safe = np.where(real_mask, bti_p, 0)

    uc_r = up_c[bt_safe]
    uc_safe = np.maximum(uc_r, 0)

    succ_raw = irc_p[rows, uc_safe, 0]
    succ_cs = irc_p[rows, uc_safe, 1]

    s_safe = np.maximum(succ_raw, 0)
    dc_s = down_c[np.maximum(bti_p[s_safe], 0)]

    valid = (
        real_mask
        & (uc_r >= 0)
        & (succ_raw >= 0)
        & real_mask[s_safe]
        & (succ_cs == dc_s)
    )

    poly_succ_arr = np.full(n_res, -1, dtype=np.int64)
    poly_pred_arr = np.full(n_res, -1, dtype=np.int64)

    valid_r = np.where(valid)[0]
    valid_s = succ_raw[valid]
    poly_succ_arr[valid_r] = valid_s
    poly_pred_arr[valid_s] = valid_r

    # Walk linear chains from each N-terminus (real residue, no pred).
    chains: List[Tuple[int, int]] = []
    visited = np.zeros(n_res, dtype=bool)
    for r in np.where(real_mask & (poly_pred_arr < 0))[0]:
        r = int(r)
        cur = r
        while poly_succ_arr[cur] >= 0:
            visited[cur] = True
            cur = int(poly_succ_arr[cur])
        visited[cur] = True
        chains.append((r, cur))

    # Break cycles (C→N cyclisation) at the lowest-index residue.
    cyclized_res = np.where(real_mask & ~visited)[0]
    for r in cyclized_res:
        r = int(r)
        if visited[r]:
            continue
        cycle: List[int] = []
        cur = r
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = int(poly_succ_arr[cur])
        n_term = min(cycle)
        c_term = int(poly_pred_arr[n_term])
        chains.append((n_term, c_term))

    # Emit edges: one root-jump or jump per chain, plus a polymer edge if
    # the chain spans >1 residues. Chains that belong to the same
    # biological chain as the preceding residue become internal jumps.
    result: List[List[int]] = []
    jump_idx = 0
    for start, end in sorted(chains, key=lambda c: c[0]):
        prev = start - 1
        if (
            prev >= 0
            and bti_p[prev] >= 0
            and chain_id_p[prev] == chain_id_p[start]
        ):
            result.append([int(EdgeType.jump), int(prev), int(start), jump_idx])
        else:
            result.append(
                [int(EdgeType.root_jump), -1, int(start), jump_idx]
            )
        jump_idx += 1
        if start != end:
            result.append(
                [int(EdgeType.polymer), int(start), int(end), -1]
            )
    return result


# ---------------------------------------------------------------------------
# FoldForest
# ---------------------------------------------------------------------------


@pytree_dataclass
@attrs.define(frozen=True, slots=False)
class FoldForest:
    """Batched fold-forest description for a :class:`PoseStack`.

    Each tensor has a leading axis over the number of poses.

    The core definition is the *edge*, a 4-tuple
    ``(type, start_block, end_block, meta)``:

    * ``type`` is an :class:`EdgeType` value (``polymer``, ``jump``,
      ``chemical``, or ``root_jump``).
    * ``start_block`` / ``end_block`` are block indices within the pose.
      For ``root_jump`` edges ``start_block`` is ``-1`` (the root lives
      at the global frame, not on a block).
    * ``meta`` carries the *jump index* for ``jump`` edges (unique,
      dense, starting from ``0``), and ``-1`` for ``polymer`` /
      ``root_jump`` edges.

    Attributes
    ----------
    edges: ``int32[n_poses, max_n_edges, 4]``
        Padded edge table. Unused rows are ``-1``-filled.
    n_edges: ``int32[n_poses]``
        True number of edges per pose (length of the used prefix of
        ``edges[i]``).
    roots: ``int32[n_poses]``
        Root block index per pose. When a pose has multiple root-jumps
        the *first* root-jump's downstream block is reported; callers
        that need the full list should filter ``edges`` directly.

    Notes
    -----
    ``FoldForest`` is a pytree — it flows through ``jit`` / ``vmap`` /
    ``grad`` / ``scan`` transparently. Construction uses numpy; the final
    arrays are ``jax.Array`` (int32).
    """

    edges: Array  # int32[n_poses, max_n_edges, 4]
    n_edges: Array  # int32[n_poses]
    roots: Array  # int32[n_poses]

    # ---------------- derived properties ----------------

    @property
    def n_poses(self) -> int:
        return int(self.edges.shape[0])

    @property
    def max_n_edges(self) -> int:
        return int(self.edges.shape[1])

    # ---------------- constructors ----------------

    @classmethod
    def polymeric(cls, n_blocks_per_pose: Any) -> "FoldForest":
        """Build the trivial polymer fold forest for a batch of chain-sized
        poses.

        Each pose becomes a single chain:

        * one ``root_jump`` edge landing on block 0 (jump index 0)
        * one ``polymer`` edge spanning block 0 .. (n_blocks - 1)
          (omitted when ``n_blocks == 1``)

        Poses with zero real blocks are padded away (empty edge list).

        Parameters
        ----------
        n_blocks_per_pose:
            Shape ``[n_poses]`` int-like, or a scalar (single pose).
        """
        nbp = as_np(n_blocks_per_pose, dtype=np.int64).reshape(-1)
        n_poses = int(nbp.shape[0])

        # Each pose needs up to 2 edges (root-jump + polymer) when
        # len > 1, else 1 edge. Reserve max_n_edges=2 always for a
        # uniform padded shape.
        max_n_edges = 2 if n_poses > 0 else 1
        edges_np = np.full((n_poses, max_n_edges, 4), -1, dtype=np.int64)
        n_edges_np = np.zeros(n_poses, dtype=np.int64)
        roots_np = np.full(n_poses, -1, dtype=np.int64)

        for p in range(n_poses):
            n = int(nbp[p])
            if n <= 0:
                continue
            # root-jump at block 0, jump index 0
            edges_np[p, 0] = [int(EdgeType.root_jump), -1, 0, 0]
            roots_np[p] = 0
            count = 1
            if n > 1:
                edges_np[p, 1] = [int(EdgeType.polymer), 0, n - 1, -1]
                count = 2
            n_edges_np[p] = count

        return cls(
            edges=as_jnp(edges_np, dtype=jnp.int32),
            n_edges=as_jnp(n_edges_np, dtype=jnp.int32),
            roots=as_jnp(roots_np, dtype=jnp.int32),
        )

    @classmethod
    def from_pose_stack(cls, ps: Any) -> "FoldForest":
        """Build the default polymer + jump fold forest from a
        :class:`PoseStack`.

        Uses ``inter_residue_connections`` to detect chain breaks, emits
        polymer edges within each chain, and a jump (or root-jump) per
        chain. Non-polymer inter-residue connections (e.g. disulfides)
        are ignored. Cyclic polymers are broken at the lowest-index bond.

        This is the JAX analogue of upstream
        ``FoldForest.reasonable_fold_forest``.
        """
        irc = as_np(ps.inter_residue_connections, dtype=np.int64)
        bti = as_np(ps.block_type_ind, dtype=np.int64)
        pbt = ps.packed_block_types
        up_c, down_c = _pbt_up_down_conn_arrays(pbt)

        # PoseStack in the JAX port has no ``chain_id`` field; synthesize
        # one per-pose from the polymer connectivity.
        chain_id = np.stack(
            [_pose_chain_id(bti[p], irc[p], up_c, down_c) for p in range(bti.shape[0])],
            axis=0,
        )

        all_pose_edges = [
            _build_pose_fold_forest(bti[p], irc[p], up_c, down_c, chain_id[p])
            for p in range(bti.shape[0])
        ]

        max_n_edges = max((len(e) for e in all_pose_edges), default=1)
        n_poses = bti.shape[0]

        edges_np = np.full((n_poses, max_n_edges, 4), -1, dtype=np.int64)
        n_edges_np = np.zeros(n_poses, dtype=np.int64)
        roots_np = np.full(n_poses, -1, dtype=np.int64)

        for p, edges_p in enumerate(all_pose_edges):
            n = len(edges_p)
            if n > 0:
                edges_np[p, :n] = edges_p
            n_edges_np[p] = n
            # First root-jump's downstream block is the canonical root.
            for row in edges_p:
                if row[0] == int(EdgeType.root_jump):
                    roots_np[p] = row[2]
                    break

        return cls(
            edges=as_jnp(edges_np, dtype=jnp.int32),
            n_edges=as_jnp(n_edges_np, dtype=jnp.int32),
            roots=as_jnp(roots_np, dtype=jnp.int32),
        )

    @classmethod
    def from_edges(cls, edges: Any, roots: Optional[Any] = None) -> "FoldForest":
        """Construct from a pre-built ``[n_poses, max_n_edges, 4]`` edge
        table. ``n_edges`` is inferred from non-sentinel rows; ``roots``
        is inferred from the first ``root_jump`` edge of each pose when
        not supplied.
        """
        edges_np = as_np(edges, dtype=np.int64)
        n_poses, max_n_edges, _ = edges_np.shape
        n_edges_np = np.sum(edges_np[:, :, 0] != -1, axis=1).astype(np.int64)
        if roots is None:
            roots_np = np.full(n_poses, -1, dtype=np.int64)
            for p in range(n_poses):
                for j in range(max_n_edges):
                    if edges_np[p, j, 0] == int(EdgeType.root_jump):
                        roots_np[p] = edges_np[p, j, 2]
                        break
        else:
            roots_np = as_np(roots, dtype=np.int64)
        return cls(
            edges=as_jnp(edges_np, dtype=jnp.int32),
            n_edges=as_jnp(n_edges_np, dtype=jnp.int32),
            roots=as_jnp(roots_np, dtype=jnp.int32),
        )

    # ---------------- validation ----------------

    def validate(self, n_blocks_per_pose: Optional[Any] = None) -> None:
        """Check fold-forest integrity.

        Raises
        ------
        ValueError
            If any pose's fold forest contains out-of-range blocks,
            cycles, unreachable blocks, or malformed jump indices.

        Parameters
        ----------
        n_blocks_per_pose:
            Optional ``[n_poses]`` array of per-pose block counts. If
            ``None``, inferred from the largest block index appearing in
            any edge (which is only safe when there are no padding blocks
            beyond the last referenced block).
        """
        edges = as_np(self.edges, dtype=np.int64)
        n_poses, max_n_edges, _ = edges.shape

        if n_blocks_per_pose is None:
            # Fallback: use per-pose max referenced block + 1.
            nb = np.zeros(n_poses, dtype=np.int64)
            for p in range(n_poses):
                mx = -1
                for j in range(max_n_edges):
                    if edges[p, j, 0] == -1:
                        continue
                    mx = max(mx, int(edges[p, j, 1]), int(edges[p, j, 2]))
                nb[p] = mx + 1
            n_blocks = nb
        else:
            n_blocks = as_np(n_blocks_per_pose, dtype=np.int64).reshape(-1)

        errors: List[str] = []

        # -- Step 1: mark implicit polymer bonds, check out-of-range. --
        max_n_blocks = int(n_blocks.max()) if n_blocks.size else 0
        # connections[p, a, b] counts polymer/jump edges landing at (a,b).
        connections = np.zeros(
            (n_poses, max(max_n_blocks, 1), max(max_n_blocks, 1)),
            dtype=np.int64,
        )

        for p in range(n_poses):
            for j in range(max_n_edges):
                row = edges[p, j]
                t = int(row[0])
                if t == -1:
                    continue
                s, e = int(row[1]), int(row[2])
                if t == int(EdgeType.root_jump):
                    # root-jump: start should be -1, end must be in range
                    if e >= n_blocks[p] or e < 0:
                        errors.append(
                            f"FOLD FOREST ERROR: Bad root-jump edge {j} "
                            f"in pose {p} gives end index {e} out of range; "
                            f"(n_blocks[{p}] = {n_blocks[p]})"
                        )
                    continue
                if s >= n_blocks[p] or s < 0:
                    errors.append(
                        f"FOLD FOREST ERROR: Bad edge {j} in pose {p} "
                        f"gives start index {s} out of range; "
                        f"(n_blocks[{p}] = {n_blocks[p]})"
                    )
                if e >= n_blocks[p] or e < 0:
                    errors.append(
                        f"FOLD FOREST ERROR: Bad edge {j} in pose {p} "
                        f"gives end index {e} out of range; "
                        f"(n_blocks[{p}] = {n_blocks[p]})"
                    )
                if errors:
                    continue
                if t == int(EdgeType.polymer):
                    increment = 1 if s < e else -1
                    for k in range(s, e, increment):
                        connections[p, k, k + increment] += 1

        if errors:
            raise ValueError("\n".join(errors))

        # -- Step 2: absorb jump/chemical edges, collect roots. --
        roots = np.full((n_poses, max(max_n_edges, 1)), -1, dtype=np.int64)
        n_roots = np.zeros(n_poses, dtype=np.int64)
        for p in range(n_poses):
            for j in range(max_n_edges):
                row = edges[p, j]
                t = int(row[0])
                if t == int(EdgeType.jump) or t == int(EdgeType.chemical):
                    r1, r2 = int(row[1]), int(row[2])
                    if 0 <= r1 < max_n_blocks and 0 <= r2 < max_n_blocks:
                        connections[p, r1, r2] += 1
                elif t == int(EdgeType.root_jump):
                    r2 = int(row[2])
                    roots[p, n_roots[p]] = r2
                    n_roots[p] += 1

        # -- Step 3: BFS each forest. --
        for p in range(n_poses):
            nb_p = int(n_blocks[p])
            color = np.zeros(max(max_n_blocks, 1), dtype=np.int64)
            queue: List[int] = []
            for j in range(int(n_roots[p])):
                r = int(roots[p, j])
                if r < 0:
                    break
                if color[r] == 1:
                    errors.append(
                        f"FOLD FOREST ERROR: Cycle detected in pose {p} "
                        f"at block {r} (duplicate root)"
                    )
                    break
                queue.append(r)
                color[r] = 1
            found_cycle = False
            while queue and not found_cycle:
                focused = queue.pop(0)
                for j in range(max_n_blocks):
                    c = int(connections[p, focused, j])
                    if c == 0:
                        continue
                    if c > 1 or color[j] == 1:
                        errors.append(
                            f"FOLD FOREST ERROR: Cycle detected in pose "
                            f"{p} at block {j}"
                        )
                        found_cycle = True
                        break
                    color[j] = 1
                    queue.append(j)
            for j in range(nb_p):
                if color[j] != 1:
                    errors.append(
                        f"FOLD FOREST ERROR: Block {j} unreachable in pose {p}"
                    )

        # -- Step 4: jumps numbered contiguously from 0, no repeats. --
        for p in range(n_poses):
            seen: dict = {}
            n_jumps = 0
            for j in range(max_n_edges):
                row = edges[p, j]
                if int(row[0]) != int(EdgeType.jump):
                    continue
                n_jumps += 1
                idx = int(row[3])
                if idx < 0 or idx >= max_n_edges:
                    errors.append(
                        f"FOLD FOREST ERROR: Jump [p={row[0]}, s={row[1]}, "
                        f"e={row[2]}, ind={row[3]}] in pose {p} has "
                        f"negative jump index"
                        if idx < 0
                        else f"FOLD FOREST ERROR: Jump edge {j} in pose {p} "
                             f"has out-of-range jump index {idx}"
                    )
                    continue
                if idx in seen:
                    errors.append(
                        f"FOLD FOREST ERROR: Jump [p={row[0]}, s={row[1]}, "
                        f"e={row[2]}, ind={row[3]}] in pose {p} has "
                        f"repeated jump index with edge {seen[idx]}"
                    )
                    continue
                seen[idx] = j
            # Ensure indices are 0..n_jumps-1 (contiguous).
            for idx in seen:
                if idx >= n_jumps:
                    errors.append(
                        f"FOLD FOREST ERROR: Jump edge {seen[idx]} in "
                        f"pose {p} has a non-contiguous-starting-at-0 "
                        f"jump index (n jumps total: {n_jumps})"
                    )

        if errors:
            raise ValueError("\n".join(errors))

    # ---------------- kinforest construction ----------------

    def build_kinforest(self, ps: Any):
        """Build a simplified atom-level :class:`KinForest` for this
        fold forest and pose-stack.

        This is **not** a faithful port of the generational scan-ordering
        builder from upstream ``tmol.kinematics.scan_ordering``. It
        produces a single-pose (or per-pose concatenated) KinForest that
        is *topologically* correct — every real atom has a parent that
        was emitted earlier, the three-atom reference frame is well-
        defined, and jumps / root-jumps land on the right atoms — but
        does not attempt to minimize the number of generations or
        replicate the C++ scan paths.

        Walk
        ----
        * Node ``0`` is the global root (``NodeType.root``) with
          ``id=-1`` and self-referencing frame indices.
        * For each pose, each tree in the fold forest is entered via a
          single jump or root-jump node. That node lands on the
          *first atom* of the root block (or jump destination block).
        * Polymer edges are walked block by block from ``start`` to
          ``end``. For each block, atoms are emitted in icoor-tree order
          (``RefinedResidueType.icoors_ancestors``). The first atom of
          block ``b`` (b > 0 within the edge) is parented to the last
          atom of block ``b - 1`` that is marked as the ``up``
          connection atom (if any); otherwise the first atom is parented
          to the previous block's last emitted atom.
        * Reference-frame indices ``(frame_x, frame_y, frame_z)`` are
          set to ``(self, parent, grandparent)`` — where "grandparent"
          falls back to the parent when no grandparent exists.

        Returns
        -------
        :class:`jtmol.kinematics.datatypes.KinForest`
        """
        # Local import to avoid a circular dependency at module load time.
        from jtmol.kinematics.datatypes import KinForest, NodeType

        edges = as_np(self.edges, dtype=np.int64)
        n_edges_arr = as_np(self.n_edges, dtype=np.int64)
        bti = as_np(ps.block_type_ind, dtype=np.int64)
        bco = as_np(ps.block_coord_offset, dtype=np.int64)
        pbt = ps.packed_block_types
        n_atoms_per_type = as_np(pbt.n_atoms, dtype=np.int64)
        conn_atoms = as_np(pbt.conn_atoms, dtype=np.int64)
        up_c, down_c = _pbt_up_down_conn_arrays(pbt)

        n_poses = bti.shape[0]
        max_n_atoms_per_pose = int(ps.max_n_atoms_per_pose)
        atom_offset = np.arange(n_poses, dtype=np.int64) * max_n_atoms_per_pose

        # Initialise growable node lists; entry 0 is the root node.
        ids: List[int] = [-1]
        doftypes: List[int] = [int(NodeType.root)]
        parents: List[int] = [0]
        frame_xs: List[int] = [0]
        frame_ys: List[int] = [0]
        frame_zs: List[int] = [0]

        # Global atom-index (in pose-stack coord layout) -> KFO index.
        # Missing atoms will simply not be emitted.
        to_kfo: dict = {}

        def emit(
            id_: int,
            doftype: int,
            parent: int,
            frame_x: int,
            frame_y: int,
            frame_z: int,
        ) -> int:
            idx = len(ids)
            ids.append(id_)
            doftypes.append(doftype)
            parents.append(parent)
            frame_xs.append(frame_x)
            frame_ys.append(frame_y)
            frame_zs.append(frame_z)
            return idx

        def emit_block(
            p: int,
            b: int,
            first_atom_parent_kfo: int,
            first_atom_doftype: int,
        ) -> Optional[int]:
            """Emit all atoms of block ``b`` in pose ``p`` in icoor order.

            ``first_atom_parent_kfo`` is the KFO index that should become
            the parent of this block's first emitted atom (either a jump
            node, or the ``up`` atom of the previous block in a polymer
            walk, or the previous block's last atom as a fallback).
            ``first_atom_doftype`` is the doftype assigned to that atom —
            typically ``bond``, but ``jump`` when entering via a jump.

            Returns the KFO index of this block's ``up``-connection atom
            (i.e. the atom that the *next* polymer block should be
            parented to), or the last emitted atom if no up connection
            exists, or ``None`` if the block is padding.
            """
            bt = int(bti[p, b])
            if bt < 0:
                return None
            n_at = int(n_atoms_per_type[bt])
            if n_at == 0:
                return None

            # Per-atom ancestor table in local block indices.
            rt = pbt.active_block_types[bt]
            ancestors_local = getattr(rt, "icoors_ancestors", None)
            if ancestors_local is None:
                # Fall back to a flat parent=previous-atom chain.
                ancestors_local = np.stack(
                    [
                        np.clip(np.arange(n_at) - 1, 0, None),
                        np.clip(np.arange(n_at) - 2, 0, None),
                        np.clip(np.arange(n_at) - 3, 0, None),
                    ],
                    axis=1,
                ).astype(np.int64)
            ancestors_local = np.asarray(ancestors_local, dtype=np.int64)

            global_base = int(bco[p, b]) + int(atom_offset[p])
            # Map local atom index -> KFO index as we emit.
            local_to_kfo: dict = {}

            for a in range(n_at):
                global_atom = global_base + a
                if a == 0:
                    parent_kfo = first_atom_parent_kfo
                    dof = first_atom_doftype
                    # Frame y/z fall back to the parent's own frame if we
                    # don't yet have a local grandparent.
                    frame_y_kfo = parent_kfo
                    frame_z_kfo = (
                        parents[parent_kfo] if parent_kfo != 0 else 0
                    )
                else:
                    # Local ancestors: (parent, gp, ggp) in block-local
                    # atom indices. Indices < 0 map to the block's first
                    # atom (already emitted).
                    par_local = int(ancestors_local[a, 0])
                    gp_local = int(ancestors_local[a, 1])
                    if par_local < 0 or par_local >= a:
                        par_local = a - 1
                    if gp_local < 0 or gp_local >= a:
                        gp_local = max(par_local - 1, 0)
                    parent_kfo = local_to_kfo[par_local]
                    frame_y_kfo = parent_kfo
                    frame_z_kfo = local_to_kfo.get(gp_local, parent_kfo)
                    dof = int(NodeType.bond)

                my_kfo = emit(
                    id_=global_atom,
                    doftype=dof,
                    parent=parent_kfo,
                    frame_x=0,  # frame_x = self; filled in below
                    frame_y=frame_y_kfo,
                    frame_z=frame_z_kfo,
                )
                # Patch frame_x to self after we know our own KFO index.
                frame_xs[my_kfo] = my_kfo
                local_to_kfo[a] = my_kfo
                to_kfo[global_atom] = my_kfo

            # Determine the "up" connection atom for this block, for use
            # by the next block in a polymer walk.
            up_ind = int(up_c[bt])
            if up_ind >= 0:
                up_atom_local = int(conn_atoms[bt, up_ind])
                if 0 <= up_atom_local < n_at:
                    return local_to_kfo[up_atom_local]
            # Fallback: the last emitted atom.
            return local_to_kfo[n_at - 1]

        def block_down_atom_kfo(p: int, b: int) -> Optional[int]:
            """Return the KFO index of block ``b``'s ``down`` atom if
            that block has already been emitted; otherwise ``None``."""
            bt = int(bti[p, b])
            if bt < 0:
                return None
            down_ind = int(down_c[bt])
            if down_ind < 0:
                return None
            down_atom_local = int(conn_atoms[bt, down_ind])
            global_atom = (
                int(bco[p, b]) + int(atom_offset[p]) + down_atom_local
            )
            return to_kfo.get(global_atom)

        # Emit atoms pose by pose. For each pose we first plant the
        # root-jump trees and then extend them through polymer / jump
        # edges. We make a single pass in edge order, which is valid
        # because ``from_pose_stack`` lists root-jumps before their
        # descendants.
        for p in range(n_poses):
            ne = int(n_edges_arr[p])
            # First, identify every polymer edge's start to know where
            # polymer walks begin from.
            polymer_starts: dict = {}
            for j in range(ne):
                t = int(edges[p, j, 0])
                if t == int(EdgeType.polymer):
                    polymer_starts[int(edges[p, j, 1])] = int(edges[p, j, 2])

            def walk_polymer(p: int, start: int, end: int, parent_kfo: int) -> None:
                """Emit a polymer chain from ``start`` to ``end`` (inclusive)
                whose first block is parented to ``parent_kfo`` via a jump
                / root-jump."""
                step = 1 if start <= end else -1
                prev_up_kfo = parent_kfo
                prev_dof = int(NodeType.jump)
                for b in range(start, end + step, step):
                    # For the first block of the walk, the incoming
                    # connection is the jump; for subsequent blocks, it's
                    # a polymer bond so doftype=bond.
                    new_up = emit_block(
                        p,
                        b,
                        first_atom_parent_kfo=prev_up_kfo,
                        first_atom_doftype=prev_dof,
                    )
                    if new_up is None:
                        break
                    prev_up_kfo = new_up
                    prev_dof = int(NodeType.bond)

            for j in range(ne):
                row = edges[p, j]
                t = int(row[0])
                if t == int(EdgeType.root_jump):
                    end_b = int(row[2])
                    # Emit root-jump node landing on end block's first
                    # atom.
                    # First atom of the root block is handled inside
                    # emit_block with doftype=jump; it is parented to the
                    # global root (KFO index 0).
                    if end_b in polymer_starts:
                        # Fuse with polymer edge starting at end_b.
                        walk_polymer(
                            p, end_b, polymer_starts[end_b], parent_kfo=0
                        )
                    else:
                        emit_block(
                            p,
                            end_b,
                            first_atom_parent_kfo=0,
                            first_atom_doftype=int(NodeType.jump),
                        )
                elif t == int(EdgeType.jump):
                    start_b = int(row[1])
                    end_b = int(row[2])
                    # Parent = down-connection atom (if any) of
                    # start_b, or fallback to start_b's last atom.
                    parent_kfo = block_down_atom_kfo(p, start_b)
                    if parent_kfo is None:
                        # start_b may not have been emitted yet; emit it.
                        emit_block(
                            p,
                            start_b,
                            first_atom_parent_kfo=0,
                            first_atom_doftype=int(NodeType.jump),
                        )
                        parent_kfo = block_down_atom_kfo(p, start_b)
                        if parent_kfo is None:
                            # Fallback: last atom of start_b we just emitted.
                            bt = int(bti[p, start_b])
                            last_local = int(n_atoms_per_type[bt]) - 1
                            parent_kfo = to_kfo[
                                int(bco[p, start_b])
                                + int(atom_offset[p])
                                + last_local
                            ]
                    if end_b in polymer_starts:
                        walk_polymer(
                            p,
                            end_b,
                            polymer_starts[end_b],
                            parent_kfo=parent_kfo,
                        )
                    else:
                        emit_block(
                            p,
                            end_b,
                            first_atom_parent_kfo=parent_kfo,
                            first_atom_doftype=int(NodeType.jump),
                        )
                # polymer edges handled via ``walk_polymer``; chemical
                # edges are currently unsupported (treated as no-op in
                # this simplified builder).

        arr = lambda xs: jnp.asarray(np.asarray(xs, dtype=np.int32), dtype=jnp.int32)
        return KinForest(
            id=arr(ids),
            doftype=arr(doftypes),
            parent=arr(parents),
            frame_x=arr(frame_xs),
            frame_y=arr(frame_ys),
            frame_z=arr(frame_zs),
        )


__all__ = ["EdgeType", "FoldForest"]
