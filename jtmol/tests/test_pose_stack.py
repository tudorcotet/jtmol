"""Unit tests for ``jtmol.pose.pose_stack.PoseStack`` pytree behavior.

These tests hand-craft a minimal ``PoseStack`` (no PDB loading, no
chemical database) and verify that:

  * It flattens / unflattens as a JAX pytree.
  * It can be threaded through ``jit`` and ``vmap``.
  * Pose-axis indexing (``ps[0]``) returns a well-formed 1-pose stack.

We build a stand-in ``packed_block_types`` object — just enough for
PoseStack construction and slicing to work. Since :mod:`jtmol.types`
``pytree_dataclass`` decides leaves/aux by *per-instance* ``is_array``
checks, any non-array stand-in becomes pytree aux automatically.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

pose_stack_mod = pytest.importorskip("jtmol.pose.pose_stack")
PoseStack = pose_stack_mod.PoseStack


# ---------------------------------------------------------------------------
# Minimal fixture helpers
# ---------------------------------------------------------------------------


class _FakePackedBlockTypes:
    """Non-array stand-in used as the ``packed_block_types`` aux leaf.

    ``PoseStack`` only touches ``n_atoms`` / ``active_block_types`` when
    resolving block-level helpers; the pytree tests below don't exercise
    those paths. We still provide the fields so :meth:`PoseStack.n_res_per_pose`
    and the ``max_n_atoms_per_block`` property work without crashing.
    """

    def __init__(self, n_atoms_per_type, max_n_atoms):
        # Stored as a plain ``numpy`` array so the pytree registration of
        # ``PoseStack`` keeps this object in aux (not leaves).
        self.n_atoms = np.asarray(n_atoms_per_type, dtype=np.int32)
        self.max_n_atoms = int(max_n_atoms)
        # Minimal sequence so ``block_type(...)`` wouldn't explode if called.
        self.active_block_types = tuple(
            object() for _ in range(len(self.n_atoms))
        )


def _minimal_pose_stack(
    n_poses: int = 2, packed_block_types=None
) -> "PoseStack":
    """Construct a small, self-consistent ``PoseStack``.

    Shape choices:
      * n_poses poses, each with 3 blocks.
      * Each of 2 block types has 4 atoms, so the per-pose coord slab is
        3*4=12 atoms.
      * max_n_conn = 1 for the connectivity tensors.

    ``packed_block_types`` is accepted as an argument so callers (e.g.
    the ``tree_map`` test) can share the same palette across multiple
    pose stacks — the pytree aux includes the palette by Python
    identity, so two stacks must share the exact same object to have
    matching treedefs.
    """
    n_blocks = 3
    n_types = 2
    atoms_per_type = 4
    max_n_atoms_per_pose = n_blocks * atoms_per_type
    max_n_conn = 1

    if packed_block_types is None:
        packed_block_types = _FakePackedBlockTypes(
            n_atoms_per_type=[atoms_per_type] * n_types,
            max_n_atoms=atoms_per_type,
        )
    pbt = packed_block_types

    # block_type_ind: fill with a predictable repeating pattern of types.
    block_type_ind = jnp.asarray(
        np.tile(np.arange(n_blocks) % n_types, (n_poses, 1)).astype(np.int32)
    )

    # coords: an arbitrary-but-distinct float32 grid so sums stay finite and
    # each entry has a unique address for debugging.
    coords = jnp.asarray(
        np.arange(n_poses * max_n_atoms_per_pose * 3, dtype=np.float32).reshape(
            n_poses, max_n_atoms_per_pose, 3
        )
    )

    inter_residue_connections = jnp.full(
        (n_poses, n_blocks, max_n_conn, 2), -1, dtype=jnp.int32
    )
    inter_block_bondsep = jnp.full(
        (n_poses, n_blocks, n_blocks, max_n_conn, max_n_conn),
        6,
        dtype=jnp.int32,
    )

    return PoseStack.from_components(
        packed_block_types=pbt,
        coords=coords,
        block_type_ind=block_type_ind,
        inter_residue_connections=inter_residue_connections,
        inter_block_bondsep=inter_block_bondsep,
        pdb_info=None,
    )


# ---------------------------------------------------------------------------
# Pytree flatten / unflatten
# ---------------------------------------------------------------------------


def test_pytree_flatten_unflatten():
    """``jax.tree.leaves(ps)`` returns every array field; ``unflatten`` rebuilds.

    ``PoseStack`` is registered as a pytree whose leaves are its array
    fields (``coords``, ``block_coord_offset``, ``block_coord_offset64``,
    ``inter_residue_connections``, ``inter_block_bondsep``, ``block_type_ind``).
    The non-array fields (``packed_block_types``, ``pdb_info``) land in aux
    and must be preserved by a flatten/unflatten roundtrip.
    """
    ps = _minimal_pose_stack()

    leaves, treedef = jax.tree_util.tree_flatten(ps)

    # Every leaf should be a jax.Array (or ndarray); aux contains the
    # non-array stand-in packed_block_types + None pdb_info.
    for leaf in leaves:
        assert isinstance(leaf, (jax.Array, np.ndarray))

    expected_array_fields = {
        "coords",
        "block_coord_offset",
        "block_coord_offset64",
        "inter_residue_connections",
        "inter_block_bondsep",
        "block_type_ind",
    }
    assert len(leaves) == len(expected_array_fields), (
        f"expected {len(expected_array_fields)} array leaves; got {len(leaves)}"
    )

    ps2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(ps2, PoseStack)
    # Aux data (packed_block_types, pdb_info) must be preserved by identity.
    assert ps2.packed_block_types is ps.packed_block_types
    assert ps2.pdb_info is ps.pdb_info
    # Array fields must round-trip bit-identically.
    np.testing.assert_array_equal(np.asarray(ps2.coords), np.asarray(ps.coords))
    np.testing.assert_array_equal(
        np.asarray(ps2.block_type_ind), np.asarray(ps.block_type_ind)
    )


# ---------------------------------------------------------------------------
# jit / vmap
# ---------------------------------------------------------------------------


def test_pytree_jit():
    """A function taking a PoseStack and returning ``coords.sum()`` jits.

    This is the main reason PoseStack is a pytree: we need JAX tracing to
    see through the container and treat coords as a differentiable leaf.
    """
    ps = _minimal_pose_stack()

    @jax.jit
    def total(ps_in):
        return jnp.sum(ps_in.coords)

    eager = float(jnp.sum(ps.coords))
    jitted = float(total(ps))
    # Tolerance: we're summing float32 values; the only source of drift is
    # associativity of sum, which JAX preserves here.
    assert jitted == pytest.approx(eager, rel=0.0, abs=1e-3)


def test_pytree_vmap():
    """vmap over a leading pose-axis of PoseStack-like pytree works.

    We stack two ``PoseStack`` objects with ``jax.tree.map`` so the leading
    axis becomes a batch axis over PoseStacks, then vmap a function that
    sums each pose stack's coords. This exercises the same treedef /
    leaf-shape contract that the minimizer relies on.

    Both pose stacks must share the same ``packed_block_types`` instance
    (it is pytree aux and compared by Python identity), so treedefs
    match and ``tree_map`` can zip leaves together.
    """
    shared_pbt = _FakePackedBlockTypes(
        n_atoms_per_type=[4, 4], max_n_atoms=4
    )
    ps_a = _minimal_pose_stack(packed_block_types=shared_pbt)
    ps_b = _minimal_pose_stack(packed_block_types=shared_pbt)

    stacked = jax.tree_util.tree_map(
        lambda a, b: jnp.stack([a, b], axis=0), ps_a, ps_b
    )

    # Each leaf should now have an extra leading dim of length 2.
    leaves = jax.tree_util.tree_leaves(stacked)
    assert all(leaf.shape[0] == 2 for leaf in leaves), (
        "leading batch axis should have size 2 after stacking two pose-stacks"
    )

    def per_pose_sum(ps_in):
        return jnp.sum(ps_in.coords)

    out = jax.vmap(per_pose_sum)(stacked)
    assert out.shape == (2,)
    # Both halves were built identically, so the vmapped sums should match
    # and equal the eager sum on ``ps_a``.
    expected = float(jnp.sum(ps_a.coords))
    np.testing.assert_allclose(
        np.asarray(out),
        np.asarray([expected, expected]),
        rtol=0.0,
        atol=1e-3,
    )


# ---------------------------------------------------------------------------
# Pose-axis indexing
# ---------------------------------------------------------------------------


def test_pose_stack_indexing():
    """``ps[0]`` returns a 1-pose PoseStack with sliced array leaves.

    Checks the slicing behaviour defined in ``PoseStack.__getitem__``:

      * Returned object is still a ``PoseStack``.
      * ``n_poses == 1`` after indexing.
      * Array leaves are sliced along axis 0; their trailing shapes match.
      * ``packed_block_types`` (aux) is preserved by identity.
    """
    ps = _minimal_pose_stack(n_poses=3)
    assert ps.n_poses == 3

    sub = ps[0]
    assert isinstance(sub, PoseStack)
    assert sub.n_poses == 1
    assert sub.coords.shape == (1, ps.coords.shape[1], ps.coords.shape[2])
    assert sub.block_type_ind.shape == (1, ps.block_type_ind.shape[1])
    # Aux must be preserved by identity — the palette is shared.
    assert sub.packed_block_types is ps.packed_block_types
    # Content of the slice must match the original's first pose.
    np.testing.assert_array_equal(
        np.asarray(sub.coords[0]), np.asarray(ps.coords[0])
    )
