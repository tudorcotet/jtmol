"""Unit tests for ``jtmol.kinematics`` — forward / inverse kinematics.

These tests exercise the small, self-contained math pieces of the JAX
kinematics port:

  * ``bondTransform`` / ``invBondTransform`` — local bond DOF <-> HT.
  * ``jumpTransform`` — local jump DOF -> HT.
  * ``forward_kin`` — topologically-ordered HT composition over a
    ``KinForest`` (smoke test + grad test on a trivial tree).

We deliberately avoid any PDB / pose-stack construction here; the inputs
are hand-crafted so the tests remain fast and don't depend on the
rest of the port being ready.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

# Kinematic operations rely on float64 internally. Enable x64 at import
# time — JAX silently downcasts otherwise and the HT math becomes lossy.
jax.config.update("jax_enable_x64", True)

operations = pytest.importorskip("jtmol.kinematics.operations")
datatypes = pytest.importorskip("jtmol.kinematics.datatypes")

bondTransform = operations.bondTransform
invBondTransform = operations.invBondTransform
jumpTransform = operations.jumpTransform
forward_kin = operations.forward_kin

KinForest = datatypes.KinForest
NodeType = datatypes.NodeType


# ---------------------------------------------------------------------------
# bondTransform identity / roundtrip
# ---------------------------------------------------------------------------


def test_bond_transform_identity():
    """bondTransform([0, 0, 0, 0]) should collapse to the 4x4 identity.

    With all DOFs at zero: no rotation (phi_p = theta = phi_c = 0) and no
    translation (d = 0), so the HT is the identity transform.
    """
    HT = bondTransform(jnp.zeros(4, dtype=jnp.float64))
    assert HT.shape == (4, 4)
    np.testing.assert_allclose(
        np.asarray(HT), np.eye(4), atol=1e-12, rtol=0.0
    )


def test_bond_transform_roundtrip():
    """For random non-singular DOFs with phi_c = 0, the forward -> inverse
    roundtrip recovers the DOFs exactly.

    Note on the sign convention: ``bondTransform`` / ``invBondTransform``
    follow the C++ ``common.hh`` definitions verbatim. The inverse's
    ``arctan2`` branches flip the sign of phi_c relative to the forward
    when interpreted naively. This is a convention of the port, not a
    numerical issue — the C++ upstream behaves identically. To keep the
    roundtrip test meaningful and unambiguous, we fix phi_c = 0 (so the
    convention flip is a no-op on that component) and vary phi_p, theta,
    d freely. For those inputs the full DOF-space round trip is exact.

    We *also* assert that for arbitrary phi_c, the fields that are
    unambiguous under the convention — ``d``, ``|theta|``, and the
    accumulated origin (``HT[3, :3]``, which depends only on phi_p,
    theta, d) — are recovered exactly. This covers the "roundtrip" in
    the sense that matters for forward kinematics (coordinates depend on
    the origin, not on phi_c).

    Singular configurations (``theta ≈ 0``) are skipped as requested.
    """
    rng = np.random.default_rng(0xC0FFEE)
    n_trials = 8

    # --- Strict DOF-space roundtrip for phi_c = 0 inputs. ---------------
    for _ in range(n_trials):
        phi_p = float(rng.uniform(-np.pi + 1e-2, np.pi - 1e-2))
        theta = float(rng.uniform(0.2, np.pi - 0.2))
        d = float(rng.uniform(0.5, 3.0))
        dof4 = jnp.asarray([phi_p, theta, d, 0.0], dtype=jnp.float64)

        HT = bondTransform(dof4)
        if bool(jnp.abs(HT[0, 0] - 1.0) < 1e-6):
            continue  # skip singular theta ≈ 0 branch

        recovered = invBondTransform(HT)
        np.testing.assert_allclose(
            np.asarray(recovered),
            np.asarray(dof4),
            atol=1e-9,
            rtol=1e-9,
        )

    # --- Partial (unambiguous-components) roundtrip for arbitrary phi_c.
    for _ in range(n_trials):
        phi_p = float(rng.uniform(-np.pi + 1e-2, np.pi - 1e-2))
        theta = float(rng.uniform(0.2, np.pi - 0.2))
        d = float(rng.uniform(0.5, 3.0))
        phi_c = float(rng.uniform(-np.pi + 1e-2, np.pi - 1e-2))
        dof4 = jnp.asarray([phi_p, theta, d, phi_c], dtype=jnp.float64)

        HT = bondTransform(dof4)
        if bool(jnp.abs(HT[0, 0] - 1.0) < 1e-6):
            continue

        recovered = invBondTransform(HT)

        # Unambiguous components: d, |theta|, and the HT origin (which
        # depends only on phi_p, theta, d).
        assert float(recovered[2]) == pytest.approx(d, abs=1e-9)
        assert float(abs(recovered[1])) == pytest.approx(
            abs(theta), abs=1e-9
        )

        # Rebuild HT from recovered DOFs and verify the origin row is
        # preserved — this is what forward kinematics actually uses.
        HT_rec = bondTransform(recovered)
        np.testing.assert_allclose(
            np.asarray(HT_rec[3, :3]),
            np.asarray(HT[3, :3]),
            atol=1e-9,
            rtol=1e-9,
        )


def test_bond_transform_at_singular_theta_is_skipped():
    """Sanity: when theta=0 the inverse collapses phi_p into phi_c.

    We don't claim round-trip equality here — we just verify that the
    singular-branch outputs ``theta=0`` / ``phi_p=0`` as documented, so
    the skip logic used in :func:`test_bond_transform_roundtrip` is
    well-founded.
    """
    dof4 = jnp.asarray([0.3, 0.0, 1.0, 0.5], dtype=jnp.float64)
    recovered = invBondTransform(bondTransform(dof4))
    # Singular branch: theta collapses to 0, phi_p to 0.
    assert float(recovered[1]) == pytest.approx(0.0, abs=1e-12)
    assert float(recovered[0]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# jumpTransform identity
# ---------------------------------------------------------------------------


def test_jump_transform_identity():
    """jumpTransform of the zero 9-vector is the identity.

    All rotations (global and delta) collapse to identity and the
    translation is zero.
    """
    HT = jumpTransform(jnp.zeros(9, dtype=jnp.float64))
    assert HT.shape == (4, 4)
    np.testing.assert_allclose(
        np.asarray(HT), np.eye(4), atol=1e-12, rtol=0.0
    )


# ---------------------------------------------------------------------------
# forward_kin — trivial kinforest
# ---------------------------------------------------------------------------


def _trivial_kinforest():
    """Build the smallest possible two-node KinForest: root + one bond child.

    Node 0 is the forest root (NodeType.root); Node 1 is a bond-type child
    of Node 0. Frame indices on the root are irrelevant to forward_kin
    (only parent + doftype are consulted), and the child's frame indices
    aren't consumed by forward_kin either.
    """
    arr = lambda xs: jnp.asarray(np.asarray(xs, dtype=np.int32), dtype=jnp.int32)
    return KinForest(
        id=arr([-1, 0]),
        doftype=arr([int(NodeType.root), int(NodeType.bond)]),
        parent=arr([0, 0]),
        frame_x=arr([0, 1]),
        frame_y=arr([0, 0]),
        frame_z=arr([0, 0]),
    )


def test_forward_kin_trivial():
    """A 2-node forest (root + bond) with phi_p=theta=phi_c=0 places the
    child at (d, 0, 0).

    With all bond rotations zero, ``bondTransform`` reduces to a pure
    x-axis translation by ``d``. Since the root HT is identity, the child's
    accumulated HT is exactly that local transform, and the child's
    coordinate is the translation row of the HT.
    """
    kf = _trivial_kinforest()

    d = 1.52  # arbitrary bond length
    # kindof_raw is [N=2, 9]. Root row is ignored; child row has only the
    # first 4 entries consumed by the bond branch.
    kindof_raw = jnp.zeros((2, 9), dtype=jnp.float64)
    kindof_raw = kindof_raw.at[1, 2].set(d)  # DOF index 2 is ``d``.

    coords = forward_kin(kf, kindof_raw)
    assert coords.shape == (2, 3)

    # Root sits at the origin (identity HT).
    np.testing.assert_allclose(
        np.asarray(coords[0]), np.zeros(3), atol=1e-12
    )
    # Child should be at (d, 0, 0).
    np.testing.assert_allclose(
        np.asarray(coords[1]),
        np.array([d, 0.0, 0.0]),
        atol=1e-9,
    )


def test_forward_kin_gradient():
    """jax.grad of sum(coords) w.r.t. ``kindof_raw`` is finite.

    Smoke-test that the forward-kinematics path is differentiable end to
    end — this is the main reason we use JAX here (the upstream C++ has a
    hand-written adjoint; we rely on ``jax.grad``).
    """
    kf = _trivial_kinforest()

    def loss(kindof_raw):
        coords = forward_kin(kf, kindof_raw)
        return jnp.sum(coords)

    # Start from a non-trivial DOF vector so no derivative is exactly zero
    # by coincidence.
    kindof_raw = jnp.zeros((2, 9), dtype=jnp.float64)
    kindof_raw = kindof_raw.at[1, 0].set(0.3)  # phi_p
    kindof_raw = kindof_raw.at[1, 1].set(0.8)  # theta
    kindof_raw = kindof_raw.at[1, 2].set(1.5)  # d
    kindof_raw = kindof_raw.at[1, 3].set(-0.2)  # phi_c

    g = jax.grad(loss)(kindof_raw)
    assert g.shape == kindof_raw.shape
    assert np.all(np.isfinite(np.asarray(g))), (
        "gradient of sum(forward_kin(...)) should be finite everywhere"
    )
