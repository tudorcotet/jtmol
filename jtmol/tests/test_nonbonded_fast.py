"""Parity of the fast S2 nonbonded path against the generic substrate.

:func:`jtmol.nonbonded.fast.build_fast_nonbonded` is an optimized-but-parity-
exact replacement for :func:`jtmol.nonbonded.multipose.s2_energies`: it swaps
the ljlk/elec backward for a sorted ``segment_sum`` (custom_vjp) and the lk_ball
forward for the bridge-split (OPT3) edge/water decomposition. The forward math
is reused verbatim, so both the per-channel energy and the weighted-total
gradient must match the substrate to float reduction order.

Runs at ``float64`` on CPU — no GPU or tmol dependency — so it is a
first-class part of the suite. (Performance benchmarks live in
``benchmarks/``; this file only guards correctness.)
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

# Official beta2016 nonbonded weights — signs matter for the gradient check
# (lk_ball_iso/bridge/bridge_uncpl are negative), so a channel-swapped bug that
# an unweighted sum would hide still shows up.
_WEIGHTS = {
    "fa_ljatr": 1.0, "fa_ljrep": 0.55, "fa_lk": 1.0, "fa_elec": 1.0,
    "lk_ball_iso": -0.38, "lk_ball": 0.92, "lk_bridge": -0.33,
    "lk_bridge_uncpl": -0.33, "hbond": 1.0,
}


@pytest.fixture(scope="module")
def s2_setup(pdb_1ubq):
    """Build an S2 bundle (P=4 perturbed 1ubq) + fast/generic scorers, or skip."""
    try:
        import jtmol
        from jtmol.database import ParameterDatabase
        from jtmol.nonbonded.multipose import setup_s2, s2_energies
        from jtmol.nonbonded.fast import build_fast_nonbonded
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"nonbonded fast/multipose import failed: {exc!r}")

    from jtmol.score.score_types import ScoreType

    ps = jtmol.pose_stack_from_pdb(pdb_1ubq)
    db = ParameterDatabase.get_default()
    base = np.asarray(ps.coords[0], dtype=np.float64)
    rng = np.random.default_rng(0)
    coords = jnp.asarray(np.stack([base + rng.normal(size=base.shape) * 0.3 for _ in range(4)]))

    bundle = setup_s2(ps, np.asarray(coords), db)
    fast = build_fast_nonbonded(bundle)
    weights = {getattr(ScoreType, k): w for k, w in _WEIGHTS.items()}
    return {
        "coords": coords,
        "ref_fn": lambda c: s2_energies(bundle, c),
        "fast_fn": fast,
        "channels": list(weights),
        "weights": weights,
    }


def test_fast_nonbonded_value_parity(s2_setup):
    """Every channel's per-pose energy matches the generic substrate."""
    ref = s2_setup["ref_fn"](s2_setup["coords"])
    got = s2_setup["fast_fn"](s2_setup["coords"])
    for st in s2_setup["channels"]:
        r, g = np.asarray(ref[st]), np.asarray(got[st])
        rel = float(np.abs(r - g).max() / (np.abs(r).max() + 1e-12))
        assert rel < 1e-6, f"{st.name} value parity failed: rel {rel:.2e}"


def test_fast_nonbonded_grad_parity(s2_setup):
    """The weighted-total gradient w.r.t. coords matches the generic substrate."""
    coords = s2_setup["coords"]
    weights = s2_setup["weights"]

    def total(fn, c):
        e = fn(c)
        return sum(jnp.sum(e[st]) * w for st, w in weights.items())

    gref = np.asarray(jax.grad(lambda c: total(s2_setup["ref_fn"], c))(coords))
    gfast = np.asarray(jax.grad(lambda c: total(s2_setup["fast_fn"], c))(coords))
    rel = float(np.abs(gref - gfast).max() / (np.abs(gref).max() + 1e-12))
    assert rel < 1e-4, f"weighted-total grad parity failed: rel {rel:.2e}"


@pytest.fixture(scope="module")
def s3_setup(pdb_1ubq, pdb_1qys):
    """Build an S3 ragged bundle (2 distinct topologies) + fast/generic scorers."""
    try:
        import jtmol
        from jtmol.database import ParameterDatabase
        from jtmol.nonbonded.multipose import setup_s3, s3_energies
        from jtmol.nonbonded.fast import build_fast_nonbonded_ragged
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"nonbonded fast/multipose S3 import failed: {exc!r}")

    from jtmol.score.score_types import ScoreType

    ps1 = jtmol.pose_stack_from_pdb(pdb_1ubq)
    ps2 = jtmol.pose_stack_from_pdb(pdb_1qys)
    db = ParameterDatabase.get_default()
    b3 = setup_s3([ps1, ps2], db)
    fast = build_fast_nonbonded_ragged(b3)
    weights = {getattr(ScoreType, k): w for k, w in _WEIGHTS.items()}
    return {
        "coords": jnp.asarray(b3.coords_flat0),
        "ref_fn": lambda c: s3_energies(b3, c),
        "fast_fn": fast,
        "channels": list(weights),
        "weights": weights,
    }


def test_fast_nonbonded_ragged_value_parity(s3_setup):
    """Every channel's per-pose energy matches the generic ragged substrate (S3)."""
    ref = s3_setup["ref_fn"](s3_setup["coords"])
    got = s3_setup["fast_fn"](s3_setup["coords"])
    for st in s3_setup["channels"]:
        r, g = np.asarray(ref[st]), np.asarray(got[st])
        rel = float(np.abs(r - g).max() / (np.abs(r).max() + 1e-12))
        assert rel < 1e-6, f"{st.name} S3 value parity failed: rel {rel:.2e}"


def test_fast_nonbonded_ragged_grad_parity(s3_setup):
    """The weighted-total gradient matches the generic ragged substrate (S3)."""
    coords = s3_setup["coords"]
    weights = s3_setup["weights"]

    def total(fn, c):
        e = fn(c)
        return sum(jnp.sum(e[st]) * w for st, w in weights.items())

    gref = np.asarray(jax.grad(lambda c: total(s3_setup["ref_fn"], c))(coords))
    gfast = np.asarray(jax.grad(lambda c: total(s3_setup["fast_fn"], c))(coords))
    rel = float(np.abs(gref - gfast).max() / (np.abs(gref).max() + 1e-12))
    assert rel < 1e-4, f"S3 weighted-total grad parity failed: rel {rel:.2e}"
