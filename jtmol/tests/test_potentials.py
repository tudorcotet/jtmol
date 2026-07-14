"""Numerical unit tests for the jtmol pure-JAX potentials.

These tests target the pure functional potentials and shared math
primitives delivered by the parallel porting agents:

* ``jtmol.score.ljlk.potentials`` — LJ (attractive/repulsive split,
  shift + fade) and LK isotropic desolvation.
* ``jtmol.score.elec.potentials`` — sigmoidal-dielectric Coulomb with
  short- and long-range cubic Hermite fades.
* ``jtmol.score.hbond.potentials`` — (imported but not exercised
  numerically here beyond the primitive geom/polynomial helpers it
  relies on).
* ``jtmol.score.common`` — count-pair connectivity weight, cubic
  Hermite interpolation, bounded polynomial evaluation, and vector
  geometry helpers (dihedral angle).

The tests are defensive: every test first ``pytest.importorskip``s the
module(s) it needs, so if a parallel agent did not ship their module,
only the tests for that module are skipped. Nothing hard-imports at
module level.

The tests exercise shape/sign behaviour only, not exact numerical
parity with upstream ``tmol`` — that is covered elsewhere. The goal is
to catch regressions like sign flips, broken cutoffs, and NaN leaks
through ``jax.grad``.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys

import jax
import jax.numpy as jnp
import pytest


def _import_or_skip(modname: str):
    """Import ``modname`` directly, even if the parent package ``__init__``
    is broken because a *different* parallel agent hasn't landed yet.

    Wraps :func:`pytest.importorskip` but first tries a file-based
    submodule load: if the target ``.py`` exists on disk we import it
    through :func:`importlib.util.spec_from_file_location`, which
    bypasses the parent package ``__init__.py``. That way a test for
    ``jtmol.score.ljlk.potentials`` still runs even if the sibling
    ``ljlk_energy_term`` (owned by a different parallel agent) isn't
    yet in place.

    Missing target modules (module file doesn't exist on disk at all)
    still produce a clean ``Skipped`` via ``pytest.importorskip``.
    """
    # Fast path: normal import succeeds (package is fully in place).
    if modname in sys.modules:
        return sys.modules[modname]
    try:
        return importlib.import_module(modname)
    except ImportError:
        pass

    # Slow path: locate the file on disk under jtmol/... and load it
    # directly. We walk from ``jtmol`` (the top-level package this
    # repo owns) upward from this test file's location, so the test
    # works whether run from the repo root or elsewhere.
    import pathlib
    here = pathlib.Path(__file__).resolve()
    # ``here`` is .../jtmol/tests/test_potentials.py; the jtmol
    # package root is two parents up.
    pkg_root = here.parent.parent
    # ``modname`` starts with ``jtmol.`` — strip that prefix and turn
    # the remainder into a file path.
    parts = modname.split(".")
    if parts[0] != pkg_root.name:
        return pytest.importorskip(modname)
    rel = parts[1:]
    candidate = pkg_root.joinpath(*rel).with_suffix(".py")
    if not candidate.exists():
        # Fall back to importorskip -- yields a proper skip.
        return pytest.importorskip(modname)

    spec = importlib.util.spec_from_file_location(modname, str(candidate))
    if spec is None or spec.loader is None:
        return pytest.importorskip(modname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        # Internal import inside the target module itself failed —
        # the target *is* delivered but a dep isn't. Treat as a skip.
        sys.modules.pop(modname, None)
        pytest.skip(f"could not load {modname}: dependencies not ready")
    return mod


# --------------------------------------------------------------------- #
# LJ helpers                                                             #
# --------------------------------------------------------------------- #


def _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1,
                        lk_volume=10.0, lk_dgfree=5.0, lk_lambda=3.5):
    """Build a duck-typed LJTypeParams stand-in with scalar fields.

    ``lj_atr_rep`` / ``lk_isotropic_pair`` just read attributes from
    ``i_params`` / ``j_params`` via ``.lj_radius`` etc., so any object
    with those attributes works (see module docstring in
    ``jtmol.score.ljlk.potentials``). Using a plain ``SimpleNamespace``
    keeps us decoupled from ``LJTypeParams`` — if that class ever needs
    its pytree registration changed, these tests still run.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        lj_radius=jnp.asarray(lj_radius, dtype=jnp.float32),
        lj_wdepth=jnp.asarray(lj_wdepth, dtype=jnp.float32),
        is_donor=jnp.asarray(False),
        is_acceptor=jnp.asarray(False),
        is_hydroxyl=jnp.asarray(False),
        is_polarh=jnp.asarray(False),
        lk_volume=jnp.asarray(lk_volume, dtype=jnp.float32),
        lk_dgfree=jnp.asarray(lk_dgfree, dtype=jnp.float32),
        lk_lambda=jnp.asarray(lk_lambda, dtype=jnp.float32),
    )


def _make_ljlk_global(lj_hbond_dis=3.0, lj_hbond_OH_donor_dis=2.6,
                      lj_hbond_hdis=1.75, max_dis=6.0,
                      lj_dlin_sigma_factor=0.6):
    """Duck-typed LJLKGlobal with the sigma-override + cutoff scalars."""
    from types import SimpleNamespace
    return SimpleNamespace(
        lj_hbond_dis=jnp.float32(lj_hbond_dis),
        lj_hbond_OH_donor_dis=jnp.float32(lj_hbond_OH_donor_dis),
        lj_hbond_hdis=jnp.float32(lj_hbond_hdis),
        max_dis=jnp.float32(max_dis),
        lj_dlin_sigma_factor=jnp.float32(lj_dlin_sigma_factor),
    )


def _make_elec_global(D=80.0, D0=6.0, S=0.36, min_dis=1.6, max_dis=5.5):
    """Duck-typed ElecGlobalParams. Defaults are close to beta_nov2016."""
    from types import SimpleNamespace
    return SimpleNamespace(
        D=jnp.float32(D),
        D0=jnp.float32(D0),
        S=jnp.float32(S),
        min_dis=jnp.float32(min_dis),
        max_dis=jnp.float32(max_dis),
    )


# A count-pair path length that yields weight 1.0 (no attenuation), so
# every energy sign/magnitude comes directly from the potential itself.
_PATH_LENGTH_FULL = jnp.int32(5)


# --------------------------------------------------------------------- #
# LJ / LK tests                                                          #
# --------------------------------------------------------------------- #


def test_lj_attractive_at_4A():
    """At dist=4 A, sigma=3.8, epsilon=0.1: the attractive channel should
    be negative (we're past the well but still inside the fade onset).

    At 4.0 A we are below cpoly_dmin (=4.5 A since sigma <= 4.5) so we
    sit on the bare vdW curve: ``V = eps * (sd^12 - 2 sd^6)`` with
    ``sd = 3.8/4``. That numerically gives roughly -0.093 kcal/mol —
    well within the negative half plane.
    """
    ljlk = _import_or_skip("jtmol.score.ljlk.potentials")

    i = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    j = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    gp = _make_ljlk_global()

    dist = jnp.float32(4.0)
    Vatr, Vrep = ljlk.lj_atr_rep(dist, _PATH_LENGTH_FULL, i, j, gp)

    # Shape: scalar-in -> scalar-out
    assert Vatr.shape == ()
    assert Vrep.shape == ()
    # dist > sigma -> Vrep is zero, Vatr is the full vdW value.
    assert float(Vrep) == pytest.approx(0.0, abs=1e-6)
    assert float(Vatr) < 0.0


def test_lj_zero_beyond_cutoff():
    """Past ``cpoly_dmax = 6.0 A``: both channels must return exactly 0."""
    ljlk = _import_or_skip("jtmol.score.ljlk.potentials")

    i = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    j = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    gp = _make_ljlk_global()

    dist = jnp.float32(7.0)
    Vatr, Vrep = ljlk.lj_atr_rep(dist, _PATH_LENGTH_FULL, i, j, gp)

    assert float(Vatr) == pytest.approx(0.0, abs=1e-6)
    assert float(Vrep) == pytest.approx(0.0, abs=1e-6)


def test_lj_repulsive_below_sigma():
    """Inside sigma the atoms clash and Vrep must go positive.

    With sigma=3.8 and dist=2 A we are *below* ``d_lin = 0.6*sigma = 2.28``
    so the potential is linearly extrapolated from ``d_lin``; the
    attractive portion is clamped at ``-epsilon`` and the rest is piled
    into ``Vrep``. Numerically Vrep ~ 100 kcal/mol — solidly positive.
    """
    ljlk = _import_or_skip("jtmol.score.ljlk.potentials")

    i = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    j = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    gp = _make_ljlk_global()

    dist = jnp.float32(2.0)
    Vatr, Vrep = ljlk.lj_atr_rep(dist, _PATH_LENGTH_FULL, i, j, gp)

    # Below sigma the attractive channel clamps at -epsilon.
    assert float(Vatr) == pytest.approx(-0.1, abs=1e-5)
    assert float(Vrep) > 0.0


def test_lj_differentiable():
    """``jax.grad`` through the LJ call must produce finite numbers.

    Branch-free constructs in :func:`lj_atr_rep` are what make the
    function grad-safe; this test guards against a regression where a
    ``jnp.where`` cascade accidentally leaks NaN through the unused
    branch (e.g. ``0 * inf``).
    """
    ljlk = _import_or_skip("jtmol.score.ljlk.potentials")

    i = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    j = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1)
    gp = _make_ljlk_global()

    def f(d):
        Vatr, Vrep = ljlk.lj_atr_rep(d, _PATH_LENGTH_FULL, i, j, gp)
        return Vatr + Vrep

    # Sample dist at several distinct regions: hard wall, well, fade,
    # beyond cutoff — grad should be finite for all of them.
    for d_val in [1.5, 2.5, 3.8, 4.8, 5.9, 6.5]:
        g = jax.grad(f)(jnp.float32(d_val))
        assert g.shape == ()
        assert jnp.isfinite(g), f"grad non-finite at d={d_val}: {g}"


def test_lk_isotropic_negative():
    """With positive ``lk_dgfree`` on both atoms, LK desolvation is
    attractive (negative energy) in its well.

    Using the symmetric pair ``(1.9 A, wdepth=0.1, volume=10, dg=5.0)``
    and dist=3.5 A (well inside ``d_min = 0.89*sigma = 3.38`` cutoff,
    before the far fade starts at 4.5). Each one-sided contribution is
    ``-vol * dg / (2 pi^1.5 lam) / d^2 * exp(-z^2)`` which is negative
    for positive ``dg`` — so the sum is strongly negative.
    """
    ljlk = _import_or_skip("jtmol.score.ljlk.potentials")

    i = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1,
                             lk_volume=10.0, lk_dgfree=5.0, lk_lambda=3.5)
    j = _make_lj_type_params(lj_radius=1.9, lj_wdepth=0.1,
                             lk_volume=10.0, lk_dgfree=5.0, lk_lambda=3.5)
    gp = _make_ljlk_global()

    dist = jnp.float32(3.5)
    V = ljlk.lk_isotropic_pair(dist, _PATH_LENGTH_FULL, i, j, gp)
    assert V.shape == ()
    assert float(V) < 0.0

    # Also check the cutoff: at 7 A (beyond cpoly_far_dmax=6) we get 0.
    V_far = ljlk.lk_isotropic_pair(jnp.float32(7.0), _PATH_LENGTH_FULL, i, j, gp)
    assert float(V_far) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------- #
# Electrostatics                                                          #
# --------------------------------------------------------------------- #


def test_elec_sign():
    """Opposite-sign charges attract (E < 0), same-sign charges repel (E > 0)
    inside the true-Coulomb region.

    At dist=4 A with the beta_nov2016 globals (``max_dis=5.5`` so the
    Coulombic region is ``[1.85, 4.5)``), the function is strictly the
    sigmoidal-dielectric Coulomb minus the baseline shift — sign
    follows ``e_i * e_j`` up to the positive (C1 / (d eps(d)) - C2)
    term, which is positive inside the Coulombic shell.
    """
    elec = _import_or_skip("jtmol.score.elec.potentials")

    gp = _make_elec_global()

    dist = jnp.float32(4.0)
    path = _PATH_LENGTH_FULL

    E_opp = elec.elec_pair(dist, jnp.float32(+1.0), jnp.float32(-1.0), path, gp)
    E_same = elec.elec_pair(dist, jnp.float32(+1.0), jnp.float32(+1.0), path, gp)

    assert E_opp.shape == ()
    assert E_same.shape == ()
    assert float(E_opp) < 0.0
    assert float(E_same) > 0.0
    # Symmetry: swapping signs negates the energy.
    assert float(E_opp) == pytest.approx(-float(E_same), rel=1e-5)

    # Beyond max_dis, energy is zero regardless of sign.
    E_far = elec.elec_pair(jnp.float32(7.0), jnp.float32(+1.0),
                           jnp.float32(-1.0), path, gp)
    assert float(E_far) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------- #
# Geometry primitives (dihedral)                                         #
# --------------------------------------------------------------------- #


def test_dihedral_zero():
    """Dihedral angle on canonical inputs:

    1. Co-linear: undefined in general. JAX's ``atan2(0, 0)`` returns 0
       rather than NaN, so we accept either.
    2. Trans conformation (planar, opposite sides of b-c): returns ±π.
    3. Cis conformation (planar, same side): returns 0.
    """
    geom = _import_or_skip("jtmol.score.common.geom")

    # --- case 1: co-linear ---
    a = jnp.asarray([0.0, 0.0, 0.0])
    b = jnp.asarray([1.0, 0.0, 0.0])
    c = jnp.asarray([2.0, 0.0, 0.0])
    d = jnp.asarray([3.0, 0.0, 0.0])
    phi = geom.dihedral_angle(a, b, c, d)
    assert phi.shape == ()
    # Allow either 0 or NaN — both are legitimate for a degenerate geometry.
    assert jnp.isnan(phi) or float(phi) == pytest.approx(0.0, abs=1e-5)

    # --- case 2: trans ---
    a = jnp.asarray([0.0, 1.0, 0.0])
    b = jnp.asarray([0.0, 0.0, 0.0])
    c = jnp.asarray([1.0, 0.0, 0.0])
    d = jnp.asarray([1.0, -1.0, 0.0])
    phi = geom.dihedral_angle(a, b, c, d)
    assert abs(float(phi)) == pytest.approx(math.pi, abs=1e-5)

    # --- case 3: cis ---
    a = jnp.asarray([0.0, 1.0, 0.0])
    b = jnp.asarray([0.0, 0.0, 0.0])
    c = jnp.asarray([1.0, 0.0, 0.0])
    d = jnp.asarray([1.0, 1.0, 0.0])
    phi = geom.dihedral_angle(a, b, c, d)
    assert float(phi) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------- #
# Bounded polynomial                                                     #
# --------------------------------------------------------------------- #


def test_bound_poly_clamping():
    """``bound_poly`` clamps below xlo to ylo and above xhi to yhi."""
    poly_mod = _import_or_skip("jtmol.score.common.polynomial")

    # Coefficients in descending order for ``f(x) = x^2 + 1`` — over the
    # support [0, 2], it ranges from 1 to 5.
    coeffs = jnp.asarray([1.0, 0.0, 1.0])
    xrange = (jnp.float32(0.0), jnp.float32(2.0))
    ybound = (jnp.float32(-99.0), jnp.float32(99.0))

    # Below the low bound -> y_lo sentinel.
    below = poly_mod.bound_poly(jnp.float32(-1.0), coeffs, xrange, ybound)
    assert below.shape == ()
    assert float(below) == pytest.approx(-99.0)

    # Above the high bound -> y_hi sentinel.
    above = poly_mod.bound_poly(jnp.float32(5.0), coeffs, xrange, ybound)
    assert float(above) == pytest.approx(99.0)

    # Inside the support -> evaluated polynomial.
    inside = poly_mod.bound_poly(jnp.float32(1.0), coeffs, xrange, ybound)
    assert float(inside) == pytest.approx(2.0, rel=1e-5)

    # Array input broadcasts.
    xs = jnp.asarray([-2.0, 0.5, 1.5, 3.0], dtype=jnp.float32)
    ys = poly_mod.bound_poly(xs, coeffs, xrange, ybound)
    assert ys.shape == (4,)
    # x=-2 -> ylo ; x=3 -> yhi ; interior -> x^2 + 1.
    assert float(ys[0]) == pytest.approx(-99.0)
    assert float(ys[1]) == pytest.approx(1.25, rel=1e-5)
    assert float(ys[2]) == pytest.approx(3.25, rel=1e-5)
    assert float(ys[3]) == pytest.approx(99.0)


# --------------------------------------------------------------------- #
# Cubic Hermite                                                          #
# --------------------------------------------------------------------- #


def test_cubic_hermite_endpoints():
    """At the endpoints, ``interpolate`` must reproduce ``(y0, y1)`` and
    ``jax.grad`` must reproduce ``(dy0, dy1)``.

    This guards against a regression in the ``dp0 = dy0 * dx`` rescale.
    """
    ch = _import_or_skip("jtmol.score.common.cubic_hermite_polynomial")

    x0 = jnp.float32(1.0)
    y0 = jnp.float32(2.0)
    dy0 = jnp.float32(-3.0)
    x1 = jnp.float32(4.0)
    y1 = jnp.float32(5.0)
    dy1 = jnp.float32(1.5)

    # Value at the endpoints.
    v0 = ch.interpolate(x0, x0, y0, dy0, x1, y1, dy1)
    v1 = ch.interpolate(x1, x0, y0, dy0, x1, y1, dy1)
    assert v0.shape == ()
    assert float(v0) == pytest.approx(float(y0), abs=1e-5)
    assert float(v1) == pytest.approx(float(y1), abs=1e-5)

    # Derivatives via autograd (dxs along the x dimension).
    def f(x):
        return ch.interpolate(x, x0, y0, dy0, x1, y1, dy1)

    g0 = jax.grad(f)(x0)
    g1 = jax.grad(f)(x1)
    assert float(g0) == pytest.approx(float(dy0), abs=1e-4)
    assert float(g1) == pytest.approx(float(dy1), abs=1e-4)

    # ``interpolate_to_zero`` endpoint: both y and dy are 0 at x1.
    v0z = ch.interpolate_to_zero(x0, x0, y0, dy0, x1)
    v1z = ch.interpolate_to_zero(x1, x0, y0, dy0, x1)
    assert float(v0z) == pytest.approx(float(y0), abs=1e-5)
    assert float(v1z) == pytest.approx(0.0, abs=1e-5)

    def fz(x):
        return ch.interpolate_to_zero(x, x0, y0, dy0, x1)

    g0z = jax.grad(fz)(x0)
    g1z = jax.grad(fz)(x1)
    assert float(g0z) == pytest.approx(float(dy0), abs=1e-4)
    assert float(g1z) == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------- #
# Count-pair connectivity weight                                         #
# --------------------------------------------------------------------- #


def test_connectivity_weight():
    """Upstream convention (``count_pair.hh``): path length <4 -> 0,
    ==4 -> 0.2, >4 -> 1.0.

    Sampled here for 0..5, and 10 for a far-apart pair.
    """
    cp = _import_or_skip("jtmol.score.common.count_pair")

    bpls = jnp.asarray([0, 1, 2, 3, 4, 5, 10], dtype=jnp.int32)
    w = cp.connectivity_weight(bpls)

    assert w.shape == (7,)
    # 0..3 are excluded (1-1 through 1-3): weight 0.
    for i in range(4):
        assert float(w[i]) == pytest.approx(0.0, abs=1e-6), (
            f"weight at path_length={int(bpls[i])} should be 0 (excluded)"
        )
    # path_length == 4 is the 1-4 attenuated case.
    assert float(w[4]) == pytest.approx(0.2, abs=1e-6)
    # path_length > 4 is the full-strength case.
    assert float(w[5]) == pytest.approx(1.0, abs=1e-6)
    assert float(w[6]) == pytest.approx(1.0, abs=1e-6)

    # Scalar input also works.
    assert float(cp.connectivity_weight(jnp.int32(4))) == pytest.approx(0.2)
    assert float(cp.connectivity_weight(jnp.int32(5))) == pytest.approx(1.0)
    assert float(cp.connectivity_weight(jnp.int32(0))) == pytest.approx(0.0)
