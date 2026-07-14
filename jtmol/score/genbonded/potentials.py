"""GenBonded torsion / improper potentials (pure JAX).

Energy forms (verbatim from upstream ``tmol/score/genbonded/potentials/potentials.hh``):

* **Proper torsion** (bare-cosine sum, 4 periods, single ``offset``, with
  negative-K lift to keep each per-period term non-negative)::

      E = sum_{n=1..4} k_n * (1 + cos(n * theta))
        + sum_{n: k_n < 0} (-2 * k_n)
        + offset

  The header doc in ``potentials.hh`` describes ``cos(n*theta - phi_n)``
  but the actual C++ ignores the per-period phase (currently all-zero in
  the DB anyway). The ``offset`` is an additive constant, not a phase.

* **Improper torsion** (harmonic in the dihedral angle)::

      E = k * (theta - delta)^2

Derivatives come from :func:`jax.grad` over these value functions —
the upstream ``_V_dV`` helpers bake the chain rule through
``dihedral_angle`` into C++; we rely on JAX autodiff to do the same.
"""

from __future__ import annotations

import jax.numpy as jnp

from jtmol.types import Array


def gen_proper_E(
    theta: Array,
    k1: Array,
    k2: Array,
    k3: Array,
    k4: Array,
    offset: Array,
) -> Array:
    """Bare-cosine 4-period proper torsion energy with negative-K lift + offset.

    All inputs broadcast together. ``theta`` is in radians.
    """
    e = (
        k1 * (1.0 + jnp.cos(1.0 * theta))
        + k2 * (1.0 + jnp.cos(2.0 * theta))
        + k3 * (1.0 + jnp.cos(3.0 * theta))
        + k4 * (1.0 + jnp.cos(4.0 * theta))
    )
    # Negative-K lift: any term with k_n < 0 has range [2*k_n, 0]; add
    # -2*k_n to lift it to [0, -2*k_n]. ``jnp.where`` is the branchless
    # equivalent of the C++ ``if (k_n < 0) E += -2*k_n``.
    e = e + jnp.where(k1 < 0.0, -2.0 * k1, 0.0)
    e = e + jnp.where(k2 < 0.0, -2.0 * k2, 0.0)
    e = e + jnp.where(k3 < 0.0, -2.0 * k3, 0.0)
    e = e + jnp.where(k4 < 0.0, -2.0 * k4, 0.0)
    return e + offset


def gen_improper_E(theta: Array, k: Array, delta: Array) -> Array:
    """Harmonic improper torsion: ``E = k * (theta - delta)^2``."""
    return k * jnp.square(theta - delta)


__all__ = ["gen_proper_E", "gen_improper_E"]
