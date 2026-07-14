"""jtmol.relax — FastRelax protocol, JAX/Python port.

Ported from ``tmol.relax.fast_relax``. The upstream torch version is built
around the full MonomerRelax2019 schedule (four pack/min steps with a
tuned split between pack and min fa_rep fractions, plus a ramped
constraint weight). This JAX port keeps the same public entry point —
:func:`fast_relax` — but exposes a simpler default schedule
(``[0.02, 0.25, 0.55, 1.0]``) that matches the "classic" FastRelax ramp
most external callers expect.

The protocol is a pure Python loop around shared JAX modules:

    :func:`jtmol.pack.packer.pack_rotamers` — rotamer repacking
    :func:`jtmol.optimization.minimizers.run_min` /
    :func:`jtmol.optimization.minimizers.run_cart_min` — LBFGS minimization
    :func:`jtmol.score.constraint.utility.create_mainchain_coordinate_constraints`
        — adds CA/N/C/O harmonic coord constraints to keep the backbone
        near its starting position (``bb_cst_sigma`` defaults to 0.5 Å).

See :func:`fast_relax` and :func:`fast_relax_no_pack` for the two entry
points.
"""

from __future__ import annotations

from jtmol.relax.fast_relax import (
    DEFAULT_RAMP_SCHEDULE,
    fast_relax,
    fast_relax_no_pack,
)

__all__ = [
    "DEFAULT_RAMP_SCHEDULE",
    "fast_relax",
    "fast_relax_no_pack",
]
