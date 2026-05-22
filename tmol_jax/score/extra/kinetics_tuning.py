"""Kinetics-tuning composite loss: separate gradient channels for
``k_on``, ``k_off``, and ``K_d``.

The standard ``TmolInterfaceLoss`` gives you a thermodynamic ``K_d``
gradient via the bound-minus-separated energy difference. That single
scalar doesn't tell you whether to optimize for ``k_on`` (electrostatic
steering, encounter complex) or ``k_off`` (long-residence Hbond ladder,
no buried-unsat polars) — both contribute to ``K_d`` but they're
addressable by different mutations.

``KineticsTuningLoss`` exposes three separate weighted channels:

* ``w_kon`` — Selzer-Schreiber / TransComp electrostatic ``k_on`` proxy.
  Minimizing the loss pushes the design toward more favorable
  electrostatic steering (faster ``k_on``).
* ``w_koff`` — Stranges-Kuhlman BUH (buried-unsatisfied polars), the
  coords-only path. Minimizing pushes toward zero buried-unsat polars
  (slower ``k_off`` — interfaces with all polars satisfied dissociate
  more slowly).
* ``w_kd`` — Rosetta beta_nov2016_cart interface dG_separated. The
  thermodynamic ``K_d`` proxy.

Minimizing the composite loss improves each kinetic property the
corresponding weight emphasizes. Set unused weights to zero to tune
a single axis at a time.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from tmol_jax.score.extra.ss99 import ss99_energy
from tmol_jax.score.extra.buh import buh_score


class KineticsTuningLoss:
    """Composite loss with separate gradient channels for kon, koff, Kd.

    This is a plain dataclass-style holder, not a ``LossTerm``
    subclass — drop into your own gradient loop or wrap in a
    ``LossTerm`` for mosaic. The ``__call__`` returns
    ``(scalar, aux_dict)`` so you can read off each component for
    diagnostics.

    Sign convention (consistent with minimization):
      * Lower ``U_SS99`` = more favorable electrostatics = faster kon
        → multiplied by ``+w_kon``.
      * Lower BUH = fewer buried-unsat polars = slower koff
        → multiplied by ``+w_koff``.
      * Lower interface dG = tighter binding = lower Kd
        → multiplied by ``+w_kd``.
    """

    def __init__(self, *, w_kon: float = 1.0, w_koff: float = 1.0,
                 w_kd: float = 1.0,
                 # SS99 inputs
                 charges: Optional[Float[Array, "n_atoms"]] = None,
                 chain_id_atoms: Optional[Int[Array, "n_atoms"]] = None,
                 binder_chain_id: int = 0,
                 # BUH inputs
                 polar_atom_idx: Optional[Int[Array, "n_polar"]] = None,
                 polar_atom_res: Optional[Int[Array, "n_polar"]] = None,
                 polar_capacity: Optional[Float[Array, "n_polar"]] = None,
                 polar_is_donor: Optional[Float[Array, "n_polar"]] = None,
                 polar_is_acceptor: Optional[Float[Array, "n_polar"]] = None,
                 res_ca_idx: Optional[Int[Array, "n_res"]] = None,
                 res_cb_idx: Optional[Int[Array, "n_res"]] = None,
                 atom_to_res: Optional[Int[Array, "n_atoms"]] = None,
                 # Kd: callable returning dG_interface given coords
                 kd_fn: Optional[Any] = None,
                 # Numerical scaling
                 scale_kon: float = 1.0,
                 scale_koff: float = 0.1,
                 scale_kd: float = 1e-3):
        self.w_kon = w_kon
        self.w_koff = w_koff
        self.w_kd = w_kd
        self.charges = charges
        self.chain_id_atoms = chain_id_atoms
        self.binder_chain_id = binder_chain_id
        self.polar_atom_idx = polar_atom_idx
        self.polar_atom_res = polar_atom_res
        self.polar_capacity = polar_capacity
        self.polar_is_donor = polar_is_donor
        self.polar_is_acceptor = polar_is_acceptor
        self.res_ca_idx = res_ca_idx
        self.res_cb_idx = res_cb_idx
        self.atom_to_res = atom_to_res
        self.kd_fn = kd_fn
        self.scale_kon = scale_kon
        self.scale_koff = scale_koff
        self.scale_kd = scale_kd

    def components(self, coords):
        """Return ``(U_SS99, BUH, dG_Kd)`` at ``coords`` — the three
        un-weighted physical channels. Useful for diagnostics.
        """
        if self.w_kon > 0 and self.charges is not None:
            U_SS99 = ss99_energy(
                coords, self.charges, self.chain_id_atoms,
                binder_chain_id=self.binder_chain_id,
            )
        else:
            U_SS99 = jnp.asarray(0.0)

        if self.w_koff > 0 and self.polar_atom_idx is not None:
            BUH = buh_score(
                coords, self.polar_atom_idx, self.polar_atom_res,
                self.polar_capacity,
                jnp.zeros_like(self.polar_capacity),
                self.res_ca_idx, self.res_cb_idx, self.atom_to_res,
                polar_is_donor=self.polar_is_donor,
                polar_is_acceptor=self.polar_is_acceptor,
            )
        else:
            BUH = jnp.asarray(0.0)

        if self.w_kd > 0 and self.kd_fn is not None:
            dG_Kd = self.kd_fn(coords)
        else:
            dG_Kd = jnp.asarray(0.0)

        return U_SS99, BUH, dG_Kd

    def __call__(self, coords):
        U_SS99, BUH, dG_Kd = self.components(coords)
        loss = (
            self.w_kon  * self.scale_kon  * U_SS99   # min → faster kon
            + self.w_koff * self.scale_koff * BUH     # min → slower koff
            + self.w_kd   * self.scale_kd   * dG_Kd   # min → tighter Kd
        )
        return loss, {
            "U_SS99": U_SS99, "BUH": BUH, "dG_Kd": dG_Kd,
            "loss": loss,
        }


def per_residue_kinetic_gradient_norms(
    loss_obj: KineticsTuningLoss,
    coords: Float[Array, "n_atoms 3"],
    atom_to_res: np.ndarray,
    n_res: int,
):
    """Per-residue ``|∂loss/∂x|`` for each kinetic axis independently.

    Returns a dict of shape-``(n_res,)`` arrays:
        ``"kon"``  — per-residue gradient norm of the kon channel only
        ``"koff"`` — per-residue gradient norm of the koff channel only
        ``"kd"``   — per-residue gradient norm of the Kd channel only

    Use these to identify which residues are sensitive on EACH kinetic
    axis — the same residue can be a hot-spot for koff but irrelevant
    for kon, or vice versa.
    """
    def loss_kon(c):
        L = KineticsTuningLoss(
            w_kon=1.0, w_koff=0.0, w_kd=0.0,
            charges=loss_obj.charges,
            chain_id_atoms=loss_obj.chain_id_atoms,
            binder_chain_id=loss_obj.binder_chain_id,
            scale_kon=loss_obj.scale_kon,
        )
        v, _ = L(c)
        return v

    def loss_koff(c):
        L = KineticsTuningLoss(
            w_kon=0.0, w_koff=1.0, w_kd=0.0,
            polar_atom_idx=loss_obj.polar_atom_idx,
            polar_atom_res=loss_obj.polar_atom_res,
            polar_capacity=loss_obj.polar_capacity,
            polar_is_donor=loss_obj.polar_is_donor,
            polar_is_acceptor=loss_obj.polar_is_acceptor,
            res_ca_idx=loss_obj.res_ca_idx,
            res_cb_idx=loss_obj.res_cb_idx,
            atom_to_res=loss_obj.atom_to_res,
            scale_koff=loss_obj.scale_koff,
        )
        v, _ = L(c)
        return v

    def loss_kd(c):
        L = KineticsTuningLoss(
            w_kon=0.0, w_koff=0.0, w_kd=1.0,
            kd_fn=loss_obj.kd_fn,
            scale_kd=loss_obj.scale_kd,
        )
        v, _ = L(c)
        return v

    g_kon = np.asarray(jax.grad(loss_kon)(coords))
    g_koff = np.asarray(jax.grad(loss_koff)(coords))
    g_kd = np.asarray(jax.grad(loss_kd)(coords))

    out = {}
    for label, g in [("kon", g_kon), ("koff", g_koff), ("kd", g_kd)]:
        per_atom_norm = np.linalg.norm(g, axis=-1)
        per_res = np.zeros(n_res)
        for ai, ri in enumerate(atom_to_res):
            if 0 <= ri < n_res:
                per_res[ri] += per_atom_norm[ai] ** 2
        out[label] = np.sqrt(per_res)
    return out


__all__ = [
    "KineticsTuningLoss",
    "per_residue_kinetic_gradient_norms",
]
