"""``LossTerm`` wrappers for the extra interface scores.

Each wrapper exposes the same protocol as ``mosaic.common.LossTerm``
so the new scores compose with mosaic / Boltz-2 hallucination losses
via ``+`` / ``*``:

    loss = 1.0 * TmolInterfaceLoss(ps_tpl, scale=1e-2) \\
         + 0.5 * SS99Loss(ps_tpl, charges, chain_id_atoms) \\
         + 1.0 * BUHLoss(ps_tpl, polar_info) \\
         + 0.1 * TransCompLoss(ps_tpl, charges, chain_id_atoms)

The wrappers all assume the same atom37-coords input convention as
``TmolInterfaceLoss``: ``output.structure_coordinates`` is the Boltz-2
all-atom output, and an internal ``flat_from_atom37`` mapping
(prepared by ``build_pose_stack_template``) is applied to lift atom37
back to the tmol_jax flat-layout coords the score functions expect.

For the smoke-test / standalone-eval case (no mosaic installed), the
``LossTerm`` base falls back to the same stand-in algebra as
``TmolInterfaceLoss`` (see
``examples/integration_mosaic_hallucinator.py``).
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Reuse the LossTerm shim from the existing mosaic integration so the
# composition algebra and the atom37→flat plumbing both stay in one
# place.
try:
    from examples.integration_mosaic_hallucinator import (
        _BaseLoss as _LossTerm,
    )
except Exception:  # pragma: no cover
    import equinox as eqx

    class _LossTerm(eqx.Module):
        def __call__(self, *args, key, **kwargs):
            raise NotImplementedError

        def __rmul__(self, s):
            if not isinstance(s, (int, float)):
                return NotImplemented
            return _ScaledLoss(loss=self, scale=float(s))

        def __add__(self, other):
            return _SumLoss(losses=[self, other])

    class _ScaledLoss(eqx.Module):
        loss: Any
        scale: float

        def __call__(self, *args, key, **kwargs):
            v, aux = self.loss(*args, key=key, **kwargs)
            return self.scale * v, aux

    class _SumLoss(eqx.Module):
        losses: list

        def __call__(self, *args, key, **kwargs):
            r = 0.0
            auxes = []
            for L in self.losses:
                v, a = L(*args, key=key, **kwargs)
                key = jax.random.fold_in(key, 1)
                r = r + v
                auxes.append(a)
            return r, auxes


from tmol_jax.score.extra.ss99 import ss99_energy
from tmol_jax.score.extra.buh import buh_score, buh_per_kA2
from tmol_jax.score.extra.prodigy import prodigy_dG
from tmol_jax.score.extra.transcomp import transcomp_log_kon_enhancement
from tmol_jax.score.extra.dfire import dfire_energy


def _atom37_to_flat(pose_stack: Any, atom37: Float[Array, "N 37 3"]) -> Float[Array, "M 3"]:
    """Lift atom37 coords back to the flat tmol_jax layout the score
    functions expect, using the precomputed ``_atom37_map`` /
    ``_atom37_is_mapped`` mapping on ``pose_stack``.
    """
    coords_b = pose_stack.coords
    flat = atom37.reshape(-1, 3)[pose_stack._atom37_map]
    total = flat.shape[0]
    return coords_b.at[0, :total].set(
        jnp.where(
            pose_stack._atom37_is_mapped[:, None],
            flat,
            coords_b[0, :total],
        )
    )[0, :total]


# ===================================================================== #
# SS99 / TransComp electrostatic kon                                     #
# ===================================================================== #


class SS99Loss(_LossTerm):
    """Selzer-Schreiber 1999 electrostatic ``k_on`` proxy as a
    ``LossTerm``. Lower is better.

    Parameters
    ----------
    pose_stack_template
        Pre-built PoseStack from ``build_pose_stack_template``.
    charges
        Per-atom SS99 charges. Build via ``build_ss99_charges``.
    chain_id_atoms
        Per-atom chain index.
    scale
        Global scaling factor. Default ``1e-2`` since SS99 energies
        are typically a few kcal/mol.
    binder_chain_id
        Which chain is the binder side. Default 0.
    """

    pose_stack: Any
    charges: Float[Array, "n_atoms"]
    chain_id_atoms: Int[Array, "n_atoms"]
    scale: float
    binder_chain_id: int

    def __init__(self, pose_stack_template, charges, chain_id_atoms,
                 scale: float = 1e-2, binder_chain_id: int = 0):
        self.pose_stack = pose_stack_template
        self.charges = charges
        self.chain_id_atoms = chain_id_atoms
        self.scale = scale
        self.binder_chain_id = binder_chain_id

    def __call__(self, sequence, output, key):
        atom37 = output.structure_coordinates
        coords = _atom37_to_flat(self.pose_stack, atom37)
        U = ss99_energy(coords, self.charges, self.chain_id_atoms,
                        binder_chain_id=self.binder_chain_id)
        return self.scale * U, {"U_SS99": U}


class TransCompLoss(_LossTerm):
    """Qin-Zhou 2011 transition-state-theory ``k_on`` enhancement.
    Returns the *negative* log-ratio so that lower-is-better matches
    other loss terms — minimizing this loss maximizes ``k_on``
    enhancement.
    """

    pose_stack: Any
    charges: Float[Array, "n_atoms"]
    chain_id_atoms: Int[Array, "n_atoms"]
    scale: float
    binder_chain_id: int
    kappa_inv: float

    def __init__(self, pose_stack_template, charges, chain_id_atoms,
                 scale: float = 1.0, binder_chain_id: int = 0,
                 kappa_inv: float = 8.0):
        self.pose_stack = pose_stack_template
        self.charges = charges
        self.chain_id_atoms = chain_id_atoms
        self.scale = scale
        self.binder_chain_id = binder_chain_id
        self.kappa_inv = kappa_inv

    def __call__(self, sequence, output, key):
        atom37 = output.structure_coordinates
        coords = _atom37_to_flat(self.pose_stack, atom37)
        ln_ratio = transcomp_log_kon_enhancement(
            coords, self.charges, self.chain_id_atoms,
            binder_chain_id=self.binder_chain_id,
            kappa_inv=self.kappa_inv,
        )
        # Loss = -ln(k_on / k_on0). Minimizing the loss maximizes
        # k_on enhancement.
        return -self.scale * ln_ratio, {"ln_kon_ratio": ln_ratio}


# ===================================================================== #
# BUH                                                                     #
# ===================================================================== #


class BUHLoss(_LossTerm):
    """Stranges-Kuhlman 2013 buried-unsatisfied-polars surrogate.
    Lower is better.

    Uses the coords-only Hbond satisfaction path. Pass per-polar-atom
    indicator arrays (``polar_is_donor``, ``polar_is_acceptor``) so
    the gradient flows end-to-end through coords.

    If ``interface_bsa`` is provided, the loss is normalized to
    "BUH per kÅ²" (Stranges-Kuhlman 2013 convention, ≤ 2 = good).
    """

    pose_stack: Any
    polar_atom_idx: Int[Array, "n_polar"]
    polar_atom_res: Int[Array, "n_polar"]
    polar_capacity: Float[Array, "n_polar"]
    polar_is_donor: Float[Array, "n_polar"]
    polar_is_acceptor: Float[Array, "n_polar"]
    res_ca_idx: Int[Array, "n_res"]
    res_cb_idx: Int[Array, "n_res"]
    atom_to_res: Int[Array, "n_atoms"]
    scale: float
    interface_bsa: Optional[Float[Array, ""]]

    def __init__(self, pose_stack_template, polar_info, res_ca_idx, res_cb_idx,
                 atom_to_res, scale: float = 0.1,
                 interface_bsa: Optional[Float[Array, ""]] = None):
        """``polar_info`` is a dict with keys ``idx``, ``res``,
        ``capacity``, ``is_donor``, ``is_acceptor``.
        """
        self.pose_stack = pose_stack_template
        self.polar_atom_idx = polar_info["idx"]
        self.polar_atom_res = polar_info["res"]
        self.polar_capacity = polar_info["capacity"]
        self.polar_is_donor = polar_info["is_donor"]
        self.polar_is_acceptor = polar_info["is_acceptor"]
        self.res_ca_idx = res_ca_idx
        self.res_cb_idx = res_cb_idx
        self.atom_to_res = atom_to_res
        self.scale = scale
        self.interface_bsa = interface_bsa

    def __call__(self, sequence, output, key):
        atom37 = output.structure_coordinates
        coords = _atom37_to_flat(self.pose_stack, atom37)
        BUH = buh_score(
            coords, self.polar_atom_idx, self.polar_atom_res,
            self.polar_capacity,
            jnp.zeros_like(self.polar_capacity),     # ignored in coords-only path
            self.res_ca_idx, self.res_cb_idx, self.atom_to_res,
            polar_is_donor=self.polar_is_donor,
            polar_is_acceptor=self.polar_is_acceptor,
        )
        if self.interface_bsa is not None:
            v = buh_per_kA2(BUH, self.interface_bsa)
        else:
            v = BUH
        return self.scale * v, {"BUH_raw": BUH, "BUH_loss": v}


# ===================================================================== #
# PRODIGY                                                                 #
# ===================================================================== #


class PRODIGYLoss(_LossTerm):
    """Vangone-Bonvin 2015 contact-based ΔG. Lower is better."""

    pose_stack: Any
    atom_to_res: Int[Array, "n_atoms"]
    res_polarity: Int[Array, "n_res"]
    chain_id_per_res: Int[Array, "n_res"]
    nis_apolar: float
    nis_charged: float
    scale: float
    binder_chain_id: int

    def __init__(self, pose_stack_template, atom_to_res, res_polarity,
                 chain_id_per_res, nis_apolar: float = 0.0,
                 nis_charged: float = 0.0, scale: float = 1e-2,
                 binder_chain_id: int = 0):
        self.pose_stack = pose_stack_template
        self.atom_to_res = atom_to_res
        self.res_polarity = res_polarity
        self.chain_id_per_res = chain_id_per_res
        self.nis_apolar = nis_apolar
        self.nis_charged = nis_charged
        self.scale = scale
        self.binder_chain_id = binder_chain_id

    def __call__(self, sequence, output, key):
        atom37 = output.structure_coordinates
        coords = _atom37_to_flat(self.pose_stack, atom37)
        dG = prodigy_dG(
            coords, self.atom_to_res, self.res_polarity,
            self.chain_id_per_res,
            nis_apolar=jnp.asarray(self.nis_apolar),
            nis_charged=jnp.asarray(self.nis_charged),
            binder_chain_id=self.binder_chain_id,
        )
        return self.scale * dG, {"dG_PRODIGY": dG}


# ===================================================================== #
# DFIRE                                                                   #
# ===================================================================== #


class DFIRELoss(_LossTerm):
    """Zhang-Zhou 2005 atom-pair statistical potential. Lower is
    better. Needs a precomputed potential table (call
    ``make_placeholder_dfire_table()`` for infrastructure testing or
    load the published Zhou-Yang coefficients via the conversion
    script).
    """

    pose_stack: Any
    atom_types: Int[Array, "n_atoms"]
    chain_id_atoms: Int[Array, "n_atoms"]
    potential_table: Float[Array, "n_types n_types n_bins"]
    scale: float
    binder_chain_id: int
    cross_chain_only: bool

    def __init__(self, pose_stack_template, atom_types, chain_id_atoms,
                 potential_table, scale: float = 1e-2,
                 binder_chain_id: int = 0, cross_chain_only: bool = True):
        self.pose_stack = pose_stack_template
        self.atom_types = atom_types
        self.chain_id_atoms = chain_id_atoms
        self.potential_table = potential_table
        self.scale = scale
        self.binder_chain_id = binder_chain_id
        self.cross_chain_only = cross_chain_only

    def __call__(self, sequence, output, key):
        atom37 = output.structure_coordinates
        coords = _atom37_to_flat(self.pose_stack, atom37)
        E = dfire_energy(
            coords, self.atom_types, self.chain_id_atoms,
            self.potential_table,
            cross_chain_only=self.cross_chain_only,
            binder_chain_id=self.binder_chain_id,
        )
        return self.scale * E, {"E_DFIRE": E}


__all__ = [
    "SS99Loss",
    "TransCompLoss",
    "BUHLoss",
    "PRODIGYLoss",
    "DFIRELoss",
]
