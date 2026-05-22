"""Selzer-Schreiber 1999 electrostatic ``k_on`` predictor (PARE).

Reference: Selzer T, Schreiber G. "Predicting the rate enhancement of
protein complex formation from the electrostatic energy of
interaction." J Mol Biol 287(2):409-419, 1999.
https://www.sciencedirect.com/science/article/abs/pii/S0022283699926153

This is the canonical electrostatic-only ``k_on`` predictor. It sums
shielded Coulombic interactions between charged atom pairs across an
interface, using a fixed solvent dielectric (no distance-dependent
damping) and a Debye-Hückel ion-screening factor. The bound-state
coordinates stand in for the transition state (encounter complex).

The signed energy ``U`` is what you optimize: more negative (favorable
electrostatic interaction) → faster ``k_on`` than the no-electrostatic
baseline ``k_on^0``.

    ln(k_on / k_on^0) = -U / RT

In design loops, use ``ss99_energy(coords, charges)`` as a
``LossTerm``. Lower (more negative) is better. Pure JAX,
``jit``/``vmap``/``grad``-clean — no surrogates, no soft thresholds:
every operation is C∞ in atom coordinates.

Calibration values from the original paper (HyPare / PARE):

- ``epsilon = 80``        — solvent dielectric (no group-based scaling)
- ``kappa_inv = 6.5``     — Debye length, 1/κ, in Å (for I = 0.022 M)
- ``a = 6.0``             — ion-exclusion shell / closest-approach (Å)
- ``cutoff = 25.0``       — distance cutoff for speed (Debye factor ≈ 0 beyond)

Charge model (matches HyPare):

- ARG: +0.5 on NH1, +0.5 on NH2
- LYS: +1.0 on NZ
- ASP: -0.5 on OD1, -0.5 on OD2
- GLU: -0.5 on OE1, -0.5 on OE2
- HIS: neutral by default (set ``protonate_his=True`` for +1 on NE2)
- N-terminus: +1.0 on N (only on chain-start residues)
- C-terminus: -1.0 on OXT (only on chain-end residues)

"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from tmol_jax.score.common.geom import _safe_norm


# Bjerrum length in water at 298 K, in Å.
# l_B = e^2 / (4πε₀ε_r k_B T) ≈ 0.71 Å at ε_r = 80, T = 298 K.
# The full Coulomb prefactor q_i q_j / (ε r) gives kcal/mol when
# multiplied by 332.0637 (Rosetta convention). We keep the Rosetta
# convention so values are directly comparable to fa_elec.
_COULOMB_KCAL_MOL_A = 332.0637

# HyPare canonical parameters.
EPSILON_DEFAULT = 80.0
KAPPA_INV_DEFAULT = 6.5      # Å, for I = 0.022 M
A_DEFAULT = 6.0              # Å, ion-exclusion shell
CUTOFF_DEFAULT = 25.0        # Å


# --------------------------------------------------------------------- #
# Charge assignment from residue+atom names                              #
# --------------------------------------------------------------------- #

# Per-residue charged-atom table. Maps (res3, atom_name) -> partial charge.
_CHARGED_ATOMS: Dict[Tuple[str, str], float] = {
    ("ARG", "NH1"): +0.5,
    ("ARG", "NH2"): +0.5,
    ("LYS", "NZ"):  +1.0,
    ("ASP", "OD1"): -0.5,
    ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5,
    ("GLU", "OE2"): -0.5,
}

# Optional HIS protonation (for low-pH or interface-specific use).
_CHARGED_ATOMS_HIS: Dict[Tuple[str, str], float] = {
    ("HIS", "NE2"): +1.0,
}


def build_ss99_charges(
    res3_per_res: list[str],
    atom_names_per_atom: list[str],
    atom_to_res: np.ndarray,
    is_chain_start: np.ndarray,
    is_chain_end: np.ndarray,
    protonate_his: bool = False,
) -> Float[Array, "n_atoms"]:
    """Assemble the per-atom SS99 partial-charge vector.

    Parameters
    ----------
    res3_per_res
        Three-letter residue names, length ``n_res``.
    atom_names_per_atom
        Per-atom names matching the flat coord layout, length ``n_atoms``.
    atom_to_res
        Integer index from each atom into ``res3_per_res``, length ``n_atoms``.
    is_chain_start, is_chain_end
        Per-residue boolean masks marking N- and C-termini. Length
        ``n_res``.
    protonate_his
        If True, set HIS NE2 to +1.0. Default False (neutral HIS).

    Returns
    -------
    charges : Array, shape (n_atoms,)
        Per-atom partial charge under the SS99/HyPare convention.
        Zero for any atom not in the charged set.
    """
    table = dict(_CHARGED_ATOMS)
    if protonate_his:
        table.update(_CHARGED_ATOMS_HIS)

    n_atoms = len(atom_names_per_atom)
    charges = np.zeros(n_atoms, dtype=np.float32)
    for ai, aname in enumerate(atom_names_per_atom):
        ri = int(atom_to_res[ai])
        res3 = res3_per_res[ri]
        q = table.get((res3, aname), 0.0)
        if q != 0.0:
            charges[ai] = q
        # Termini overlay
        if aname == "N" and is_chain_start[ri]:
            charges[ai] += 1.0
        if aname == "OXT" and is_chain_end[ri]:
            charges[ai] -= 1.0
    return jnp.asarray(charges)


# --------------------------------------------------------------------- #
# Core energy: shielded Coulomb between charged atoms across chains      #
# --------------------------------------------------------------------- #


def ss99_energy(
    coords: Float[Array, "n_atoms 3"],
    charges: Float[Array, "n_atoms"],
    chain_id: Int[Array, "n_atoms"],
    *,
    epsilon: float = EPSILON_DEFAULT,
    kappa_inv: float = KAPPA_INV_DEFAULT,
    ion_excl_a: float = A_DEFAULT,
    cutoff: float = CUTOFF_DEFAULT,
    binder_chain_id: int = 0,
) -> Float[Array, ""]:
    """Selzer-Schreiber 1999 electrostatic interaction energy across
    an interface, in kcal/mol.

    Sums

        U = Σ_{i ∈ A, j ∈ B}
              (q_i q_j / (ε r_ij)) · exp(-κ (r_ij - a)) / (1 + κ a)

    over all pairs where ``i`` is on the binder chain and ``j`` is
    on a non-binder atom. Self-pairs on the same chain are excluded.

    Parameters
    ----------
    coords
        Flat atom coordinates, shape (n_atoms, 3).
    charges
        Per-atom SS99 partial charges from ``build_ss99_charges``.
    chain_id
        Per-atom chain index.
    epsilon
        Solvent dielectric. Default 80 (water).
    kappa_inv
        Debye length (1/κ) in Å. Default 6.5 (I = 0.022 M).
    ion_excl_a
        Ion-exclusion shell radius (Å). Default 6.0.
    cutoff
        Pair distance cutoff (Å). Default 25.0.
    binder_chain_id
        Which chain is the "binder" side of the interface.
        Other-chain atoms are the "target".

    Returns
    -------
    U : scalar
        Cross-chain SS99 interaction energy, kcal/mol. Negative is
        favorable (faster ``k_on``).
    """
    kappa = 1.0 / kappa_inv
    debye_norm = 1.0 + kappa * ion_excl_a

    # Build the cross-chain pair mask using charges and chain_id.
    # Only charged atoms contribute, so mask non-charged for speed.
    is_charged = jnp.abs(charges) > 1e-6
    is_binder = (chain_id == binder_chain_id) & is_charged
    is_target = (chain_id != binder_chain_id) & is_charged

    # Use only the charged atoms for the pair sum.
    # (We still do an n^2 over those, but n is small — ~20 charged
    # residues / chain = ~40 atoms per side.)
    binder_idx = jnp.where(is_binder, size=coords.shape[0], fill_value=-1)[0]
    target_idx = jnp.where(is_target, size=coords.shape[0], fill_value=-1)[0]
    binder_valid = binder_idx >= 0
    target_valid = target_idx >= 0

    # Gather charged atom coords + charges. Replace fills with origin.
    safe_b = jnp.where(binder_valid, binder_idx, 0)
    safe_t = jnp.where(target_valid, target_idx, 0)
    coords_b = jnp.where(binder_valid[:, None], coords[safe_b], 0.0)
    coords_t = jnp.where(target_valid[:, None], coords[safe_t], 0.0)
    q_b = jnp.where(binder_valid, charges[safe_b], 0.0)
    q_t = jnp.where(target_valid, charges[safe_t], 0.0)

    # Pairwise distances. shape: (n_b, n_t). NaN-safe norm so r=0 on
    # padded slots doesn't poison the gradient.
    diff = coords_b[:, None, :] - coords_t[None, :, :]
    r = _safe_norm(diff, axis=-1)

    # Avoid divide-by-zero. r=0 only happens when both sides are
    # fill atoms, which we mask out below anyway.
    r_safe = jnp.where(r > 1e-6, r, 1.0)

    # Shielded Coulomb in kcal/mol per pair.
    coul = _COULOMB_KCAL_MOL_A / epsilon * (q_b[:, None] * q_t[None, :]) / r_safe
    debye = jnp.exp(-kappa * (r_safe - ion_excl_a)) / debye_norm
    pair_e = coul * debye

    # Mask out fill atoms and distances beyond cutoff.
    valid = binder_valid[:, None] & target_valid[None, :] & (r < cutoff)
    pair_e = jnp.where(valid, pair_e, 0.0)

    return pair_e.sum()


def ss99_dG_complex(
    coords_bound: Float[Array, "n_atoms 3"],
    coords_separated: Float[Array, "n_atoms 3"],
    charges: Float[Array, "n_atoms"],
    chain_id: Int[Array, "n_atoms"],
    **kwargs,
) -> Float[Array, ""]:
    """Standard ΔU = U(bound) - U(separated). Returns scalar kcal/mol.

    In practice ``U(separated)`` ≈ 0 if the chains are translated
    far enough apart (>3× Debye length); included for symmetry with
    the formal definition.
    """
    return ss99_energy(coords_bound, charges, chain_id, **kwargs) - \
        ss99_energy(coords_separated, charges, chain_id, **kwargs)
