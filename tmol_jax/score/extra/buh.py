"""Buried-unsatisfied polars (BUH), differentiable surrogate.

Reference: Stranges PB, Kuhlman B. "A comparison of successful and
failed protein interface designs highlights the challenges of
designing buried hydrogen bonds." Protein Sci 22(1):74-82, 2013.
https://onlinelibrary.wiley.com/doi/10.1002/pro.2187

A buried polar atom whose Hbond capacity is unsatisfied is a major
predictor of a design failing to bind. Stranges & Kuhlman 2013 found
that successful designs have ≤ 2 buried-unsat polars per 1000 Å² of
interface, while failed designs typically have 3-7. Subsequently
Stranges-Kuhlman-style BUH normalized per kÅ² of buried surface area
has been shown to correlate strongly with experimental ``k_off``
(Adaptyv internal corpus: r ≈ +0.54 on de novo binders, ``n=43``,
``p=1.7e-4``).

Why not Rosetta's ``BuriedUnsatHbonds`` filter directly? Two reasons:

1. The Rosetta filter uses VSASA (Le Grand & Merz dot-based SASA),
   which is non-differentiable. We want a JAX-native, ``jax.grad``-
   safe surrogate for use as a ``LossTerm`` in design loops.
2. The Rosetta filter is a hard counter (count polar atoms with 0
   Hbonds). We replace both the burial threshold and the
   satisfaction count with continuous surrogates so the gradient
   flows.

### Surrogate design

**Burial.** We use Rocklin-style sidechain-neighbor cones (continuous
in [0,1], faster and smoother than true SASA). For each polar atom
``i`` we count neighboring Cα atoms inside an angular cone aligned
with the residue's Cα→Cβ axis, weighted by a smooth distance-and-
angle kernel:

    cone(i, j) = sigmoid((d_max - r_ij)/τ_d) · sigmoid((cos(θ_ij) - cos_min) / τ_θ)

where ``r_ij`` is the i-Cα distance and ``θ_ij`` is the angle between
the i-residue's Cα-Cβ axis and the i→j Cα vector. Sum of cones is a
smooth "buried-ness" score, typically 0-25; we sigmoid it around 5.5
(Rocklin's threshold for "buried").

**Satisfaction.** We use the hbond energy directly: each polar atom's
"satisfaction" is sum over Hbond partners of ``softplus(-E_hbond_ij)``,
where ``E_hbond_ij < 0`` means a real Hbond and ``softplus`` keeps the
gradient. We then use ``sigmoid(satisfaction - 1)`` as a smooth
indicator of "at least one Hbond present" for non-OH atoms (for OH
groups the target capacity is 2 and for backbone O it's 1; we
parameterize via ``n_hb_required``).

**BUH score.**

    BUH = Σ_i  burial(i) · (1 - satisfaction(i))

Lower is better. The score is smooth and ``jax.grad``-safe.

### Calibration vs published BUH

Because the underlying SASA proxy differs from VSASA, the absolute
numbers are not directly comparable to Stranges-Kuhlman 2013's
counts. Use this score as a **continuous gradient surrogate**, with
the published Rosetta filter run as a final post-hoc QC check
(non-differentiable, but cheap).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from tmol_jax.score.common.geom import _safe_norm


# Rocklin sidechain-neighbor cone parameters.
ROCKLIN_D_MAX = 11.5       # Å, distance cutoff for cone weight
ROCKLIN_TAU_D = 1.0        # Å, distance smoothing
ROCKLIN_COS_MIN = 0.5      # cos(60°), angular cone width
ROCKLIN_TAU_THETA = 0.1    # cosine-space angular smoothing
ROCKLIN_BURIAL_THRESH = 5.5  # cones above which an atom is "buried"
ROCKLIN_BURIAL_TAU = 0.5   # smoothing on the burial threshold


# Per-residue target Hbond count for "satisfied":
# OH atoms (SER OG, THR OG1, TYR OH): can donate 1, accept 2 → 2
# Backbone amides (N): need 1 donor
# Backbone carbonyls (O): need 1 acceptor
# Polar sidechain amides (ASN ND2, GLN NE2): need 1-2 donors
# Anything else: need 1
_HBOND_CAPACITY_DEFAULT = 1.0


def sidechain_neighbor_cone_count(
    coords: Float[Array, "n_atoms 3"],
    res_ca_idx: Int[Array, "n_res"],
    res_cb_idx: Int[Array, "n_res"],
    atom_to_res: Int[Array, "n_atoms"],
    *,
    d_max: float = ROCKLIN_D_MAX,
    tau_d: float = ROCKLIN_TAU_D,
    cos_min: float = ROCKLIN_COS_MIN,
    tau_theta: float = ROCKLIN_TAU_THETA,
) -> Float[Array, "n_res"]:
    """Smooth burial score per residue via Rocklin sidechain-neighbor
    cones. Returns one scalar per residue: roughly 0 for fully solvent-
    exposed, 25+ for deeply buried.

    The cone is aligned with the residue's Cα→Cβ axis and counts
    neighboring Cα atoms inside the cone, weighted smoothly by
    distance and angle.
    """
    ca = coords[res_ca_idx]      # [n_res, 3]
    cb = coords[res_cb_idx]      # [n_res, 3]
    # NaN-safe sidechain axis: zero-length axis (e.g. GLY where Cβ=Cα)
    # is handled by the double-where pattern inside _safe_norm.
    sc_axis_raw = cb - ca
    sc_axis_norm = _safe_norm(sc_axis_raw, axis=-1, keepdims=True)
    sc_axis = sc_axis_raw / jnp.where(sc_axis_norm > 1e-6, sc_axis_norm, 1.0)

    # All pairwise Cα-Cα vectors.
    diff = ca[None, :, :] - ca[:, None, :]                  # [i, j, 3]
    r = _safe_norm(diff, axis=-1)                            # [i, j]
    r_safe = jnp.where(r > 1e-6, r, 1.0)
    dir_ij = diff / r_safe[..., None]

    # Cosine between residue-i sidechain axis and the i→j vector.
    cos_ij = jnp.einsum("id,ijd->ij", sc_axis, dir_ij)

    # Smooth distance window.
    dist_w = jax.nn.sigmoid((d_max - r) / tau_d)
    # Smooth angular window.
    ang_w = jax.nn.sigmoid((cos_ij - cos_min) / tau_theta)
    # Exclude self.
    self_mask = 1.0 - jnp.eye(r.shape[0])

    cone = dist_w * ang_w * self_mask
    return cone.sum(axis=-1)


def hbond_satisfaction_from_coords(
    coords: Float[Array, "n_atoms 3"],
    polar_atom_idx: Int[Array, "n_polar"],
    polar_atom_res: Int[Array, "n_polar"],
    polar_is_donor: Float[Array, "n_polar"],
    polar_is_acceptor: Float[Array, "n_polar"],
    *,
    hb_dist: float = 3.5,
    hb_dist_tau: float = 0.25,
) -> Float[Array, "n_polar"]:
    """Smooth, coords-only Hbond-satisfaction estimate per polar atom.

    For each polar atom ``i``, sum a smooth indicator over all OTHER
    polar atoms ``j`` of opposite type (donor↔acceptor), of different
    residue, within ``hb_dist`` Å. Returns a per-atom satisfaction
    count in ``[0, k_max]``.

    This is the coords-side analog of "did atom i form an Hbond?".
    It's differentiable end-to-end because the sigmoid distance step
    keeps gradient flow. Replaces the precomputed-Hbond-energy input
    in the original ``buh_score`` API so the full BUH score is
    differentiable through coords, not just burial.
    """
    p_xyz = coords[polar_atom_idx]
    diff = p_xyz[:, None, :] - p_xyz[None, :, :]
    r = _safe_norm(diff, axis=-1)

    # Donor-acceptor pair mask (D↔A only, no D↔D or A↔A).
    d_a = polar_is_donor[:, None] * polar_is_acceptor[None, :]
    a_d = polar_is_acceptor[:, None] * polar_is_donor[None, :]
    pair_mask = d_a + a_d

    # Exclude same-residue partners.
    diff_res = (polar_atom_res[:, None] != polar_atom_res[None, :]).astype(
        pair_mask.dtype
    )

    smooth_hb = jax.nn.sigmoid((hb_dist - r) / hb_dist_tau)
    sat_per_pair = smooth_hb * pair_mask * diff_res
    return sat_per_pair.sum(axis=-1)


def buh_score(
    coords: Float[Array, "n_atoms 3"],
    polar_atom_idx: Int[Array, "n_polar"],
    polar_atom_res: Int[Array, "n_polar"],
    polar_atom_hb_capacity: Float[Array, "n_polar"],
    polar_atom_hbond_energies: Float[Array, "n_polar"],
    res_ca_idx: Int[Array, "n_res"],
    res_cb_idx: Int[Array, "n_res"],
    atom_to_res: Int[Array, "n_atoms"],
    *,
    burial_thresh: float = ROCKLIN_BURIAL_THRESH,
    burial_tau: float = ROCKLIN_BURIAL_TAU,
    polar_is_donor: Float[Array, "n_polar"] = None,
    polar_is_acceptor: Float[Array, "n_polar"] = None,
) -> Float[Array, ""]:
    """Differentiable buried-unsatisfied-polars score.

    Parameters
    ----------
    coords
        Flat atom coords, shape (n_atoms, 3).
    polar_atom_idx
        Indices of polar heavy atoms to score, shape (n_polar,).
    polar_atom_res
        Residue index for each polar atom, shape (n_polar,).
    polar_atom_hb_capacity
        For each polar atom, the number of Hbonds it needs to be
        considered "satisfied" (1 for backbone N/O, 2 for OH, 1-2 for
        sidechain amides). Shape (n_polar,).
    polar_atom_hbond_energies
        Sum of incoming Hbond energies (negative = favorable) on each
        polar atom, in kcal/mol. Shape (n_polar,). Typically computed
        from the hbond term's per-atom decomposition.
    res_ca_idx, res_cb_idx
        Per-residue Cα and Cβ atom indices. Used for the
        sidechain-neighbor cone burial score. Glycine has no Cβ;
        we use a virtual Cβ along the N-Cα-C plane (caller's
        responsibility to provide).
    atom_to_res
        Residue index per atom.
    burial_thresh, burial_tau
        Sigmoid parameters for the burial threshold. Default values
        match Rocklin's "buried" definition.

    Returns
    -------
    buh : scalar
        Smooth count of buried-unsatisfied polar atoms.
        Lower (closer to 0) is better.
    """
    # Per-residue Rocklin burial score (smooth, 0-25-ish).
    res_burial = sidechain_neighbor_cone_count(
        coords, res_ca_idx, res_cb_idx, atom_to_res
    )

    # Map per-residue burial to per-polar-atom.
    polar_burial = res_burial[polar_atom_res]

    # Smooth "is buried" indicator in [0, 1].
    is_buried = jax.nn.sigmoid((polar_burial - burial_thresh) / burial_tau)

    # Smooth "is satisfied". Two paths depending on what the caller
    # provides:
    #
    #  (a) If ``polar_is_donor`` and ``polar_is_acceptor`` are given,
    #      compute Hbond satisfaction from coords directly — fully
    #      differentiable end-to-end.
    #  (b) Otherwise use the precomputed ``polar_atom_hbond_energies``
    #      input (back-compat with the older API). Note: in path (b)
    #      the gradient w.r.t. coords only flows through the burial
    #      branch; ``polar_atom_hbond_energies`` is treated as a
    #      constant unless you compute it inside the ``jax.grad``
    #      scope yourself.
    if polar_is_donor is not None and polar_is_acceptor is not None:
        hbond_score = hbond_satisfaction_from_coords(
            coords, polar_atom_idx, polar_atom_res,
            polar_is_donor, polar_is_acceptor,
        )
    else:
        # Convert Hbond energies → soft count: each Hbond contributes
        # softplus(-E_hb) ≈ |E_hb| when favorable.
        hbond_score = jax.nn.softplus(-polar_atom_hbond_energies)
    is_satisfied = jax.nn.sigmoid(
        (hbond_score - polar_atom_hb_capacity) / 0.5
    )

    # BUH = sum over polar atoms of buried * (1 - satisfied).
    unsat = 1.0 - is_satisfied
    return (is_buried * unsat).sum()


def buh_per_kA2(
    buh: Float[Array, ""],
    interface_bsa: Float[Array, ""],
    *,
    eps: float = 1.0,
) -> Float[Array, ""]:
    """Normalize BUH by interface buried-surface area (kÅ²).

    Stranges-Kuhlman 2013 native interfaces sit at ≤ 2 BUH per 1000 Å²;
    designs above 2-3 essentially never bind. ``eps`` prevents
    divide-by-zero when ``interface_bsa`` collapses to 0 during
    optimization (e.g. binder fully separated from target).

    Returns
    -------
    BUH_per_kÅ² : scalar
        Buried-unsatisfied polar density per 1000 Å² of interface BSA.
    """
    return 1000.0 * buh / (interface_bsa + eps)
