"""Hessian-based flexibility analysis for jtmol energies.

Because jtmol's energy is differentiable in JAX, the *second* derivative
w.r.t. atomic coordinates, the Hessian, is one ``jax.hessian`` call
away. The Hessian is the curvature of the energy well, and its spectrum
carries physics that single-point scoring throws away:

* **Per-atom fluctuations / B-factors**: the pseudo-inverse of the
  Hessian is (up to ``kT``) the covariance of harmonic fluctuations. Its
  per-atom diagonal predicts crystallographic B-factors. This is the
  cheapest, best-validated output and needs no mass-weighting.
* **Vibrational entropy**: the eigenvalue spectrum gives a harmonic
  (log-frequency) configurational-entropy estimate, the ``-TΔS`` term
  that end-point ΔG methods (MM/PBSA + NMA) add on top of enthalpy.

Method (normal-mode analysis): at (or near) an energy minimum, build
``H = ∂²E/∂x²`` over the real atoms, diagonalise, drop the 6 rigid-body
modes (3 translation + 3 rotation, eigenvalue ≈ 0), and read flexibility
off the remaining ``3M-6`` modes.

References: Tirion 1996 / Atilgan 2001 (elastic network); Brooks-Karplus
1995 (protein NMA); Genheden-Ryde 2015 (NMA entropy in ΔG). JAX pattern
(forward-over-reverse Hessian, ``eigh``) follows the JAX autodiff
cookbook and JAX-MD's elasticity module.

Everything here works in float64. Enable it before use::

    import jax; jax.config.update("jax_enable_x64", True)

float32 is usable for B-factor *correlation* but the stiff/soft dynamic
range of a bio-force-field Hessian degrades eigenvalue precision.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

# Boltzmann constant in Rosetta energy units (~kcal/mol) at 298 K.
# ponytail: REU ≈ kcal/mol; absolute B-factor scale is approximate, but
# the *correlation* with experiment (the useful signal) is scale-free.
_KT_298 = 0.593  # kcal/mol
_B_PREFACTOR = 8.0 * jnp.pi**2 / 3.0  # Debye-Waller: B = 8π²/3 · <Δr²>


def real_atom_indices(pose_stack: Any, pose: int = 0) -> jax.Array:
    """Flat indices of the real (non-padding) atoms in one pose's coord slab.

    Reuses the scoring layer's ``pose_atom_block_index`` so the mask is
    identical to what the energy terms see, with no separate bookkeeping.
    """
    from jtmol.score.common.count_pair import pose_atom_block_index

    pbt = pose_stack.packed_block_types
    _, _, atom_is_real = pose_atom_block_index(
        jnp.asarray(pose_stack.block_type_ind),
        jnp.asarray(pose_stack.block_coord_offset),
        jnp.asarray(pbt.n_atoms),
        int(pose_stack.coords.shape[1]),
    )
    return jnp.where(atom_is_real[pose])[0]


def energy_hessian(
    score_function: Any,
    pose_stack: Any,
    coords: Optional[jax.Array] = None,
    pose: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Dense energy Hessian over the real atoms of one pose.

    Returns ``(H, atom_idx)`` where ``H`` is ``[3M, 3M]`` (M real atoms,
    xyz-major: rows ``[a0x, a0y, a0z, a1x, ...]``) and ``atom_idx`` maps
    the M Hessian atoms back to slots in the pose coord slab.

    Assembled column-by-column via ``lax.map`` over Hessian-vector
    products of the identity basis. This is *sequential* on purpose:
    ``jax.hessian`` (batched forward-over-reverse) would materialise all
    3M tangents at once, and because jtmol's LJ/elec terms build dense
    ``[N, N]`` pair tensors that blows up to tens of GB even for a small
    protein. Column assembly keeps peak memory at ~one energy evaluation,
    trading it for 3M sequential HVPs. For large systems where even the
    ``[3M, 3M]`` matrix is too big, use :func:`hvp` with an iterative
    eigensolver and never form ``H``.
    """
    matvec, atom_idx = hvp(score_function, pose_stack, pose, coords=coords)
    n = 3 * atom_idx.shape[0]
    # dtype must match the HVP primal (x0) or jvp rejects the tangent.
    basis = jnp.eye(n, dtype=jnp.result_type(float))
    # lax.map is sequential -> memory O(one HVP), not O(3M HVPs).
    cols = jax.lax.map(matvec, basis)  # [3M, 3M], cols[i] = H @ e_i
    H = cols.T
    return 0.5 * (H + H.T), atom_idx  # symmetrise away float asymmetry


def hvp(
    score_function: Any,
    pose_stack: Any,
    pose: int = 0,
    coords: Optional[jax.Array] = None,
) -> tuple[Any, jax.Array]:
    """Matrix-free Hessian-vector-product closure ``matvec(v)`` over real atoms.

    Returns ``(matvec, atom_idx)``. ``matvec`` takes and returns a flat
    ``[3M]`` vector and costs ~4 energy evaluations, never materialising
    the ``[3M, 3M]`` matrix. Feed it to a Lanczos/LOBPCG eigensolver for
    the low-frequency modes of large systems. Pattern: forward-over-
    reverse ``jvp(grad(E))`` (JAX autodiff cookbook).
    """
    state = score_function.setup_for_pose_stack(pose_stack)
    # Promote to the active float width (float64 when x64 is enabled) so
    # the Hessian spectrum is precise; jvp needs primal/tangent dtypes to
    # match, so the identity basis in energy_hessian follows x0.dtype.
    dtype = jnp.result_type(float)
    slab = jnp.asarray(
        coords if coords is not None else pose_stack.coords, dtype=dtype
    )[pose]
    atom_idx = real_atom_indices(pose_stack, pose)
    x0 = slab[atom_idx]

    def energy_of_real(real_xyz: jax.Array) -> jax.Array:
        full = slab.at[atom_idx].set(real_xyz)
        return score_function(
            pose_stack, state=state, coords_override=full[None]
        ).sum()

    grad_e = jax.grad(energy_of_real)

    @jax.jit
    def matvec(v_flat: jax.Array) -> jax.Array:
        v = v_flat.reshape(x0.shape)
        Hv = jax.jvp(grad_e, (x0,), (v,))[1]
        return Hv.reshape(-1)

    return matvec, atom_idx


def soft_modes(
    H: jax.Array, zero_tol: float = 1e-8, n_rigid: int = 6,
    k: Optional[int] = None,
) -> tuple[jax.Array, jax.Array]:
    """Low-frequency internal modes: the ones B-factors/entropy live on.

    Diagonalise ``H`` (ascending), drop the non-physical modes, and return
    the remaining **soft internal** modes softest-first:

    * drop everything with ``λ <= zero_tol``: the negative / saddle modes
      an imperfect minimum leaves, plus true zeros;
    * drop the next ``n_rigid`` (default 6) smallest positives: the
      rigid-body translation+rotation modes (``λ ≈ 0``), whose ``1/λ`` would
      otherwise dominate the fluctuation sum with rigid-body noise;
    * keep the rest (optionally only the softest ``k``).

    These low-frequency collective motions are what elastic-network / NMA
    B-factor predictions are built from. ``1/λ`` weights them heavily, and
    they carry the loop/terminus flexibility signal. Contrast
    :func:`stiff_modes`, which keeps the opposite (wrong) end. On 1ubq this
    selection gives Cα B-factor r=+0.68 (softest-K=10: +0.71) vs the stiff
    end's -0.23. ``eigvecs[:, j]`` is mode j (length 3M).
    """
    eigvals, eigvecs = jnp.linalg.eigh(H)  # ascending
    keep = eigvals > zero_tol
    e, V = eigvals[keep][n_rigid:], eigvecs[:, keep][:, n_rigid:]
    if k is not None:
        e, V = e[:k], V[:, :k]
    return e, V


def stiff_modes(
    H: jax.Array, rel_cutoff: float = 1e-3
) -> tuple[jax.Array, jax.Array]:
    """Eigendecompose ``H`` and keep the stiff, high-frequency modes.

    Returns ``(eigvals, eigvecs)`` for modes with ``λ > rel_cutoff · λ_max``.
    Because a bio-force-field ``λ_max`` (stiff bond stretch) is enormous, the
    cutoff sits *above* every soft collective mode, so this keeps only the
    stiff bond vibrations and the 6 rigid-body + negative modes fall out for
    free.

    WARNING: this is the **wrong end for B-factors / fluctuations**. Those
    are dominated by the soft low-frequency modes (see :func:`soft_modes`),
    which this discards. Kept for spectral diagnostics and as the explicit
    contrast in the B-factor benchmark. ``eigvecs[:, k]`` is mode k (3M).
    """
    eigvals, eigvecs = jnp.linalg.eigh(H)  # ascending
    cut = rel_cutoff * jnp.max(eigvals)
    keep = eigvals > cut
    return eigvals[keep], eigvecs[:, keep]


def bfactors(
    H: jax.Array, temperature: float = 298.0, k: Optional[int] = None,
) -> jax.Array:
    """Predict per-atom B-factors from the Hessian pseudo-inverse.

    ``<Δr_i²> = kT · Σ_k (1/λ_k) |u_k(i)|²`` over the **soft internal** modes
    (:func:`soft_modes`); ``B_i = 8π²/3 · <Δr_i²>``. Returns ``[M]`` (Å²).
    The softest modes dominate: the large-amplitude collective motions that
    make loops and termini flexible. Pass ``k`` to use only the softest ``k``
    modes (on 1ubq, k≈10 edges out all-modes: Cα r +0.71 vs +0.68).

    Summing the *stiff* modes instead (the wrong end of the spectrum)
    anticorrelates with the crystallographic values, r≈-0.23. See :func:`soft_modes`.
    """
    eigvals, eigvecs = soft_modes(H, k=k)
    kT = _KT_298 * (temperature / 298.0)
    inv = 1.0 / eigvals  # [K], all > 0 by construction
    M = H.shape[0] // 3
    contrib = (eigvecs**2) * inv[None, :]  # [3M, K]
    msf = contrib.sum(axis=1).reshape(M, 3).sum(axis=1) * kT  # [M] <Δr²>
    return _B_PREFACTOR * msf


def vibrational_entropy(H: jax.Array) -> jax.Array:
    """Relative harmonic vibrational entropy from the Hessian spectrum.

    Classical (high-T) harmonic form, unit masses, additive constants
    dropped::

        S / k_B  ≈  Σ_k (1 - ½ ln λ_k)   over the soft internal modes

    Sums the physical (non-rigid, positive) modes from :func:`soft_modes`.
    Only *differences* (ΔS between conformers / mutants) are meaningful. The
    absolute value carries the dropped mass and ``ħ/kT`` constants.
    Mass-weight ``H`` beforehand for absolute entropies.
    """
    eigvals, _ = soft_modes(H)
    return jnp.sum(1.0 - 0.5 * jnp.log(eigvals))
