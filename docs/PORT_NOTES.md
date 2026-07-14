# Gaps and design choices vs upstream tmol

This page tracks what is ported, what is deliberately different, and the
places where jtmol gives a different number than upstream (and why). The
list of differences is short and is documented here so you can decide
whether any of them affect your use case before building on top.

## Current status and known gaps

- Current upstream beta2016 has 22 active weighted channels. Per-term
  forward and gradient parity is tracked in [PARITY.md](PARITY.md).
- The default chemical database is not a verbatim representation port:
  the underlying YAML parameters are copied from upstream unchanged, but
  the constructed representation omits typed bond/ring/connection
  metadata.
- 29 of 128 upstream-selected structures fail native-coordinate
  reconstruction at 1e-4 Å.

## What's ported

- The weighted energy channels of `beta_nov2016_cart` (per-term forward
  and gradient parity on the multi-PDB panel is tracked in
  [PARITY.md](PARITY.md)).
- Gradient parity via `jax.grad`, with `@jax.custom_jvp` rules at the
  cubic-Hermite kinks (omega, cart_torsions, cart_hxltorsions) to match
  upstream's hand-written CUDA adjoints there.
- Pose / PoseStack / PackedBlockTypes as JAX pytrees.
- ChemicalDatabase + PatchedChemicalDatabase built from upstream YAML
  parameters, with N-/C-terminal variant auto-detection.
- KinForest with `forward_kin` / `inverse_kin` via `lax.scan`.
- L-BFGS Cartesian and kinematic minimization with strong-Wolfe line
  search.
- FastRelax (the ramped `fa_rep` schedule wrapped around pack + min).
- Rotamer packer (direct Metropolis-MC over the jitted score function).
- PDB read/write with canonical atom ordering.
- Inter-residue `cart_bonded` subgraphs (lengths, angles, torsions,
  impropers, hxltorsions across peptide bonds).
- Exact Floyd-Warshall APSP on the residue-connection graph for
  count-pair (`path_length > 4 → 1.0`, `== 4 → 0.2`, `< 4 → 0`).

## Where jtmol gives a different answer

### `lk_bridge_uncpl` gradient at sp2 backbone-O acceptors

jtmol's gradient matches central finite-difference of upstream's own
forward to ~1e-9. Upstream's hand-written CUDA adjoint
(`LKBallPoseScoreOp::backward`) disagrees by up to 0.16 kcal/mol/Å at
the affected atoms. Full reproducer and diagnosis in
[UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).
jtmol keeps the mathematically correct gradient and does not replicate
the bug for parity.

### Dunbrack semirot: trilinear vs upstream tricubic

The rotameric chi mean / sigma tables (rotdev) go through the same
Thévenaz–Blu–Unser bicubic B-spline as upstream (via
`_bicubic_bspline_interpolate` over bspline-prefiltered coefficients),
so rotdev is not an interpolation gap; its ~2.7e-4 gradient residual on
1ubq is the shared float32 kink-adjoint floor, not an
interpolation-scheme difference. The 3D (φ, ψ, χ_last) semirotameric
log-probability table (semirot) still uses trilinear interpolation
where upstream uses a tricubic B-spline. It forward-matches on the
panel; moving semirot to tricubic is a self-contained future
improvement.

## Other runtime differences from upstream

- **No C++ or CUDA, anywhere.** Gradients come from `jax.grad` on the
  pure-Python forward. jtmol uses `@jax.custom_jvp` only at the
  cubic-Hermite kinks, to reproduce upstream's hand-coded adjoints.
- **No PyTorch at runtime.** The only place torch shows up is
  `scripts/convert_binary_tables.py`, which regenerates `dunbrack.npz`
  etc. from upstream's legacy `.zip` / `.bin` files. Pre-computed
  `.npz` tables ship in the package, so most users never touch torch.
- **Generational KinForest scan is degenerate.** Upstream partitions
  the kin tree into depth-indexed generations for a GPU segmented-scan
  kernel. jtmol uses `lax.fori_loop` over the existing `parent[i] < i`
  invariant: same result, simpler code, slower for hand-tuned
  multi-chain trees. A full `get_scans` port is present but unused.

## Out of scope

- The OpenFold / RF2 ML bridges from upstream are not ported.
- Upstream's interaction-graph cached rotamer-pair energies, with the
  DEE + precomputed-pair-energy machinery underneath upstream's packer.
  jtmol's packer scores the whole pose each step under jit, which is
  correct but slower per Metropolis trial than upstream.

## jtmol-only additions

- **Neighbor-list fast path** with cell-list pair enumeration (6×
  faster, 13× smaller intermediate memory vs the dense path on 1ubq).
- **`jax.vmap` over a pose-stack batch dim** for batched scoring and
  batched gradient evaluation.
