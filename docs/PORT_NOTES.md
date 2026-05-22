# Gaps and design choices vs upstream tmol

This page tracks what's ported, what's deliberately different, and the
places where jtmol gives a different number than upstream (and why).
Most of the upstream surface is in. The list of differences is short
and is documented here so you can decide whether any of them affect
your use case before you build on top.

## What's ported

- All 20 energy channels of `beta_nov2016_cart`, forward-bit-exact on
  the multi-PDB panel (see [PARITY.md](PARITY.md)).
- Gradient parity via `jax.grad`, with `@jax.custom_jvp` rules at the
  cubic-Hermite kinks (omega, cart_torsions, cart_hxltorsions) so we
  match upstream's hand-written CUDA adjoints there.
- Pose / PoseStack / PackedBlockTypes as JAX pytrees.
- ChemicalDatabase + PatchedChemicalDatabase from upstream YAMLs
  verbatim, with N-/C-terminal variant auto-detection.
- KinForest with `forward_kin` / `inverse_kin` via `lax.scan`.
- L-BFGS Cartesian and kinematic minimization with strong-Wolfe line
  search.
- FastRelax (the ramped `fa_rep` schedule wrapped around pack + min).
- Rotamer packer (direct Metropolis-MC over the jitted score
  function).
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
the affected atoms. Full reproducer and our diagnosis in
[UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md). We
kept the mathematically correct gradient; we did not replicate the
bug for parity.

### Dunbrack rotdev / semirot bicubic interpolation

Upstream uses Thévenaz–Blu–Unser bicubic B-splines for the chi
mean / sigma tables (rotdev) and the 3D (φ, ψ, χ_last) log-probability
table (semirot). jtmol currently uses bilinear interpolation. Forward
residual is < 0.4% on 1ubq, gradient max diff ~2.7e-4. Switching to
bicubic is a self-contained future improvement; if you need it for
your loop, ping us.

## Other runtime differences from upstream

- **No C++ or CUDA, anywhere.** Gradients come from `jax.grad` on the
  pure-Python forward. We only use `@jax.custom_jvp` at the
  cubic-Hermite kinks to reproduce upstream's hand-coded adjoints, not
  to invent new physics.
- **No PyTorch at runtime.** The only place torch shows up is
  `scripts/convert_binary_tables.py`, which regenerates `dunbrack.npz`
  etc. from upstream's legacy `.zip` / `.bin` files. Pre-computed
  `.npz` tables ship in the package, so most users never touch torch.
- **Generational KinForest scan is degenerate.** Upstream partitions
  the kin tree into depth-indexed generations for a GPU segmented-scan
  kernel. We use `lax.fori_loop` over the existing `parent[i] < i`
  invariant. Same result, simpler code, slower for hand-tuned
  multi-chain trees. A full `get_scans` port is present but unused.

## Out of scope (for now)

- The OpenFold / RF2 ML bridges from upstream are not ported.
- Upstream's interaction-graph cached rotamer-pair energies, with the
  DEE + precomputed-pair-energy machinery underneath upstream's
  packer. Our packer scores the whole pose each step under jit, which
  is correct but slower per Metropolis trial than upstream.

## Things that are jtmol-only additions

- **`LossTerm` wrappers for mosaic / [Boltz-2](https://github.com/jwohlwend/boltz)**.
  See [MOSAIC_INTEGRATION.md](MOSAIC_INTEGRATION.md). This is the
  reason we did the port.
- **Neighbor-list fast path** with cell-list pair enumeration (6×
  faster, 13× smaller intermediate memory vs the dense path on 1ubq).
- **`jax.vmap` over a pose-stack batch dim** for batched scoring and
  batched gradient evaluation.
