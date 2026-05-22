# Gaps and design choices vs upstream tmol

Most of the upstream tmol surface is ported. This page tracks the
deliberate omissions, behavioural differences, and places where jtmol
gives a different answer than upstream (and why).

## What is fully ported

- All 20 energy channels of `beta_nov2016_cart` — forward bit-exact on
  the multi-PDB panel (see [PARITY.md](PARITY.md))
- Gradient parity via `jax.grad`, with custom-JVP rules at the cubic-
  Hermite kinks (omega, cart_torsions, cart_hxltorsions) so we match
  upstream's hand-written CUDA adjoints
- Pose / PoseStack / PackedBlockTypes (JAX pytrees)
- ChemicalDatabase + PatchedChemicalDatabase (verbatim YAML import,
  N-/C-terminal variant auto-detection)
- KinForest + forward_kin / inverse_kin via `lax.scan`
- L-BFGS Cartesian and kinematic minimization (strong-Wolfe line
  search)
- FastRelax (ramped fa_rep schedule wrapping pack + min)
- Rotamer packer (direct Metropolis-MC over jitted score function)
- PDB read/write, canonical atom ordering
- Inter-residue cart_bonded subgraphs (lengths, angles, torsions,
  impropers, hxltorsions across peptide bonds)
- Exact Floyd-Warshall APSP on the residue-connection graph for
  count-pair (`path_length > 4 → 1.0`, `== 4 → 0.2`, `< 4 → 0`)

## Where jtmol gives a different answer

### `lk_bridge_uncpl` gradient at sp2 backbone-O acceptors

jtmol's gradient matches central finite-difference of upstream's own
forward to ~1e-9. Upstream's hand-written CUDA adjoint
(`LKBallPoseScoreOp::backward`) disagrees by up to ~0.16 kcal/mol/Å at
the affected atoms. Full reproducer in
[UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md). We
keep the mathematically correct gradient.

### Dunbrack rotdev / semirot bicubic interpolation

Upstream uses Thévenaz–Blu–Unser bicubic B-splines for the chi
mean/sigma tables (rotdev) and the 3D (φ, ψ, χ_last) log-probability
table (semirot). jtmol uses bilinear interpolation. Forward residual is
< 0.4% on 1ubq; gradient max|dg| ≈ 2.7e-4. Switching to bicubic is a
self-contained future improvement.

## Where the rest of the runtime is intentionally different from upstream

- **No C++/CUDA extensions, anywhere.** Gradients come from `jax.grad`
  on the pure-Python forward. Custom JVPs only at cubic-Hermite kinks
  for parity with upstream's hand-coded adjoints — never to invent new
  physics.
- **No PyTorch at runtime.** Only `scripts/convert_binary_tables.py`
  uses torch, to regenerate `dunbrack.npz` etc. from upstream's legacy
  `.zip`/`.bin`. Pre-computed `.npz` tables ship in the package.
- **Generational KinForest scan is degenerate.** Upstream partitions
  the kin tree into depth-indexed generations for a GPU segmented-scan
  kernel. We use `lax.fori_loop` over the existing `parent[i] < i`
  invariant — same result, simpler code, slightly slower for
  hand-tuned multi-kchain trees. A full `get_scans` port is present
  but unused.

## Out of scope (for now)

- OpenFold / RF2 ML bridges from upstream are not ported.
- Upstream's interaction-graph cached rotamer-pair energies (the DEE +
  precomputed-pair-energy machinery underneath upstream's packer). Our
  packer scores the whole pose each step under jit — correct but
  slower per Metropolis trial than upstream.

## Things that are *not* ports but additions

- **Per-term `LossTerm` wrappers** for mosaic /
  [Boltz-2](https://github.com/jwohlwend/boltz) protein hallucination —
  see [MOSAIC_INTEGRATION.md](MOSAIC_INTEGRATION.md).
- **Neighbor-list fast path** with cell-list pair enumeration
  (6× faster, 13× smaller memory vs dense on 1ubq).
- **`jax.vmap` over a pose-stack batch dim** for batched scoring /
  gradient evaluation.
