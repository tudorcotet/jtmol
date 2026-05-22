# Port Notes

One-shot, 30-parallel-agent port from `tmol` (PyTorch + C++/CUDA) to
pure JAX. Gradients come from `jax.grad` rather than hand-written
backward kernels. The following deliberate simplifications keep the
port tractable; each is recoverable later and none changes the public
API surface.

## Chemistry

- **Variant patching in `PatchedChemicalDatabase` is simplified.**
  Upstream's `do_patch` / `RestypeGraphBuilder` pipeline uses
  `networkx` + `pysmiles` graph isomorphism. We skip it and emit one
  `RefinedResidueType` per raw residue, passing `variants` through
  untouched. N-/C-terminal variants are not auto-applied.
- **`ParameterDatabase.create_stable_subset` raises
  `NotImplementedError`**; it depends on the skipped patch pipeline.
  Use `ParameterDatabase.get_default()`.

## Kinematics

- **Generational scan ordering is degenerate.** Upstream partitions
  the `KinForest` into depth-indexed generations for a GPU
  segmented-scan kernel. Our `forward_kin` / `inverse_kin` run under
  `lax.fori_loop`, which is sequential anyway, so the default emits
  one generation containing every atom in index order. Correctness is
  preserved by the `parent[i] < i` invariant. A full `get_scans` port
  is included but unused.
- **Block-type scan-path annotation is not ported.**
  `_annotate_block_type_with_gen_scan_path_segs` and the per-block
  weaving into a whole-PoseStack KinForest are future work.

## Scoring

- **Inter-residue cart-bonded terms are omitted.** Only intra-residue
  lengths / angles / torsions / impropers / hxltorsions are evaluated.
- **Dunbrack binary library parsing is skipped.** If the binary
  `.lib` is not loadable, the Dunbrack terms fall back to no-op.
  Math uses simplified bilinear/trilinear interpolation over
  pre-evaluated probability grids rather than cubic B-splines.
- **LK-ball may degrade to plain LK iso.** `place_waters_for_atom`
  places waters along the averaged lone-pair direction rather than
  upstream's sp2/sp3/ring-specific canonical torsions. The
  `lk_ball_pair` core is faithful; the water geometry is approximate.

## Packing

- **The rotamer packer is simplified Metropolis-MC.** Upstream runs
  DEE + simulated annealing with precomputed rotamer-pair energy
  tables and an incremental interaction graph in C++/CUDA. We do
  per-step whole-pose scoring under a JAX-jitted score function:
  more work per step, but pure JAX.
- **No Dunbrack rotamer library sampling.** Chi angles come from
  each residue type's static `chi_samples`; absent those, the packer
  degenerates to the identity pose.

## External bridges

- **OpenFold / RF2 / other ML bridges may stub to
  `NotImplementedError`.** Out of scope for the first port.

## Numerical fidelity

Numerical equivalence with upstream tmol has **not** been bit-for-bit
validated. The implementation matches the mathematical forms in the
C++ headers (LJ shift + cubic-Hermite fade, sigmoidal dielectric,
hbond 11th-degree bounded polynomials, cart-bonded kernels) and uses
the upstream YAML parameters verbatim; cross-validation against
PyTorch tmol is future work.
