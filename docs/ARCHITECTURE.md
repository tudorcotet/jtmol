# Architecture

`tmol_jax` is the pure-JAX port of upstream [tmol](https://github.com/uw-ipd/tmol).
PyTorch and the custom C++/CUDA kernels are replaced with `jit`,
`vmap`, `lax.scan`, and `jax.grad`. There are no C/C++ extensions and
no hand-written backward kernels; gradients come from `jax.grad` on
the forward.

## Package layout

```
tmol_jax/
  types/        pytree helpers (@pytree_dataclass over attrs classes)
  database/     ChemicalDatabase + ScoringDatabase YAML loaders
  chemical/     RefinedResidueType, ideal_coords, all_bonds, patched_chemdb
  pose/         Pose, PoseStack, PackedBlockTypes, PoseStackBuilder
  kinematics/   KinForest / KinDOF, forward_kin / inverse_kin, FoldForest
  score/        beta_nov2016_cart energy terms
  io/           PDB parsing and writing
  optimization/ L-BFGS minimizer + strong-Wolfe line search
  pack/         Metropolis-MC rotamer packer
  relax/        FastRelax protocol
```

### `tmol_jax.types`

`@pytree_dataclass` wraps an `attrs.define` class and registers it with
`jax.tree_util`. Array fields are children (leaves); everything else is
static aux. This is the one mechanism that lets every container below
flow transparently through `jit`, `vmap`, `grad`, and `lax.scan`.

### `tmol_jax.database`

`ParameterDatabase = (ChemicalDatabase, PatchedChemicalDatabase,
ScoringDatabase)`, loaded from the YAML tree in `database/default/`
(verbatim from upstream). `get_default()` memoises the parse.
`ScoringDatabase` bundles per-term YAMLs (LJLK, hbond, elec, lk_ball,
dunbrack, rama, cartbonded, disulfide, ref).

### `tmol_jax.chemical`

`RefinedResidueType` extends the raw YAML record with derived lookup
tables (bond lists, path distances via `scipy.sparse.csgraph`, ideal
coords built from internal coords).

### `tmol_jax.pose`

`PoseStack` is a frozen pytree of `coords`, `block_coord_offset`,
`inter_residue_connections`, `inter_block_bondsep`, and
`block_type_ind` with a `PackedBlockTypes` palette. Immutable: every
update returns a new `PoseStack`. `PackedBlockTypes` flattens the
residue-type palette into block-indexed arrays and is itself a pytree.

### `tmol_jax.kinematics`

`KinForest` and `KinDOF` are pytrees. `forward_kin` composes per-atom
4x4 homogeneous transforms down the tree with `lax.fori_loop`; the
forest's `parent[i] < i` invariant makes a single linear pass
sufficient. `inverse_kin` is `vmap`-parallel across atoms. Gradients
flow via `jax.grad`; no Abe-Go adjoints.

### `tmol_jax.score`

`ScoreFunction` holds weights and lazily imports terms. Each term
implements `setup_for_pose_stack` and `__call__`. Subpackages: `ljlk`,
`elec`, `hbond`, `lk_ball`, `dunbrack`, `backbone_torsion` (rama +
omega), `cartbonded`, `disulfide`, `ref`, `constraint`.
`beta2016_score_function()` reproduces the 21-channel weighting from
upstream `_non_memoized_beta2016`.

### `tmol_jax.io`, `optimization`, `pack`, `relax`

PDB read/write (biotite + direct parser fallback); L-BFGS with
strong-Wolfe line search; Metropolis-MC rotamer packer; FastRelax
ramped `fa_rep` schedule wrapping pack + min.

## Why every container is a pytree

PoseStack, PackedBlockTypes, KinForest, KinDOF, and all per-term state
bundles are registered pytrees. This is the single structural
commitment that makes `jit`/`vmap`/`grad`/`scan` compose over the
whole stack without bespoke tracing logic.
