# jtmol — Claude Code notes

This repository is **jtmol**: a pure-JAX port of [tmol](https://github.com/uw-ipd/tmol),
the GPU-accelerated reimplementation of Rosetta's `beta_nov2016_cart` energy
function originally written on top of PyTorch with custom C++/CUDA kernels.

The Python package is still named `tmol_jax` (the import path) for historical
reasons; the GitHub project / public-facing name is **jtmol**.

## Working in this repo

- **No C++/CUDA extensions.** Everything is pure Python/JAX. Math uses
  `jit`, `vmap`, `lax.scan`, and `jax.grad` for autodiff. Do not introduce
  pybind11 / cython / custom ops.
- **Upstream reference**: the original `tmol/` repo is cloned into the
  repository root for side-by-side comparison. It is **gitignored** (`/tmol/`)
  and must not be committed — it is multi-GB and has its own `.git`.
- **Numerical parity** with upstream tmol is the primary correctness signal.
  Term-by-term parity status lives in `docs/PARITY.md` and
  `docs/UPSTREAM_PARITY.md`. A known upstream bug we replicate is documented
  in `docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`.
- **YAML parameters** under `tmol_jax/database/default/` are copied verbatim
  from upstream tmol. Do not hand-edit them — regenerate via `scripts/`
  if upstream changes.
- **Binary scoring tables** (`dunbrack.bin`, `dunbrack.npz`, `rama.zip`,
  `omega_bbdep.zip`) are tracked because they are required at import time
  and not trivially regenerable without a torch install. Keep them in-tree.

## Layout (`tmol_jax/`)

- `chemical/` — restypes, all_bonds, ideal_coords, patched_chemdb
- `database/` — ChemicalDatabase + ScoringDatabase YAML loaders
- `pose/` — `Pose`, `PoseStack`, `PackedBlockTypes` (JAX pytrees)
- `kinematics/` — `KinForest`, `KinDOF`, forward/inverse kinematics via `lax.scan`
- `score/` — all energy terms (`ljlk`, `elec`, `hbond`, `lk_ball`, `dunbrack`,
  `backbone_torsion`, `cartbonded`, `disulfide`, `ref`, `constraint`, `common`)
- `io/` — PDB read/write, canonical ordering
- `optimization/` — L-BFGS (Cartesian + kinematic)
- `pack/` — rotamer packer
- `relax/` — FastRelax protocol
- `tests/` — end-to-end smoke + gradcheck

## Useful entry points

```python
import tmol_jax

ps = tmol_jax.pose_stack_from_pdb("1ubq.pdb")
sfxn = tmol_jax.beta2016_score_function()
E = sfxn(ps)
grad = tmol_jax.grad_wrt_coords(sfxn)(ps)
ps_min = tmol_jax.run_cart_min(ps, sfxn)
tmol_jax.write_pose_stack_pdb(ps_min, "out.pdb")
```

## Don't

- Don't commit `tmol/` (the upstream clone), `scratch_pdbs/`, build artifacts,
  or anything > 1 MB that is regenerable (see `.gitignore`).
- Don't add a torch runtime dependency. Torch is only used in
  `scripts/convert_binary_tables.py` to regenerate `.npz` tables.
- Don't reorder atoms or change residue type definitions without updating
  parity tests — atom ordering is load-bearing for the kinematic forest.
