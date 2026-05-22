# jtmol

**jtmol** is a pure-JAX port of [**tmol**](https://github.com/uw-ipd/tmol) —
the GPU-accelerated reimplementation of Rosetta's `beta_nov2016_cart` energy
function originally built on PyTorch with custom C++/CUDA kernels.

This port replaces PyTorch + custom kernels with idiomatic JAX: `jit`, `vmap`,
`lax.scan`, and `jax.grad`. All numerical math is written in pure Python/JAX,
so there are **no C++/CUDA extensions to build** — installation is just one
command. Gradients come from `jax.grad` rather than hand-written backward
kernels.

The Python import path is `tmol_jax` (kept for historical and parity reasons);
the public project name is **jtmol**.

## Credit / upstream

All physics, parameters, residue-type definitions, scoring database YAMLs, and
rotamer libraries come **verbatim from upstream tmol** by the
[Baker Lab / UW Institute for Protein Design](https://www.ipd.uw.edu/).
This repository is a JAX re-implementation of their work — please cite
upstream tmol if you use jtmol in published work:

- Upstream: **<https://github.com/uw-ipd/tmol>**
- Rosetta `beta_nov2016` energy function: Park et al.,
  *J. Chem. Theory Comput.* 2016, 12, 12, 6201–6212.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10 and JAX ≥ 0.4.30. No PyTorch needed at runtime
(only optional, for regenerating binary scoring tables).

## Quick start

```python
import tmol_jax

# Load a structure
ps = tmol_jax.pose_stack_from_pdb("1ubq.pdb")

# Build the beta_nov2016_cart score function
sfxn = tmol_jax.beta2016_score_function()

# Score
E = sfxn(ps)                                  # total energy (scalar)
breakdown = tmol_jax.score_by_term(sfxn, ps)  # per-term dict

# Differentiate w.r.t. atomic coordinates
grad = tmol_jax.grad_wrt_coords(sfxn)(ps)     # ∂E/∂xyz

# Cartesian L-BFGS minimization
ps_min = tmol_jax.run_cart_min(ps, sfxn)
tmol_jax.write_pose_stack_pdb(ps_min, "out.pdb")
```

## Scope

- **Pose**: `Pose`, `PoseStack`, `PackedBlockTypes` as JAX pytrees.
- **Chemistry**: `ChemicalDatabase` loaded from tmol's YAML files verbatim.
- **Kinematics**: `KinForest` + `forward_kin` / `inverse_kin` via `lax.scan`.
- **Scoring** (`beta_nov2016_cart`): LJ (atr / rep), LK isotropic, LK-ball,
  electrostatics (sigmoidal dielectric), hbond (sp2 / sp3 / ring),
  Ramachandran, Dunbrack rotamers, omega, cart bonded, disulfide,
  reference energies.
- **Minimization**: L-BFGS (Cartesian and kinematic).
- **IO**: PDB read/write, canonical ordering.
- **Relax / Pack**: FastRelax protocol and rotamer packer.

## Documentation

- **[docs/PARITY.md](docs/PARITY.md)** — term-by-term numerical parity status
  vs. upstream tmol.
- **[docs/UPSTREAM_PARITY.md](docs/UPSTREAM_PARITY.md)** — methodology and
  results for direct float32 comparison against the upstream PyTorch
  implementation.
- **[docs/WHY_JAX.md](docs/WHY_JAX.md)** — design rationale for the JAX port:
  why no C++/CUDA, what `jit` + `vmap` + `lax.scan` buy us, and where the
  performance comes from.
- **[docs/MOSAIC_INTEGRATION.md](docs/MOSAIC_INTEGRATION.md)** — embedding
  jtmol as a scoring backend inside the Mosaic design framework.
- **[docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md)** —
  a known sign / unit bug in the upstream LK-bridge-uncoupled term that we
  intentionally replicate for bit-for-bit parity.

Additional notes: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`PHYSICS.md`](docs/PHYSICS.md), [`PORT_NOTES.md`](docs/PORT_NOTES.md),
[`MULTI_CHAIN_PARITY.md`](docs/MULTI_CHAIN_PARITY.md),
[`MULTI_PDB_PARITY.md`](docs/MULTI_PDB_PARITY.md).

## Status

Alpha. Numerical equivalence with upstream tmol is being validated term-by-term
(see `docs/UPSTREAM_PARITY.md`). The mathematical forms follow the upstream
C++ headers; the numerical parameters come from the verbatim upstream YAMLs.

## License

Apache-2.0, matching upstream tmol. See [`LICENSE`](LICENSE).
